#!/usr/bin/env python3
"""Timeout and holding-risk label diagnostics for source-quality candidates.

This is a diagnostic-only pre-training screen. It builds timeout, holding, and
time-to-resolution targets from realized outcomes, calibrating holding-time
thresholds from prior months only. It then trains small month-forward smoke
models from causal feature-store/source features and reports whether these
targets are stable, learnable, and economically usable as filters or penalties.

No production training, Optuna, policy geometry, or inference artifact is
modified.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_label_feature_store_model_smoke import _fit_predict, _month_model_frame  # noqa: E402
from scripts.run_label_quality_proxy_diagnostics import (  # noqa: E402
    DEFAULT_FEATURE_DIR,
    DEFAULT_FEATURE_LIST_CSV,
    DEFAULT_LABELS_DIR,
    _json_safe,
    _load_feature_store_columns,
    _path_metrics,
    _read_feature_list,
    _safe_mean,
    _safe_quantile,
    _spearman,
)
from scripts.run_source_quality_label_walkforward_ablation import (  # noqa: E402
    DEFAULT_MONTHS,
    DEFAULT_QUALITY_LABELS,
    DEFAULT_SEEDS,
    _load_joined_frame,
    _parse_csv,
    _parse_float_csv,
    _parse_int_csv,
    _source_feature_columns,
)


DEFAULT_CONFIG = Path("configs/source_quality_labels.yaml")
DEFAULT_OUTPUT_DIR = Path(
    "data_perp/reports/source_quality_label_walkforward_ablation_v1_sideaware_20260702/timeout_holding_risk_stage1_weekaware_v1"
)
DEFAULT_LABELS = (
    "timeout_risk_v1",
    "holding_risk_v1",
    "time_to_resolution_bucket_v1",
)
DEFAULT_FEATURE_SETS = ("base", "base_plus_source")
DEFAULT_HIGH_RISK_TOP_FRACS = (0.01, 0.03, 0.05, 0.10)
DEFAULT_LOW_RISK_KEEP_FRACS = (0.50, 0.65, 0.80)


@dataclass(frozen=True)
class TargetSpec:
    name: str
    kind: str


TARGET_SPECS = (
    TargetSpec("timeout_risk_v1", "timeout"),
    TargetSpec("holding_risk_v1", "holding"),
    TargetSpec("time_to_resolution_bucket_v1", "resolution_bucket"),
    TargetSpec("timeout_event_v1", "timeout"),
    TargetSpec("holding_time_exceeded_v1", "holding"),
    TargetSpec("slow_resolution_v1", "slow_resolution"),
    TargetSpec("late_exit_v1", "late_exit"),
    TargetSpec("early_progress_fail_v1", "early_progress_fail"),
    TargetSpec("timeout_or_low_progress_v1", "timeout_or_low_progress"),
)


def _target_specs_by_name(names: list[str]) -> list[TargetSpec]:
    available = {spec.name: spec for spec in TARGET_SPECS}
    missing = sorted(set(names) - set(available))
    if missing:
        raise ValueError(f"Unknown timeout/holding label(s): {missing}; available={sorted(available)}")
    return [available[name] for name in names]


def _safe_numeric(values: Any) -> pd.Series:
    if isinstance(values, pd.Series):
        return pd.to_numeric(values, errors="coerce")
    return pd.to_numeric(pd.Series(values), errors="coerce")


def _safe_auc(y_true: Any, score: Any) -> float:
    y = _safe_numeric(y_true)
    s = _safe_numeric(score)
    mask = y.notna() & s.notna()
    if int(mask.sum()) < 10 or y[mask].nunique(dropna=True) < 2:
        return float("nan")
    try:
        from sklearn.metrics import roc_auc_score

        return float(roc_auc_score(y[mask].astype(int), s[mask]))
    except Exception:
        return float("nan")


def _safe_average_precision(y_true: Any, score: Any) -> float:
    y = _safe_numeric(y_true)
    s = _safe_numeric(score)
    mask = y.notna() & s.notna()
    if int(mask.sum()) < 10 or y[mask].nunique(dropna=True) < 2:
        return float("nan")
    try:
        from sklearn.metrics import average_precision_score

        return float(average_precision_score(y[mask].astype(int), s[mask]))
    except Exception:
        return float("nan")


def _safe_brier(y_true: Any, score: Any) -> float:
    y = _safe_numeric(y_true)
    s = _safe_numeric(score).clip(0.0, 1.0)
    mask = y.notna() & s.notna()
    if int(mask.sum()) < 10:
        return float("nan")
    try:
        from sklearn.metrics import brier_score_loss

        return float(brier_score_loss(y[mask].astype(int), s[mask]))
    except Exception:
        return float("nan")


def _effective_n(values: Any) -> float:
    counts = pd.Series(values, dtype=object).value_counts(dropna=False)
    if counts.empty:
        return 0.0
    shares = counts.to_numpy(dtype=np.float64) / float(counts.sum())
    denom = float(np.sum(shares * shares))
    return 1.0 / denom if denom > 0.0 else 0.0


def _rank_fraction_indices(score: pd.Series, frac: float, *, highest: bool) -> np.ndarray:
    score_s = _safe_numeric(score).reset_index(drop=True)
    valid = score_s.notna().to_numpy()
    if not bool(valid.any()):
        return np.array([], dtype=np.int64)
    valid_idx = np.flatnonzero(valid)
    k = min(max(1, int(math.ceil(float(frac) * len(valid_idx)))), len(valid_idx))
    values = score_s.iloc[valid_idx].to_numpy(dtype=np.float64)
    order = np.argsort(-values if highest else values, kind="mergesort")
    return valid_idx[order[:k]].astype(np.int64, copy=False)


def _rank_fraction_by_group_indices(score: pd.Series, groups: pd.Series, frac: float, *, highest: bool) -> np.ndarray:
    score_s = _safe_numeric(score).reset_index(drop=True)
    group_s = pd.Series(groups).reset_index(drop=True).astype(str)
    selected: list[np.ndarray] = []
    for _, group_idx in group_s.groupby(group_s, dropna=False, observed=True).groups.items():
        idx = np.asarray(list(group_idx), dtype=np.int64)
        valid = score_s.iloc[idx].notna().to_numpy()
        if not bool(valid.any()):
            continue
        valid_idx = idx[valid]
        k = min(max(1, int(math.ceil(float(frac) * len(valid_idx)))), len(valid_idx))
        values = score_s.iloc[valid_idx].to_numpy(dtype=np.float64)
        order = np.argsort(-values if highest else values, kind="mergesort")
        selected.append(valid_idx[order[:k]].astype(np.int64, copy=False))
    if not selected:
        return np.array([], dtype=np.int64)
    return np.unique(np.concatenate(selected)).astype(np.int64, copy=False)


def _week_start(ts: pd.Series) -> pd.Series:
    return (
        pd.to_datetime(ts, utc=True, errors="coerce")
        .dt.tz_convert(None)
        .dt.to_period("W-SUN")
        .apply(lambda value: value.start_time.date().isoformat() if pd.notna(value) else "")
    )


def _thresholds(metrics: pd.DataFrame, train_mask: pd.Series) -> dict[str, float]:
    bars = _safe_numeric(metrics["bars_policy"]).replace([np.inf, -np.inf], np.nan)
    train_bars = bars.loc[train_mask].dropna()
    if train_bars.empty:
        train_bars = bars.dropna()
    out: dict[str, float] = {}
    for q in (0.40, 0.50, 0.80, 0.90):
        out[f"bars_q{int(q * 100):02d}"] = float(train_bars.quantile(q)) if len(train_bars) else float("nan")
    return out


def _balanced_weights(target_hard: pd.Series, train_mask: pd.Series) -> pd.Series:
    hard = _safe_numeric(target_hard).fillna(0.0).clip(0.0, 1.0)
    train = hard.loc[train_mask].dropna()
    prevalence = float(train.mean()) if len(train) else 0.0
    if prevalence <= 0.0 or prevalence >= 1.0:
        return pd.Series(1.0, index=target_hard.index, dtype=np.float32)
    pos_w = min(5.0, 0.5 / max(prevalence, 1e-6))
    neg_w = min(5.0, 0.5 / max(1.0 - prevalence, 1e-6))
    weights = hard.map({1.0: pos_w, 0.0: neg_w}).fillna(1.0)
    return weights.astype(np.float32)


def _build_target(
    *,
    metrics: pd.DataFrame,
    train_mask: pd.Series,
    spec: TargetSpec,
) -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    threshold_map = _thresholds(metrics, train_mask)
    bars = _safe_numeric(metrics["bars_policy"]).replace([np.inf, -np.inf], np.nan).fillna(24.0).clip(lower=0.0)
    timeout = metrics["is_timeout"].astype(float).fillna(0.0).clip(0.0, 1.0)
    utility = _safe_numeric(metrics["u_policy_net"]).fillna(0.0)
    mfe_norm = _safe_numeric(metrics.get("mfe_norm", pd.Series(0.0, index=metrics.index))).fillna(0.0).clip(lower=0.0)
    bars_to_mfe = (
        _safe_numeric(metrics.get("bars_to_mfe", pd.Series(np.nan, index=metrics.index)))
        .fillna(bars)
        .clip(lower=0.0)
    )
    q40 = float(threshold_map.get("bars_q40", 8.0))
    q50 = float(threshold_map.get("bars_q50", 12.0))
    q80 = float(threshold_map.get("bars_q80", 24.0))
    q90 = float(threshold_map.get("bars_q90", max(q80, 24.0)))
    if not math.isfinite(q40):
        q40 = 8.0
    if not math.isfinite(q50):
        q50 = max(q40, 12.0)
    if not math.isfinite(q80):
        q80 = max(q50, 24.0)
    if not math.isfinite(q90):
        q90 = max(q80, 24.0)
    span = max(q90 - q40, 1.0)

    if spec.kind == "timeout":
        target_soft = timeout.copy()
        target_hard = timeout.gt(0.5).astype(float)
        bucket = pd.Series(np.where(target_hard.gt(0.5), "timeout", "resolved"), index=metrics.index)
    elif spec.kind == "holding":
        target_soft = ((bars - q40) / span).clip(0.0, 1.0)
        target_soft = target_soft.where(timeout.le(0.5), 1.0)
        target_hard = (bars >= q80).astype(float).where(timeout.le(0.5), 1.0)
        bucket = pd.Series(np.where(target_hard.gt(0.5), "slow_or_timeout", "within_budget"), index=metrics.index)
    elif spec.kind == "resolution_bucket":
        is_fast = bars <= q40
        is_slow = bars >= q80
        positive = utility > 0.0
        bucket = pd.Series("slow_negative", index=metrics.index, dtype=object)
        bucket[is_fast & positive & timeout.le(0.5)] = "fast_positive"
        bucket[is_fast & ~positive & timeout.le(0.5)] = "fast_negative"
        bucket[~is_fast & positive & timeout.le(0.5)] = "slow_positive"
        bucket[(is_slow | ~positive) & timeout.le(0.5) & ~positive] = "slow_negative"
        bucket[timeout.gt(0.5)] = "timeout"
        bucket_score = {
            "fast_positive": 0.10,
            "fast_negative": 0.35,
            "slow_positive": 0.65,
            "slow_negative": 0.90,
            "timeout": 1.00,
        }
        target_soft = bucket.map(bucket_score).astype(float)
        target_hard = bucket.isin({"slow_negative", "timeout"}).astype(float)
    elif spec.kind == "slow_resolution":
        slow_score = ((bars - q50) / max(q90 - q50, 1.0)).clip(0.0, 1.0)
        target_soft = slow_score.where(timeout.le(0.5), 1.0)
        target_hard = (bars >= q80).astype(float).where(timeout.le(0.5), 1.0)
        bucket = pd.Series(np.where(target_hard.gt(0.5), "slow_or_timeout", "resolved_in_budget"), index=metrics.index)
    elif spec.kind == "late_exit":
        slow_score = ((bars - q50) / max(q90 - q50, 1.0)).clip(0.0, 1.0)
        losing_or_flat = utility <= 0.0
        target_soft = np.maximum(slow_score, losing_or_flat.astype(float) * 0.5)
        target_soft = pd.Series(target_soft, index=metrics.index).where(timeout.le(0.5), 1.0)
        target_hard = ((bars >= q80) & losing_or_flat).astype(float).where(timeout.le(0.5), 1.0)
        bucket = pd.Series(np.where(target_hard.gt(0.5), "late_bad_exit_or_timeout", "not_late_bad_exit"), index=metrics.index)
    elif spec.kind == "early_progress_fail":
        insufficient_progress = mfe_norm < 0.25
        late_first_favorable = bars_to_mfe > q50
        progress_gap = (1.0 - (mfe_norm / 0.50)).clip(0.0, 1.0)
        target_soft = np.maximum(progress_gap, ((bars_to_mfe - q40) / max(q80 - q40, 1.0)).clip(0.0, 1.0))
        target_soft = pd.Series(target_soft, index=metrics.index).where(timeout.le(0.5), 1.0)
        target_hard = (insufficient_progress | late_first_favorable).astype(float).where(timeout.le(0.5), 1.0)
        bucket = pd.Series(np.where(target_hard.gt(0.5), "early_progress_failed", "early_progress_ok"), index=metrics.index)
    elif spec.kind == "timeout_or_low_progress":
        insufficient_progress = mfe_norm < 0.50
        slow_without_progress = insufficient_progress & (bars >= q50)
        progress_gap = (1.0 - (mfe_norm / 0.75)).clip(0.0, 1.0)
        time_pressure = ((bars - q40) / max(q90 - q40, 1.0)).clip(0.0, 1.0)
        target_soft = pd.Series(np.maximum(progress_gap, time_pressure * 0.75), index=metrics.index)
        target_soft = target_soft.where(timeout.le(0.5), 1.0)
        target_hard = (timeout.gt(0.5) | slow_without_progress).astype(float)
        bucket = pd.Series(np.where(timeout.gt(0.5), "timeout", "not_timeout"), index=metrics.index)
        bucket.loc[slow_without_progress & timeout.le(0.5)] = "low_progress"
    else:
        raise ValueError(f"Unsupported target kind: {spec.kind}")

    target = pd.DataFrame(
        {
            "target_soft": _safe_numeric(target_soft).clip(0.0, 1.0),
            "target_hard": _safe_numeric(target_hard).clip(0.0, 1.0),
            "target_bucket": bucket.astype(str),
        },
        index=metrics.index,
    )
    weights = _balanced_weights(target["target_hard"], train_mask)
    report = {
        "label": spec.name,
        "kind": spec.kind,
        "bars_q40": q40,
        "bars_q50": q50,
        "bars_q80": q80,
        "bars_q90": q90,
        "train_target_hard_rate": _safe_mean(target.loc[train_mask, "target_hard"]),
        "train_target_soft_mean": _safe_mean(target.loc[train_mask, "target_soft"]),
        "train_timeout_rate": _safe_mean(timeout.loc[train_mask]),
        "train_mean_bars_policy": _safe_mean(bars.loc[train_mask]),
    }
    for bucket_name, count in target.loc[train_mask, "target_bucket"].value_counts(normalize=True).items():
        report[f"train_bucket_rate_{bucket_name}"] = float(count)
    return target, weights, report


def _summary(
    *,
    frame: pd.DataFrame,
    metrics: pd.DataFrame,
    target: pd.DataFrame,
    selected_idx: np.ndarray,
) -> dict[str, Any]:
    selected_metrics = metrics.iloc[selected_idx] if len(selected_idx) else metrics.iloc[:0]
    selected_frame = frame.iloc[selected_idx] if len(selected_idx) else frame.iloc[:0]
    selected_target = target.iloc[selected_idx] if len(selected_idx) else target.iloc[:0]
    symbols = selected_frame.get("__symbol__", pd.Series(dtype=object))
    side_source = selected_frame.get("side", selected_metrics.get("side", pd.Series(dtype=float)))
    side_values = _safe_numeric(side_source)
    long_rows = int((side_values > 0.0).sum()) if len(side_values) else 0
    short_rows = int((side_values < 0.0).sum()) if len(side_values) else 0
    side_counts = side_values.map(lambda value: "short" if value < 0.0 else "long").value_counts(normalize=True)
    side_top_share = float(side_counts.iloc[0]) if len(side_counts) else 0.0
    return {
        "selected_rows": int(len(selected_idx)),
        "target_hard_rate": _safe_mean(selected_target.get("target_hard")),
        "target_soft_mean": _safe_mean(selected_target.get("target_soft")),
        "mean_u": _safe_mean(selected_metrics.get("u_policy_net")),
        "median_u": _safe_quantile(selected_metrics.get("u_policy_net"), 0.50),
        "q10_u": _safe_quantile(selected_metrics.get("u_policy_net"), 0.10),
        "hit_u": _safe_mean(selected_metrics.get("u_policy_net") > 0.0),
        "timeout_rate": _safe_mean(selected_metrics.get("is_timeout").astype(float)) if len(selected_metrics) else float("nan"),
        "mean_bars_policy": _safe_mean(selected_metrics.get("bars_policy")),
        "p90_bars_policy": _safe_quantile(selected_metrics.get("bars_policy"), 0.90),
        "bad_mae_1r_rate": _safe_mean(selected_metrics.get("mae_norm") >= 1.0),
        "p90_mae_norm": _safe_quantile(selected_metrics.get("mae_norm"), 0.90),
        "wide_barrier_25bps_rate": _safe_mean(selected_metrics.get("barrier") > 0.025),
        "wide_barrier_35bps_rate": _safe_mean(selected_metrics.get("barrier") > 0.035),
        "symbol_effective_n": _effective_n(symbols),
        "top_symbol_share": float(symbols.value_counts(normalize=True, dropna=False).iloc[0]) if len(symbols) else 0.0,
        "unique_symbols": int(symbols.nunique(dropna=False)) if len(symbols) else 0,
        "long_rows": long_rows,
        "short_rows": short_rows,
        "long_share": float(long_rows / len(selected_idx)) if len(selected_idx) else 0.0,
        "short_share": float(short_rows / len(selected_idx)) if len(selected_idx) else 0.0,
        "side_effective_n": _effective_n(side_values.map(lambda value: "short" if value < 0.0 else "long")),
        "side_top_share": side_top_share,
    }


def _baseline_summary(frame: pd.DataFrame, metrics: pd.DataFrame, target: pd.DataFrame) -> dict[str, Any]:
    idx = np.arange(len(frame), dtype=np.int64)
    out = _summary(frame=frame, metrics=metrics, target=target, selected_idx=idx)
    return {f"valid_{key}": value for key, value in out.items()}


def _valid_weekly_stats(frame: pd.DataFrame, metrics: pd.DataFrame) -> dict[str, Any]:
    if frame.empty or metrics.empty:
        return {
            "valid_weeks": 0,
            "valid_positive_weeks": 0,
            "valid_q25_week_u": float("nan"),
            "valid_worst_week_u": float("nan"),
        }
    weekly = pd.DataFrame(
        {
            "week_start": _week_start(frame["__ts__"]),
            "u_policy_net": metrics["u_policy_net"],
        }
    )
    week_u = weekly.groupby("week_start", dropna=False, observed=True)["u_policy_net"].mean()
    return {
        "valid_weeks": int(len(week_u)),
        "valid_positive_weeks": int((week_u > 0.0).sum()),
        "valid_q25_week_u": _safe_quantile(week_u, 0.25),
        "valid_worst_week_u": _safe_quantile(week_u, 0.0),
    }


def _calibration_deciles(
    *,
    period: str,
    label: str,
    label_kind: str,
    feature_set: str,
    target: pd.DataFrame,
    score: pd.Series,
    metrics: pd.DataFrame,
) -> pd.DataFrame:
    data = pd.DataFrame(
        {
            "target_hard": _safe_numeric(target["target_hard"]),
            "target_soft": _safe_numeric(target["target_soft"]),
            "score": _safe_numeric(score).clip(0.0, 1.0),
            "is_timeout": _safe_numeric(metrics["is_timeout"]),
            "bars_policy": _safe_numeric(metrics["bars_policy"]),
            "u_policy_net": _safe_numeric(metrics["u_policy_net"]),
            "mae_norm": _safe_numeric(metrics["mae_norm"]),
            "barrier": _safe_numeric(metrics["barrier"]),
        }
    ).dropna(subset=["target_hard", "score"])
    if data.empty:
        return pd.DataFrame()
    data = data.sort_values("score", kind="mergesort").reset_index(drop=True)
    data["risk_decile"] = np.floor(np.arange(len(data), dtype=np.float64) * 10.0 / len(data)).astype(int) + 1
    data["risk_decile"] = data["risk_decile"].clip(1, 10)
    baseline_target = _safe_mean(data["target_hard"])
    baseline_timeout = _safe_mean(data["is_timeout"])
    rows: list[dict[str, Any]] = []
    for decile, group in data.groupby("risk_decile", dropna=False, observed=True):
        target_rate = _safe_mean(group["target_hard"])
        timeout_rate = _safe_mean(group["is_timeout"])
        rows.append(
            {
                "period": period,
                "label": label,
                "label_kind": label_kind,
                "feature_set": feature_set,
                "risk_decile": int(decile),
                "rows": int(len(group)),
                "score_min": _safe_quantile(group["score"], 0.0),
                "score_mean": _safe_mean(group["score"]),
                "score_max": _safe_quantile(group["score"], 1.0),
                "target_hard_rate": target_rate,
                "target_soft_mean": _safe_mean(group["target_soft"]),
                "timeout_rate": timeout_rate,
                "mean_bars_policy": _safe_mean(group["bars_policy"]),
                "mean_u": _safe_mean(group["u_policy_net"]),
                "bad_mae_1r_rate": _safe_mean(group["mae_norm"] >= 1.0),
                "wide_barrier_25bps_rate": _safe_mean(group["barrier"] > 0.025),
                "brier_score": _safe_brier(group["target_hard"], group["score"]),
                "target_rate_lift_vs_valid": (
                    target_rate / baseline_target
                    if math.isfinite(target_rate) and math.isfinite(baseline_target) and baseline_target > 0.0
                    else float("nan")
                ),
                "timeout_lift_vs_valid": (
                    timeout_rate / baseline_timeout
                    if math.isfinite(timeout_rate) and math.isfinite(baseline_timeout) and baseline_timeout > 0.0
                    else float("nan")
                ),
            }
        )
    return pd.DataFrame(rows)


def _add_delta_metrics(row: dict[str, Any]) -> None:
    for col in (
        "target_hard_rate",
        "target_soft_mean",
        "mean_u",
        "timeout_rate",
        "mean_bars_policy",
        "bad_mae_1r_rate",
        "wide_barrier_25bps_rate",
    ):
        base = row.get(f"valid_{col}")
        value = row.get(col)
        row[f"delta_{col}_vs_valid"] = (
            float(value) - float(base)
            if value is not None and base is not None and math.isfinite(float(value)) and math.isfinite(float(base))
            else float("nan")
        )
    valid_timeout = float(row.get("valid_timeout_rate", float("nan")))
    timeout = float(row.get("timeout_rate", float("nan")))
    row["timeout_reduction_frac_vs_valid"] = (
        (valid_timeout - timeout) / valid_timeout
        if math.isfinite(valid_timeout) and valid_timeout > 0.0 and math.isfinite(timeout)
        else float("nan")
    )
    valid_target = float(row.get("valid_target_hard_rate", float("nan")))
    target = float(row.get("target_hard_rate", float("nan")))
    row["target_rate_lift_vs_valid"] = (
        target / valid_target
        if math.isfinite(valid_target) and valid_target > 0.0 and math.isfinite(target)
        else float("nan")
    )
    valid_u = float(row.get("valid_mean_u", float("nan")))
    mean_u = float(row.get("mean_u", float("nan")))
    row["utility_retention_vs_valid"] = (
        mean_u / valid_u
        if math.isfinite(valid_u) and valid_u > 0.0 and math.isfinite(mean_u)
        else float("nan")
    )


def _selected_rows(
    *,
    frame: pd.DataFrame,
    metrics: pd.DataFrame,
    target: pd.DataFrame,
    score: pd.Series,
    selected_idx: np.ndarray,
    context: dict[str, Any],
) -> pd.DataFrame:
    if not len(selected_idx):
        return pd.DataFrame()
    cols = ["__ts__", "__symbol__"]
    for col in ("side", "side_name", "__side__", "timeframe", "candidate_id"):
        if col in frame.columns:
            cols.append(col)
    if "primary_source_tag" in frame.columns:
        cols.append("primary_source_tag")
    selected = frame.iloc[selected_idx][cols].copy()
    if "side" not in selected.columns and "side" in metrics.columns:
        selected["side"] = metrics["side"].iloc[selected_idx].to_numpy(dtype=np.int8, copy=False)
    if "side_name" not in selected.columns and "side" in selected.columns:
        selected["side_name"] = np.where(_safe_numeric(selected["side"]) < 0.0, "short", "long")
    for key, value in context.items():
        selected[key] = value
    selected["week_start"] = _week_start(selected["__ts__"])
    selected["risk_score"] = score.iloc[selected_idx].to_numpy(dtype=np.float32, copy=False)
    selected["target_soft"] = target["target_soft"].iloc[selected_idx].to_numpy(dtype=np.float32, copy=False)
    selected["target_hard"] = target["target_hard"].iloc[selected_idx].to_numpy(dtype=np.float32, copy=False)
    selected["target_bucket"] = target["target_bucket"].iloc[selected_idx].astype(str).to_numpy()
    for col in ("u_policy_net", "mae_norm", "barrier", "is_timeout", "bars_policy"):
        selected[col] = metrics[col].iloc[selected_idx].to_numpy()
    return selected


def _weekly_summary(selected: pd.DataFrame) -> pd.DataFrame:
    if selected.empty:
        return pd.DataFrame()
    group_cols = ["label", "label_kind", "feature_set", "selector", "fraction", "period", "week_start"]
    rows: list[dict[str, Any]] = []
    for key, group in selected.groupby(group_cols, dropna=False, observed=True):
        context = dict(zip(group_cols, key, strict=False))
        metrics = pd.DataFrame(
            {
                "u_policy_net": group["u_policy_net"],
                "mae_norm": group["mae_norm"],
                "barrier": group["barrier"],
                "is_timeout": group["is_timeout"],
                "bars_policy": group["bars_policy"],
                "side": group["side"] if "side" in group.columns else 1,
            }
        )
        frame_cols = ["__ts__", "__symbol__"] + [col for col in ("side", "side_name", "timeframe", "candidate_id") if col in group.columns]
        frame = group[frame_cols].copy()
        target = pd.DataFrame(
            {
                "target_soft": group["target_soft"],
                "target_hard": group["target_hard"],
            }
        )
        rows.append({**context, **_summary(frame=frame, metrics=metrics, target=target, selected_idx=np.arange(len(group)))})
    return pd.DataFrame(rows).sort_values(group_cols, kind="mergesort")


def _score_month(
    *,
    frame: pd.DataFrame,
    metrics: pd.DataFrame,
    month_period: pd.Series,
    month: str,
    target_specs: list[TargetSpec],
    feature_map: dict[str, list[str]],
    feature_sets: list[str],
    high_risk_top_fracs: list[float],
    low_risk_keep_fracs: list[float],
    seeds: list[int],
    train_lookback_months: int | None,
    min_train_rows: int,
    min_valid_rows: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[pd.DataFrame], list[pd.DataFrame]]:
    train_mask = month_period < str(month)
    if train_lookback_months is not None and int(train_lookback_months) > 0:
        prior_months = sorted(month_period[train_mask].dropna().unique())
        keep = set(prior_months[-int(train_lookback_months) :])
        train_mask = train_mask & month_period.isin(keep)
    valid_mask = month_period == str(month)
    if int(valid_mask.sum()) < int(min_valid_rows):
        return [], [{"period": month, "skipped": True, "reason": "too_few_valid_rows"}], [], []

    monthly_rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    selected_frames: list[pd.DataFrame] = []
    calibration_frames: list[pd.DataFrame] = []
    targets = {spec.name: _build_target(metrics=metrics, train_mask=train_mask, spec=spec) for spec in target_specs}

    for spec in target_specs:
        target, weights, target_report = targets[spec.name]
        train_label_mask = train_mask & target["target_soft"].notna() & weights.gt(0.0)
        valid_label_mask = valid_mask & target["target_soft"].notna()
        if int(train_label_mask.sum()) < int(min_train_rows) or int(valid_label_mask.sum()) < int(min_valid_rows):
            diagnostics.append(
                {
                    "period": month,
                    "label": spec.name,
                    "skipped": True,
                    "reason": "too_few_target_rows",
                    "train_rows": int(train_label_mask.sum()),
                    "valid_rows": int(valid_label_mask.sum()),
                    **target_report,
                }
            )
            continue
        for feature_set in feature_sets:
            features = feature_map.get(feature_set, [])
            if not features:
                continue
            x_train, x_valid = _month_model_frame(
                frame,
                train_mask=train_label_mask,
                valid_mask=valid_label_mask,
                features=features,
            )
            pred_matrix = np.vstack(
                [
                    _fit_predict(
                        x_train=x_train,
                        y_train=target.loc[train_label_mask, "target_soft"],
                        w_train=weights.loc[train_label_mask],
                        x_valid=x_valid,
                        seed=seed,
                    )
                    for seed in seeds
                ]
            )
            pred = pd.Series(np.mean(pred_matrix, axis=0).astype(np.float32), index=frame.loc[valid_label_mask].index)
            valid_frame = frame.loc[valid_label_mask].reset_index(drop=True)
            valid_metrics = metrics.loc[valid_label_mask].reset_index(drop=True)
            valid_target = target.loc[valid_label_mask].reset_index(drop=True)
            score = pred.reset_index(drop=True)
            baseline = _baseline_summary(valid_frame, valid_metrics, valid_target)
            valid_weekly = _valid_weekly_stats(valid_frame, valid_metrics)
            valid_prevalence = _safe_mean(valid_target["target_hard"])
            ap = _safe_average_precision(valid_target["target_hard"], score)
            brier = _safe_brier(valid_target["target_hard"], score)
            calibration = _calibration_deciles(
                period=month,
                label=spec.name,
                label_kind=spec.kind,
                feature_set=feature_set,
                target=valid_target,
                score=score,
                metrics=valid_metrics,
            )
            if not calibration.empty:
                calibration_frames.append(calibration)
                top_risk_decile = calibration[calibration["risk_decile"].eq(10)]
                top_risk_timeout_rate = _safe_mean(top_risk_decile["timeout_rate"])
                top_risk_timeout_lift = _safe_mean(top_risk_decile["timeout_lift_vs_valid"])
                top_risk_target_lift = _safe_mean(top_risk_decile["target_rate_lift_vs_valid"])
            else:
                top_risk_timeout_rate = float("nan")
                top_risk_timeout_lift = float("nan")
                top_risk_target_lift = float("nan")
            diagnostics.append(
                {
                    "period": month,
                    "label": spec.name,
                    "feature_set": feature_set,
                    "skipped": False,
                    "train_rows": int(train_label_mask.sum()),
                    "valid_rows": int(valid_label_mask.sum()),
                    "model_feature_count": int(len(features)),
                    "target_valid_hard_rate": valid_prevalence,
                    "target_valid_soft_mean": _safe_mean(valid_target["target_soft"]),
                    "score_ic_target": _spearman(score, valid_target["target_soft"]),
                    "score_ic_target_hard": _spearman(score, valid_target["target_hard"]),
                    "score_ic_timeout": _spearman(score, valid_metrics["is_timeout"].astype(float)),
                    "score_ic_bars_policy": _spearman(score, valid_metrics["bars_policy"]),
                    "target_auc": _safe_auc(valid_target["target_hard"], score),
                    "target_average_precision": ap,
                    "target_brier_score": brier,
                    "target_pr_auc_lift": ap / valid_prevalence
                    if math.isfinite(ap) and math.isfinite(valid_prevalence) and valid_prevalence > 0.0
                    else float("nan"),
                    "top_risk_decile_timeout_rate": top_risk_timeout_rate,
                    "top_risk_decile_timeout_lift": top_risk_timeout_lift,
                    "top_risk_decile_target_lift": top_risk_target_lift,
                    "prediction_seed_std_mean": float(np.std(pred_matrix, axis=0).mean()) if pred_matrix.size else float("nan"),
                    **target_report,
                }
            )

            selection_specs: list[tuple[str, float, np.ndarray]] = []
            for frac in high_risk_top_fracs:
                selection_specs.append(("high_risk_top", float(frac), _rank_fraction_indices(score, frac, highest=True)))
            for frac in low_risk_keep_fracs:
                selection_specs.append(("low_risk_keep", float(frac), _rank_fraction_indices(score, frac, highest=False)))
                selection_specs.append(
                    (
                        "low_risk_keep_weekly",
                        float(frac),
                        _rank_fraction_by_group_indices(
                            score,
                            _week_start(valid_frame["__ts__"]),
                            frac,
                            highest=False,
                        ),
                    )
                )

            for selector, frac, selected_idx in selection_specs:
                row = {
                    "period": month,
                    "label": spec.name,
                    "label_kind": spec.kind,
                    "feature_set": feature_set,
                    "selector": selector,
                    "fraction": float(frac),
                    "train_rows": int(train_label_mask.sum()),
                    "valid_rows": int(len(valid_frame)),
                    "model_feature_count": int(len(features)),
                    "score_ic_target": _spearman(score, valid_target["target_soft"]),
                    "score_ic_timeout": _spearman(score, valid_metrics["is_timeout"].astype(float)),
                    "target_auc": _safe_auc(valid_target["target_hard"], score),
                    "target_average_precision": ap,
                    "target_brier_score": brier,
                    "target_pr_auc_lift": ap / valid_prevalence
                    if math.isfinite(ap) and math.isfinite(valid_prevalence) and valid_prevalence > 0.0
                    else float("nan"),
                    "top_risk_decile_timeout_rate": top_risk_timeout_rate,
                    "top_risk_decile_timeout_lift": top_risk_timeout_lift,
                    "top_risk_decile_target_lift": top_risk_target_lift,
                    **baseline,
                    **valid_weekly,
                    **_summary(
                        frame=valid_frame,
                        metrics=valid_metrics,
                        target=valid_target,
                        selected_idx=selected_idx,
                    ),
                }
                _add_delta_metrics(row)
                monthly_rows.append(row)
                selected_frames.append(
                    _selected_rows(
                        frame=valid_frame,
                        metrics=valid_metrics,
                        target=valid_target,
                        score=score,
                        selected_idx=selected_idx,
                        context={
                            "period": month,
                            "label": spec.name,
                            "label_kind": spec.kind,
                            "feature_set": feature_set,
                            "selector": selector,
                            "fraction": float(frac),
                        },
                    )
                )
    return monthly_rows, diagnostics, selected_frames, calibration_frames


def _cv(values: Any) -> float:
    arr = _safe_numeric(values).dropna()
    if len(arr) < 2:
        return float("nan")
    mean = float(arr.mean())
    if abs(mean) < 1e-12:
        return float("nan")
    return float(arr.std(ddof=0) / abs(mean))


def _aggregate(monthly: pd.DataFrame, weekly: pd.DataFrame, *, expected_months: int) -> pd.DataFrame:
    if monthly.empty:
        return pd.DataFrame()
    group_cols = ["label", "label_kind", "feature_set", "selector", "fraction"]
    weekly_map: dict[tuple[Any, ...], pd.DataFrame] = {}
    if not weekly.empty:
        for key, group in weekly.groupby(group_cols, dropna=False, observed=True):
            weekly_map[key] = group
    rows: list[dict[str, Any]] = []
    for key, group in monthly.groupby(group_cols, dropna=False, observed=True):
        label, label_kind, feature_set, selector, fraction = key
        month_count = int(group["period"].nunique())
        prevalence = _safe_mean(group["valid_target_hard_rate"])
        prevalence_cv = _cv(group["valid_target_hard_rate"])
        mean_auc = _safe_mean(group["target_auc"])
        mean_ic = _safe_mean(group["score_ic_target"])
        mean_pr_lift = _safe_mean(group["target_pr_auc_lift"])
        ic_positive_months = int((_safe_numeric(group["score_ic_target"]) > 0.0).sum())
        learnable = (
            month_count >= expected_months
            and math.isfinite(prevalence)
            and 0.02 <= prevalence <= 0.80
            and (not math.isfinite(prevalence_cv) or prevalence_cv <= 0.75)
            and (
                (math.isfinite(mean_auc) and mean_auc >= 0.53)
                or (math.isfinite(mean_ic) and mean_ic >= 0.03)
                or (math.isfinite(mean_pr_lift) and mean_pr_lift >= 1.10)
            )
            and ic_positive_months >= max(1, expected_months - 1)
        )
        weekly_group = weekly_map.get(key, pd.DataFrame())
        weeks = int(len(weekly_group))
        positive_weeks = int((_safe_numeric(weekly_group.get("mean_u", pd.Series(dtype=float))) > 0.0).sum())
        q25_week_u = _safe_quantile(weekly_group.get("mean_u", pd.Series(dtype=float)), 0.25)
        worst_week_u = _safe_quantile(weekly_group.get("mean_u", pd.Series(dtype=float)), 0.0)
        max_week_top_symbol = _safe_quantile(weekly_group.get("top_symbol_share", pd.Series(dtype=float)), 1.0)
        max_week_side_top = _safe_quantile(weekly_group.get("side_top_share", pd.Series(dtype=float)), 1.0)
        min_week_selected = _safe_quantile(weekly_group.get("selected_rows", pd.Series(dtype=float)), 0.0)
        mean_week_selected = _safe_mean(weekly_group.get("selected_rows", pd.Series(dtype=float)))
        valid_positive_weeks = int(_safe_numeric(group.get("valid_positive_weeks", pd.Series(dtype=float))).fillna(0).sum())
        valid_weeks = int(_safe_numeric(group.get("valid_weeks", pd.Series(dtype=float))).fillna(0).sum())
        valid_q25_week_u = _safe_mean(group.get("valid_q25_week_u", pd.Series(dtype=float)))
        valid_worst_week_u = _safe_mean(group.get("valid_worst_week_u", pd.Series(dtype=float)))
        q25_week_u_delta = (
            q25_week_u - valid_q25_week_u
            if math.isfinite(q25_week_u) and math.isfinite(valid_q25_week_u)
            else float("nan")
        )
        mean_timeout_reduction = _safe_mean(group["timeout_reduction_frac_vs_valid"])
        min_timeout_reduction = _safe_quantile(group["timeout_reduction_frac_vs_valid"], 0.0)
        mean_target_lift = _safe_mean(group["target_rate_lift_vs_valid"])
        mean_retention = _safe_mean(group["utility_retention_vs_valid"])
        mean_delta_u = _safe_mean(group["delta_mean_u_vs_valid"])
        min_selected = _safe_quantile(group["selected_rows"], 0.0)
        high_risk_ok = (
            selector == "high_risk_top"
            and learnable
            and math.isfinite(mean_target_lift)
            and mean_target_lift >= 1.10
            and math.isfinite(max_week_top_symbol)
            and max_week_top_symbol <= 0.75
        )
        low_risk_ok = (
            selector in {"low_risk_keep", "low_risk_keep_weekly"}
            and learnable
            and math.isfinite(mean_timeout_reduction)
            and mean_timeout_reduction >= 0.10
            and math.isfinite(min_timeout_reduction)
            and min_timeout_reduction > 0.0
            and (not math.isfinite(mean_retention) or mean_retention >= 0.75 or mean_delta_u >= -0.005)
            and math.isfinite(min_selected)
            and min_selected >= 25.0
            and math.isfinite(max_week_top_symbol)
            and max_week_top_symbol <= 0.75
        )
        if low_risk_ok:
            decision = "candidate_timeout_filter"
        elif high_risk_ok:
            decision = "learnable_high_risk_signal"
        elif selector in {"low_risk_keep", "low_risk_keep_weekly"} and math.isfinite(mean_timeout_reduction) and mean_timeout_reduction >= 0.10:
            decision = "risk_reducer_utility_or_stability_cost"
        elif learnable:
            decision = "learnable_diagnostic_only"
        else:
            decision = "diagnostic_only"
        rows.append(
            {
                "decision": decision,
                "label": label,
                "label_kind": label_kind,
                "feature_set": feature_set,
                "selector": selector,
                "fraction": float(fraction),
                "months": month_count,
                "valid_target_hard_rate": prevalence,
                "valid_target_hard_rate_cv": prevalence_cv,
                "score_ic_target": mean_ic,
                "score_ic_timeout": _safe_mean(group["score_ic_timeout"]),
                "target_auc": mean_auc,
                "target_brier_score": _safe_mean(group.get("target_brier_score", pd.Series(dtype=float))),
                "target_pr_auc_lift": mean_pr_lift,
                "top_risk_decile_timeout_rate": _safe_mean(group.get("top_risk_decile_timeout_rate", pd.Series(dtype=float))),
                "top_risk_decile_timeout_lift": _safe_mean(group.get("top_risk_decile_timeout_lift", pd.Series(dtype=float))),
                "top_risk_decile_target_lift": _safe_mean(group.get("top_risk_decile_target_lift", pd.Series(dtype=float))),
                "ic_positive_months": ic_positive_months,
                "mean_selected_rows": _safe_mean(group["selected_rows"]),
                "min_selected_rows": min_selected,
                "mean_week_selected_rows": mean_week_selected,
                "min_week_selected_rows": min_week_selected,
                "target_rate_lift_vs_valid": mean_target_lift,
                "timeout_reduction_frac_vs_valid": mean_timeout_reduction,
                "min_timeout_reduction_frac_vs_valid": min_timeout_reduction,
                "mean_u": _safe_mean(group["mean_u"]),
                "delta_mean_u_vs_valid": mean_delta_u,
                "utility_retention_vs_valid": mean_retention,
                "timeout_rate": _safe_mean(group["timeout_rate"]),
                "valid_timeout_rate": _safe_mean(group["valid_timeout_rate"]),
                "bad_mae_1r_rate": _safe_mean(group["bad_mae_1r_rate"]),
                "wide_barrier_25bps_rate": _safe_mean(group["wide_barrier_25bps_rate"]),
                "mean_bars_policy": _safe_mean(group["mean_bars_policy"]),
                "top_symbol_share": _safe_mean(group["top_symbol_share"]),
                "long_share": _safe_mean(group.get("long_share", pd.Series(dtype=float))),
                "short_share": _safe_mean(group.get("short_share", pd.Series(dtype=float))),
                "side_effective_n": _safe_mean(group.get("side_effective_n", pd.Series(dtype=float))),
                "side_top_share": _safe_mean(group.get("side_top_share", pd.Series(dtype=float))),
                "weeks": weeks,
                "positive_weeks": positive_weeks,
                "valid_weeks": valid_weeks,
                "valid_positive_weeks": valid_positive_weeks,
                "q25_week_u": q25_week_u,
                "valid_q25_week_u": valid_q25_week_u,
                "q25_week_u_delta_vs_valid": q25_week_u_delta,
                "worst_week_u": worst_week_u,
                "valid_worst_week_u": valid_worst_week_u,
                "max_week_top_symbol_share": max_week_top_symbol,
                "max_week_side_top_share": max_week_side_top,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["decision", "timeout_reduction_frac_vs_valid", "target_rate_lift_vs_valid", "score_ic_target"],
        ascending=[True, False, False, False],
        na_position="last",
        kind="mergesort",
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


def _stage1_gate(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    candidates = frame[frame["selector"].isin(["low_risk_keep", "low_risk_keep_weekly"]) & frame["fraction"].eq(0.5)].copy()
    if candidates.empty:
        candidates = frame[frame["selector"].isin(["low_risk_keep", "low_risk_keep_weekly"])].copy()
    for _, row in candidates.sort_values(
        ["timeout_reduction_frac_vs_valid", "top_risk_decile_timeout_lift", "target_auc"],
        ascending=[False, False, False],
        na_position="last",
        kind="mergesort",
    ).iterrows():
        valid_mean_u = float(row.get("mean_u", float("nan")) - row.get("delta_mean_u_vs_valid", float("nan")))
        utility_delta = float(row.get("delta_mean_u_vs_valid", float("nan")))
        if math.isfinite(valid_mean_u) and valid_mean_u > 0.0:
            utility_drawdown_ok = bool(row.get("utility_retention_vs_valid", float("nan")) >= 0.90)
        else:
            utility_drawdown_ok = bool(math.isfinite(utility_delta) and utility_delta >= -0.001)
        checks = {
            "top_risk_decile_timeout_lift_ok": bool(row.get("top_risk_decile_timeout_lift", float("nan")) >= 1.5),
            "low_risk_timeout_rate_ok": bool(row.get("timeout_rate", float("nan")) <= 0.20),
            "utility_drawdown_ok": utility_drawdown_ok,
            "q25_week_u_improves_ok": bool(row.get("q25_week_u_delta_vs_valid", float("nan")) > 0.0),
            "positive_weeks_not_decline_ok": bool(row.get("positive_weeks", -1) >= row.get("valid_positive_weeks", 10**9)),
            "rows_per_week_viable_ok": bool(row.get("min_week_selected_rows", float("nan")) >= 5.0),
        }
        rows.append(
            {
                "stage1_gate": "pass" if all(checks.values()) else "fail",
                **{key: "yes" if value else "no" for key, value in checks.items()},
                "decision": row.get("decision"),
                "label": row.get("label"),
                "feature_set": row.get("feature_set"),
                "selector": row.get("selector"),
                "fraction": row.get("fraction"),
                "target_auc": row.get("target_auc"),
                "target_brier_score": row.get("target_brier_score"),
                "target_pr_auc_lift": row.get("target_pr_auc_lift"),
                "top_risk_decile_timeout_lift": row.get("top_risk_decile_timeout_lift"),
                "timeout_rate": row.get("timeout_rate"),
                "valid_timeout_rate": row.get("valid_timeout_rate"),
                "timeout_reduction_frac_vs_valid": row.get("timeout_reduction_frac_vs_valid"),
                "mean_u": row.get("mean_u"),
                "delta_mean_u_vs_valid": row.get("delta_mean_u_vs_valid"),
                "q25_week_u": row.get("q25_week_u"),
                "valid_q25_week_u": row.get("valid_q25_week_u"),
                "q25_week_u_delta_vs_valid": row.get("q25_week_u_delta_vs_valid"),
                "positive_weeks": row.get("positive_weeks"),
                "valid_positive_weeks": row.get("valid_positive_weeks"),
                "min_week_selected_rows": row.get("min_week_selected_rows"),
                "max_week_top_symbol_share": row.get("max_week_top_symbol_share"),
            }
        )
    return pd.DataFrame(rows)


def _write_report(
    output_dir: Path,
    aggregate: pd.DataFrame,
    diagnostics: pd.DataFrame,
    calibration: pd.DataFrame,
    manifest: dict[str, Any],
) -> Path:
    path = output_dir / "timeout_holding_risk_label_report.md"
    cols = [
        "decision",
        "label",
        "feature_set",
        "selector",
        "fraction",
        "months",
        "valid_target_hard_rate",
        "valid_target_hard_rate_cv",
        "score_ic_target",
        "score_ic_timeout",
        "target_auc",
        "target_brier_score",
        "target_pr_auc_lift",
        "top_risk_decile_timeout_lift",
        "target_rate_lift_vs_valid",
        "timeout_reduction_frac_vs_valid",
        "mean_u",
        "delta_mean_u_vs_valid",
        "timeout_rate",
        "valid_timeout_rate",
        "bad_mae_1r_rate",
        "wide_barrier_25bps_rate",
        "mean_selected_rows",
        "min_selected_rows",
        "min_week_selected_rows",
        "positive_weeks",
        "valid_positive_weeks",
        "q25_week_u",
        "valid_q25_week_u",
        "q25_week_u_delta_vs_valid",
        "worst_week_u",
        "max_week_top_symbol_share",
        "short_share",
        "side_effective_n",
        "max_week_side_top_share",
    ]
    candidate = aggregate[aggregate["decision"].eq("candidate_timeout_filter")]
    high_risk = aggregate[aggregate["decision"].eq("learnable_high_risk_signal")]
    reducers = aggregate[aggregate["decision"].eq("risk_reducer_utility_or_stability_cost")]
    learnable = aggregate[aggregate["decision"].str.contains("learnable", na=False)]
    diag_cols = [
        "period",
        "label",
        "feature_set",
        "train_rows",
        "valid_rows",
        "target_valid_hard_rate",
        "score_ic_target",
        "score_ic_timeout",
        "score_ic_bars_policy",
        "target_auc",
        "target_brier_score",
        "target_pr_auc_lift",
        "top_risk_decile_timeout_lift",
        "train_timeout_rate",
        "train_mean_bars_policy",
        "bars_q40",
        "bars_q80",
    ]
    gate_cols = [
        "stage1_gate",
        "top_risk_decile_timeout_lift_ok",
        "low_risk_timeout_rate_ok",
        "utility_drawdown_ok",
        "q25_week_u_improves_ok",
        "positive_weeks_not_decline_ok",
        "rows_per_week_viable_ok",
        "decision",
        "label",
        "feature_set",
        "selector",
        "fraction",
        "target_auc",
        "target_brier_score",
        "target_pr_auc_lift",
        "top_risk_decile_timeout_lift",
        "timeout_rate",
        "valid_timeout_rate",
        "timeout_reduction_frac_vs_valid",
        "delta_mean_u_vs_valid",
        "q25_week_u",
        "valid_q25_week_u",
        "q25_week_u_delta_vs_valid",
        "positive_weeks",
        "valid_positive_weeks",
        "min_week_selected_rows",
        "max_week_top_symbol_share",
    ]
    cal_cols = [
        "period",
        "label",
        "feature_set",
        "risk_decile",
        "rows",
        "score_mean",
        "target_hard_rate",
        "timeout_rate",
        "timeout_lift_vs_valid",
        "brier_score",
        "mean_u",
    ]
    gate = _stage1_gate(aggregate)
    lines = [
        "# Timeout / Holding-Risk Label Diagnostic",
        "",
        "Scope: diagnostic-only timeout, holding, and time-to-resolution labels. Holding thresholds are calibrated from prior months only.",
        "Horizon note: Stage 1 is framed around 3-7h side-aware candidates. The corrected roadmap utility window is 1-6 hours, not 1-60 minutes.",
        "",
        f"Rows joined to label ledger: `{manifest['rows']}`",
        f"Utility source: `{manifest.get('utility_source', '')}`",
        f"Months: `{', '.join(manifest['months'])}`",
        f"Labels: `{', '.join(manifest['labels'])}`",
        f"Feature sets: `{', '.join(manifest['feature_sets'])}`",
        f"Side counts: `{json.dumps(manifest.get('side_counts', {}), sort_keys=True)}`",
        "",
        "## Stage 1 Readiness Gate",
        "",
        "Pass requires top-risk timeout lift, low-risk timeout suppression, tolerable utility drawdown, improved q25 weekly utility, no positive-week decline, and viable rows/week.",
        "",
        _table(gate, gate_cols, limit=80),
        "",
        "## Candidate Timeout Filters",
        "",
        _table(candidate, cols, limit=80),
        "",
        "## Learnable High-Risk Signals",
        "",
        _table(high_risk, cols, limit=80),
        "",
        "## Timeout Reducers With Cost",
        "",
        _table(reducers, cols, limit=80),
        "",
        "## Other Learnable Diagnostics",
        "",
        _table(learnable, cols, limit=120),
        "",
        "## Best By Timeout Reduction",
        "",
        _table(aggregate.sort_values("timeout_reduction_frac_vs_valid", ascending=False), cols, limit=120),
        "",
        "## Label Learnability By Month",
        "",
        _table(diagnostics[diagnostics.get("skipped", False).eq(False)] if "skipped" in diagnostics else diagnostics, diag_cols, limit=120),
        "",
        "## Calibration By Risk Decile",
        "",
        _table(calibration.sort_values(["label", "feature_set", "period", "risk_decile"], kind="mergesort"), cal_cols, limit=180),
        "",
        "## Outputs",
        "",
        f"- Monthly: `{manifest['outputs']['monthly']}`",
        f"- Weekly: `{manifest['outputs']['weekly']}`",
        f"- Aggregate: `{manifest['outputs']['aggregate']}`",
        f"- Diagnostics: `{manifest['outputs']['diagnostics']}`",
        f"- Calibration: `{manifest['outputs']['calibration']}`",
        f"- Selected rows: `{manifest['outputs']['selected_rows_parquet']}`",
        f"- Manifest: `{manifest['outputs']['manifest']}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_diagnostic(
    *,
    quality_labels_path: Path,
    labels_path: Path,
    config_path: Path,
    output_dir: Path,
    feature_dir: Path,
    feature_list_csv: Path,
    max_feature_store_features: int | None,
    months: list[str],
    labels_requested: list[str],
    feature_sets: list[str],
    high_risk_top_fracs: list[float],
    low_risk_keep_fracs: list[float],
    seeds: list[int],
    train_lookback_months: int | None,
    min_train_rows: int,
    min_valid_rows: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    target_specs = _target_specs_by_name(labels_requested)
    frame, join_report = _load_joined_frame(quality_labels_path=quality_labels_path, labels_path=labels_path)
    selected_features = _read_feature_list(feature_list_csv, max_features=max_feature_store_features)
    feature_matrix, feature_report = _load_feature_store_columns(
        frame,
        feature_dir=feature_dir,
        selected_features=selected_features,
    )
    for col in feature_matrix.columns:
        frame[col] = feature_matrix[col].to_numpy(dtype=np.float32, copy=False)
    metrics = _path_metrics(frame)
    side_counts = {
        "long": int((_safe_numeric(metrics["side"]) > 0.0).sum()),
        "short": int((_safe_numeric(metrics["side"]) < 0.0).sum()),
    }
    base_features = list(feature_matrix.columns)
    source_features = _source_feature_columns(frame)
    feature_map = {
        "base": base_features,
        "base_plus_source": list(dict.fromkeys(base_features + source_features)),
    }
    month_period = frame["__ts__"].dt.to_period("M").astype(str)

    monthly_rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    selected_frames: list[pd.DataFrame] = []
    calibration_frames: list[pd.DataFrame] = []
    for month in months:
        rows, diag, selected, calibration = _score_month(
            frame=frame,
            metrics=metrics,
            month_period=month_period,
            month=month,
            target_specs=target_specs,
            feature_map=feature_map,
            feature_sets=feature_sets,
            high_risk_top_fracs=high_risk_top_fracs,
            low_risk_keep_fracs=low_risk_keep_fracs,
            seeds=seeds,
            train_lookback_months=train_lookback_months,
            min_train_rows=min_train_rows,
            min_valid_rows=min_valid_rows,
        )
        monthly_rows.extend(rows)
        diagnostics.extend(diag)
        selected_frames.extend(selected)
        calibration_frames.extend(calibration)

    monthly = pd.DataFrame(monthly_rows)
    diagnostics_frame = pd.DataFrame(diagnostics)
    selected_rows = (
        pd.concat([frame for frame in selected_frames if not frame.empty], ignore_index=True)
        if selected_frames
        else pd.DataFrame()
    )
    weekly = _weekly_summary(selected_rows)
    calibration = (
        pd.concat([frame for frame in calibration_frames if not frame.empty], ignore_index=True)
        if calibration_frames
        else pd.DataFrame()
    )
    aggregate = _aggregate(monthly, weekly, expected_months=len(months))

    paths = {
        "monthly": output_dir / "timeout_holding_risk_label_monthly.csv",
        "weekly": output_dir / "timeout_holding_risk_label_weekly.csv",
        "aggregate": output_dir / "timeout_holding_risk_label_aggregate.csv",
        "diagnostics": output_dir / "timeout_holding_risk_label_diagnostics.csv",
        "calibration": output_dir / "timeout_holding_risk_calibration_deciles.csv",
        "selected_rows_parquet": output_dir / "timeout_holding_risk_selected_rows.parquet",
        "selected_rows_csv": output_dir / "timeout_holding_risk_selected_rows.csv",
        "manifest": output_dir / "manifest.json",
    }
    monthly.to_csv(paths["monthly"], index=False)
    weekly.to_csv(paths["weekly"], index=False)
    aggregate.to_csv(paths["aggregate"], index=False)
    diagnostics_frame.to_csv(paths["diagnostics"], index=False)
    calibration.to_csv(paths["calibration"], index=False)
    selected_rows.to_parquet(paths["selected_rows_parquet"], index=False)
    selected_rows.to_csv(paths["selected_rows_csv"], index=False)
    manifest = {
        "scope": "timeout_holding_risk_label_diagnostic",
        "quality_labels_path": str(quality_labels_path),
        "labels_path": str(labels_path),
        "config_path": str(config_path),
        "output_dir": str(output_dir),
        "rows": int(len(frame)),
        "utility_source": metrics.attrs.get("utility_source"),
        "months": list(months),
        "labels": labels_requested,
        "horizon_hours": {"candidate_min": 3, "candidate_max": 7, "utility_window_corrected": "1-6h"},
        "diagnostic_only": True,
        "side_counts": side_counts,
        "feature_sets": feature_sets,
        "high_risk_top_fracs": [float(v) for v in high_risk_top_fracs],
        "low_risk_keep_fracs": [float(v) for v in low_risk_keep_fracs],
        "seeds": [int(seed) for seed in seeds],
        "join_report": join_report,
        "feature_store": feature_report,
        "base_feature_count": int(len(base_features)),
        "source_feature_count": int(len(source_features)),
        "outputs": {key: str(path) for key, path in paths.items()},
    }
    paths["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    markdown = _write_report(output_dir, aggregate, diagnostics_frame, calibration, manifest)
    manifest["outputs"]["markdown"] = str(markdown)
    paths["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quality-labels-path", type=Path, default=DEFAULT_QUALITY_LABELS)
    parser.add_argument("--labels-path", type=Path, default=DEFAULT_LABELS_DIR)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--feature-list-csv", type=Path, default=DEFAULT_FEATURE_LIST_CSV)
    parser.add_argument("--max-feature-store-features", type=int, default=96)
    parser.add_argument("--months", type=str, default=",".join(DEFAULT_MONTHS))
    parser.add_argument("--labels", type=str, default=",".join(DEFAULT_LABELS))
    parser.add_argument("--feature-sets", type=str, default=",".join(DEFAULT_FEATURE_SETS))
    parser.add_argument("--high-risk-top-fracs", type=str, default=",".join(str(v) for v in DEFAULT_HIGH_RISK_TOP_FRACS))
    parser.add_argument("--low-risk-keep-fracs", type=str, default=",".join(str(v) for v in DEFAULT_LOW_RISK_KEEP_FRACS))
    parser.add_argument("--seeds", type=str, default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--train-lookback-months", type=int, default=None)
    parser.add_argument("--min-train-rows", type=int, default=500)
    parser.add_argument("--min-valid-rows", type=int, default=100)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = run_diagnostic(
        quality_labels_path=args.quality_labels_path,
        labels_path=args.labels_path,
        config_path=args.config,
        output_dir=args.output_dir,
        feature_dir=args.feature_dir,
        feature_list_csv=args.feature_list_csv,
        max_feature_store_features=args.max_feature_store_features,
        months=_parse_csv(args.months, DEFAULT_MONTHS),
        labels_requested=_parse_csv(args.labels, DEFAULT_LABELS),
        feature_sets=_parse_csv(args.feature_sets, DEFAULT_FEATURE_SETS),
        high_risk_top_fracs=_parse_float_csv(args.high_risk_top_fracs, DEFAULT_HIGH_RISK_TOP_FRACS),
        low_risk_keep_fracs=_parse_float_csv(args.low_risk_keep_fracs, DEFAULT_LOW_RISK_KEEP_FRACS),
        seeds=_parse_int_csv(args.seeds, DEFAULT_SEEDS),
        train_lookback_months=args.train_lookback_months,
        min_train_rows=int(args.min_train_rows),
        min_valid_rows=int(args.min_valid_rows),
    )
    print(json.dumps(_json_safe(manifest), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
