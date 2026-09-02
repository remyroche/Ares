#!/usr/bin/env python3
"""Audit market-state strategy-response model quality.

This audit is intentionally replay-free. It checks whether the fold-fitted
strategy-response models learned useful residual utility and risk structure
before any portfolio controller is considered for activation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_OUTPUT_DIR = Path("data_perp/reports/market_state_strategy_response_quality_20260626")

REQUIRED_RESPONSE_COLUMNS = {
    "timestamp",
    "strategy_id",
    "head",
    "_rank",
    "_threshold",
    "_is_full_sl",
    "_is_timeout",
    "fold",
    "arm",
    "actual_resid_utility",
    "actual_resid_full_sl",
    "actual_resid_timeout",
    "pred_resid_utility",
    "pred_resid_utility_lcb",
    "pred_resid_full_sl",
    "pred_resid_timeout",
    "pred_full_sl",
    "pred_timeout",
}

DEFAULT_POLICY = {
    "min_total_rows": 100,
    "min_fold_rows": 30,
    "min_timestamp_count": 10,
    "min_mean_coverage": 0.80,
    "max_state_ood_share": 0.10,
    "min_median_utility_ic": 0.0,
    "min_q25_utility_ic": -0.02,
    "min_positive_utility_ic_share": 0.50,
    "min_median_utility_decile_spread": 0.0,
    "min_q25_utility_decile_spread": -0.005,
    "max_median_full_sl_calibration_error": 0.20,
    "max_median_timeout_calibration_error": 0.20,
    "frontier_band": 0.10,
}

SUPPORT_FAIL_REASONS = {
    "insufficient_total_rows",
    "insufficient_fold_rows",
    "insufficient_timestamps",
}
QUALITY_EPS = 1e-12


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
        return value if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _sha256(path: Path | None) -> str | None:
    if path is None or not path.exists() or path.is_dir():
        return None
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _finite_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype="float64")
    out = pd.to_numeric(frame[column], errors="coerce").astype("float64")
    return out.replace([np.inf, -np.inf], np.nan)


def _safe_mean(values: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        return float("nan")
    return float(values.mean())


def _safe_spearman(pred: pd.Series, actual: pd.Series) -> float:
    data = pd.DataFrame({"pred": pred, "actual": actual}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(data) < 3:
        return float("nan")
    if data["pred"].nunique(dropna=True) < 2 or data["actual"].nunique(dropna=True) < 2:
        return float("nan")
    return float(data["pred"].rank(method="average").corr(data["actual"].rank(method="average")))


def _top_bottom_spread(pred: pd.Series, actual: pd.Series, *, top_frac: float = 0.10) -> tuple[float, float, float, int, int]:
    data = pd.DataFrame({"pred": pred, "actual": actual}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(data) < 10 or data["pred"].nunique(dropna=True) < 2:
        return float("nan"), float("nan"), float("nan"), 0, 0
    ranks = data["pred"].rank(method="average", pct=True)
    bottom = data.loc[ranks <= top_frac, "actual"]
    top = data.loc[ranks >= 1.0 - top_frac, "actual"]
    if bottom.empty or top.empty:
        return float("nan"), float("nan"), float("nan"), int(len(top)), int(len(bottom))
    top_mean = float(top.mean())
    bottom_mean = float(bottom.mean())
    return top_mean - bottom_mean, top_mean, bottom_mean, int(len(top)), int(len(bottom))


def _brier_and_calibration(pred: pd.Series, actual: pd.Series) -> tuple[float, float, float, float]:
    data = pd.DataFrame({"pred": pred, "actual": actual}).replace([np.inf, -np.inf], np.nan).dropna()
    if data.empty:
        return float("nan"), float("nan"), float("nan"), float("nan")
    pred_clipped = data["pred"].clip(0.0, 1.0)
    actual_float = data["actual"].clip(0.0, 1.0)
    brier = float(np.mean(np.square(pred_clipped - actual_float)))
    pred_mean = float(pred_clipped.mean())
    actual_mean = float(actual_float.mean())
    return brier, abs(pred_mean - actual_mean), pred_mean, actual_mean


def _value_counts_text(values: pd.Series) -> str:
    if values.empty:
        return ""
    counts = values.dropna().astype(str).value_counts()
    return ";".join(f"{idx}:{int(val)}" for idx, val in counts.items())


def _response_fold_metrics(response: pd.DataFrame, policy: dict[str, Any]) -> pd.DataFrame:
    frontier_band = float(policy["frontier_band"])
    rows: list[dict[str, Any]] = []
    group_cols = ["arm", "head", "fold"]
    for (arm, head, fold), part in response.groupby(group_cols, dropna=False, sort=False):
        part = part.copy()
        rank = _finite_series(part, "_rank")
        threshold = _finite_series(part, "_threshold")
        selected = rank >= threshold
        frontier = rank >= (threshold - frontier_band)
        utility_spread, utility_top, utility_bottom, top_n, bottom_n = _top_bottom_spread(
            _finite_series(part, "pred_resid_utility"),
            _finite_series(part, "actual_resid_utility"),
        )
        lcb_spread, lcb_top, lcb_bottom, _lcb_top_n, _lcb_bottom_n = _top_bottom_spread(
            _finite_series(part, "pred_resid_utility_lcb"),
            _finite_series(part, "actual_resid_utility"),
        )
        full_sl_spread, full_sl_top, full_sl_bottom, _risk_top_n, _risk_bottom_n = _top_bottom_spread(
            _finite_series(part, "pred_resid_full_sl"),
            _finite_series(part, "actual_resid_full_sl"),
        )
        timeout_spread, timeout_top, timeout_bottom, _timeout_top_n, _timeout_bottom_n = _top_bottom_spread(
            _finite_series(part, "pred_resid_timeout"),
            _finite_series(part, "actual_resid_timeout"),
        )
        full_sl_brier, full_sl_cal_error, pred_full_sl_rate, actual_full_sl_rate = _brier_and_calibration(
            _finite_series(part, "pred_full_sl"),
            _finite_series(part, "_is_full_sl"),
        )
        timeout_brier, timeout_cal_error, pred_timeout_rate, actual_timeout_rate = _brier_and_calibration(
            _finite_series(part, "pred_timeout"),
            _finite_series(part, "_is_timeout"),
        )
        rows.append(
            {
                "arm": str(arm),
                "head": str(head),
                "fold": fold,
                "rows": int(len(part)),
                "timestamp_count": int(pd.to_datetime(part["timestamp"], utc=True, errors="coerce").nunique()),
                "strategy_count": int(part["strategy_id"].nunique(dropna=True)),
                "selected_rows": int(selected.fillna(False).sum()),
                "frontier_rows": int(frontier.fillna(False).sum()),
                "selected_share": float(selected.fillna(False).mean()) if len(part) else float("nan"),
                "frontier_share": float(frontier.fillna(False).mean()) if len(part) else float("nan"),
                "mean_state_feature_coverage": _safe_mean(_finite_series(part, "state_feature_coverage")),
                "mean_response_feature_coverage": _safe_mean(_finite_series(part, "response_feature_coverage")),
                "state_ood_share": _safe_mean(_finite_series(part, "state_ood_flag").fillna(0.0)),
                "mean_state_ood_score": _safe_mean(_finite_series(part, "state_ood_score")),
                "utility_spearman": _safe_spearman(
                    _finite_series(part, "pred_resid_utility"),
                    _finite_series(part, "actual_resid_utility"),
                ),
                "utility_lcb_spearman": _safe_spearman(
                    _finite_series(part, "pred_resid_utility_lcb"),
                    _finite_series(part, "actual_resid_utility"),
                ),
                "utility_decile_spread": utility_spread,
                "utility_top_decile_actual": utility_top,
                "utility_bottom_decile_actual": utility_bottom,
                "utility_top_decile_rows": top_n,
                "utility_bottom_decile_rows": bottom_n,
                "utility_lcb_decile_spread": lcb_spread,
                "utility_lcb_top_decile_actual": lcb_top,
                "utility_lcb_bottom_decile_actual": lcb_bottom,
                "full_sl_resid_spearman": _safe_spearman(
                    _finite_series(part, "pred_resid_full_sl"),
                    _finite_series(part, "actual_resid_full_sl"),
                ),
                "timeout_resid_spearman": _safe_spearman(
                    _finite_series(part, "pred_resid_timeout"),
                    _finite_series(part, "actual_resid_timeout"),
                ),
                "full_sl_resid_decile_spread": full_sl_spread,
                "full_sl_top_decile_actual_resid": full_sl_top,
                "full_sl_bottom_decile_actual_resid": full_sl_bottom,
                "timeout_resid_decile_spread": timeout_spread,
                "timeout_top_decile_actual_resid": timeout_top,
                "timeout_bottom_decile_actual_resid": timeout_bottom,
                "full_sl_brier": full_sl_brier,
                "full_sl_calibration_error": full_sl_cal_error,
                "pred_full_sl_rate": pred_full_sl_rate,
                "actual_full_sl_rate": actual_full_sl_rate,
                "timeout_brier": timeout_brier,
                "timeout_calibration_error": timeout_cal_error,
                "pred_timeout_rate": pred_timeout_rate,
                "actual_timeout_rate": actual_timeout_rate,
                "state_prediction_contracts": _value_counts_text(part.get("state_prediction_contract", pd.Series(dtype=str))),
            }
        )
    return pd.DataFrame(rows)


def _quantile(series: pd.Series, q: float) -> float:
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        return float("nan")
    return float(values.quantile(q))


def _median(series: pd.Series) -> float:
    return _quantile(series, 0.5)


def _first_non_empty(series: pd.Series) -> str:
    values = series.dropna().astype(str)
    values = values.loc[values.ne("")]
    return "" if values.empty else str(values.iloc[0])


def _compute_quality_reasons(row: pd.Series, policy: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if float(row.get("rows_total", 0.0) or 0.0) + QUALITY_EPS < float(policy["min_total_rows"]):
        reasons.append("insufficient_total_rows")
    if float(row.get("min_fold_rows", 0.0) or 0.0) + QUALITY_EPS < float(policy["min_fold_rows"]):
        reasons.append("insufficient_fold_rows")
    if float(row.get("timestamp_count_total", 0.0) or 0.0) + QUALITY_EPS < float(policy["min_timestamp_count"]):
        reasons.append("insufficient_timestamps")
    if float(row.get("mean_response_feature_coverage", 0.0) or 0.0) + QUALITY_EPS < float(policy["min_mean_coverage"]):
        reasons.append("low_response_feature_coverage")
    if float(row.get("mean_state_feature_coverage", 0.0) or 0.0) + QUALITY_EPS < float(policy["min_mean_coverage"]):
        reasons.append("low_state_feature_coverage")
    if float(row.get("mean_state_ood_share", 0.0) or 0.0) > float(policy["max_state_ood_share"]) + QUALITY_EPS:
        reasons.append("state_ood_share_too_high")
    if float(row.get("median_utility_spearman", -np.inf) or -np.inf) <= float(policy["min_median_utility_ic"]):
        reasons.append("median_utility_ic_not_positive")
    if float(row.get("q25_utility_spearman", -np.inf) or -np.inf) + QUALITY_EPS < float(policy["min_q25_utility_ic"]):
        reasons.append("q25_utility_ic_too_low")
    if float(row.get("positive_utility_ic_share", 0.0) or 0.0) + QUALITY_EPS < float(policy["min_positive_utility_ic_share"]):
        reasons.append("utility_ic_not_recurrent")
    if float(row.get("median_utility_decile_spread", -np.inf) or -np.inf) <= float(policy["min_median_utility_decile_spread"]):
        reasons.append("median_decile_spread_not_positive")
    if float(row.get("q25_utility_decile_spread", -np.inf) or -np.inf) + QUALITY_EPS < float(policy["min_q25_utility_decile_spread"]):
        reasons.append("q25_decile_spread_too_low")
    if float(row.get("median_full_sl_calibration_error", np.inf) or np.inf) > (
        float(policy["max_median_full_sl_calibration_error"]) + QUALITY_EPS
    ):
        reasons.append("full_sl_calibration_error_too_high")
    if float(row.get("median_timeout_calibration_error", np.inf) or np.inf) > (
        float(policy["max_median_timeout_calibration_error"]) + QUALITY_EPS
    ):
        reasons.append("timeout_calibration_error_too_high")
    return reasons


def _response_head_metrics(fold_metrics: pd.DataFrame, policy: dict[str, Any]) -> pd.DataFrame:
    if fold_metrics.empty:
        return pd.DataFrame()
    min_fold_rows_required = int(policy["min_fold_rows"])
    grouped = (
        fold_metrics.groupby(["arm", "head"], dropna=False, sort=False)
        .agg(
            folds=("fold", "nunique"),
            rows_total=("rows", "sum"),
            min_fold_rows=("rows", "min"),
            timestamp_count_total=("timestamp_count", "sum"),
            strategy_count_max=("strategy_count", "max"),
            selected_rows_total=("selected_rows", "sum"),
            frontier_rows_total=("frontier_rows", "sum"),
            mean_selected_share=("selected_share", "mean"),
            mean_frontier_share=("frontier_share", "mean"),
            mean_state_feature_coverage=("mean_state_feature_coverage", "mean"),
            mean_response_feature_coverage=("mean_response_feature_coverage", "mean"),
            mean_state_ood_share=("state_ood_share", "mean"),
            max_state_ood_share=("state_ood_share", "max"),
            median_utility_spearman=("utility_spearman", _median),
            q25_utility_spearman=("utility_spearman", lambda x: _quantile(x, 0.25)),
            positive_utility_ic_share=("utility_spearman", lambda x: float((pd.to_numeric(x, errors="coerce") > 0).mean())),
            median_utility_lcb_spearman=("utility_lcb_spearman", _median),
            median_utility_decile_spread=("utility_decile_spread", _median),
            q25_utility_decile_spread=("utility_decile_spread", lambda x: _quantile(x, 0.25)),
            positive_decile_spread_share=(
                "utility_decile_spread",
                lambda x: float((pd.to_numeric(x, errors="coerce") > 0).mean()),
            ),
            median_utility_lcb_decile_spread=("utility_lcb_decile_spread", _median),
            median_full_sl_resid_spearman=("full_sl_resid_spearman", _median),
            median_timeout_resid_spearman=("timeout_resid_spearman", _median),
            median_full_sl_resid_decile_spread=("full_sl_resid_decile_spread", _median),
            median_timeout_resid_decile_spread=("timeout_resid_decile_spread", _median),
            median_full_sl_brier=("full_sl_brier", _median),
            median_full_sl_calibration_error=("full_sl_calibration_error", _median),
            median_timeout_brier=("timeout_brier", _median),
            median_timeout_calibration_error=("timeout_calibration_error", _median),
            contracts=("state_prediction_contracts", _first_non_empty),
        )
        .reset_index()
    )
    support_rows: list[dict[str, Any]] = []
    for (arm, head), part in fold_metrics.groupby(["arm", "head"], dropna=False, sort=False):
        fold_rows = pd.to_numeric(part["rows"], errors="coerce").fillna(0).astype(int)
        too_small = part.loc[fold_rows < min_fold_rows_required, ["fold", "rows"]].copy()
        support_rows.append(
            {
                "arm": str(arm),
                "head": str(head),
                "min_fold_rows_required": min_fold_rows_required,
                "folds_below_min_rows": int(len(too_small)),
                "under_supported_folds": ";".join(
                    f"{row.fold}:{int(row.rows)}" for row in too_small.itertuples(index=False)
                ),
            }
        )
    if support_rows:
        grouped = grouped.merge(pd.DataFrame(support_rows), on=["arm", "head"], how="left")
    reasons = []
    support_reasons = []
    non_support_reasons = []
    passed = []
    support_passed = []
    signal_passed = []
    for _, row in grouped.iterrows():
        row_reasons = _compute_quality_reasons(row, policy)
        row_support_reasons = [reason for reason in row_reasons if reason in SUPPORT_FAIL_REASONS]
        row_non_support_reasons = [reason for reason in row_reasons if reason not in SUPPORT_FAIL_REASONS]
        reasons.append(";".join(row_reasons))
        support_reasons.append(";".join(row_support_reasons))
        non_support_reasons.append(";".join(row_non_support_reasons))
        passed.append(not row_reasons)
        support_passed.append(not row_support_reasons)
        signal_passed.append(not row_non_support_reasons)
    grouped["response_quality_passed"] = passed
    grouped["response_support_passed"] = support_passed
    grouped["response_signal_passed"] = signal_passed
    grouped["response_quality_fail_reasons"] = reasons
    grouped["response_support_fail_reasons"] = support_reasons
    grouped["response_signal_fail_reasons"] = non_support_reasons
    return grouped


def _state_effect_summary(artifact_dir: Path, output_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = artifact_dir / "strategy_state_effect_matrix.csv"
    if not path.exists():
        return pd.DataFrame(), {"exists": False}
    effects = pd.read_csv(path)
    required = {"fold", "arm", "scope", "scope_value", "state_feature", "target", "rows", "spearman", "target_q90_minus_q10"}
    if not required.issubset(effects.columns):
        return pd.DataFrame(), {"exists": True, "error": "missing required effect columns"}
    head_effects = effects.loc[effects["scope"].astype(str).eq("head")].copy()
    if head_effects.empty:
        return pd.DataFrame(), {"exists": True, "rows": int(len(effects)), "head_rows": 0}
    head_effects["abs_spearman"] = pd.to_numeric(head_effects["spearman"], errors="coerce").abs()
    head_effects["abs_q90_minus_q10"] = pd.to_numeric(head_effects["target_q90_minus_q10"], errors="coerce").abs()
    summary = (
        head_effects.groupby(["arm", "scope_value", "state_feature", "target"], dropna=False, sort=False)
        .agg(
            folds=("fold", "nunique"),
            rows_median=("rows", "median"),
            median_abs_spearman=("abs_spearman", "median"),
            max_abs_spearman=("abs_spearman", "max"),
            median_abs_q90_minus_q10=("abs_q90_minus_q10", "median"),
        )
        .reset_index()
        .rename(columns={"scope_value": "head"})
        .sort_values(["median_abs_spearman", "median_abs_q90_minus_q10"], ascending=[False, False])
    )
    summary.to_csv(output_dir / "market_state_strategy_response_top_state_effects.csv", index=False)
    payload = {
        "exists": True,
        "rows": int(len(effects)),
        "head_rows": int(len(head_effects)),
        "top_effects_written": int(len(summary)),
    }
    return summary, payload


def _arm_summary(head_metrics: pd.DataFrame) -> pd.DataFrame:
    if head_metrics.empty:
        return pd.DataFrame()
    grouped = (
        head_metrics.groupby("arm", dropna=False, sort=False)
        .agg(
            heads=("head", "nunique"),
            passed_heads=("response_quality_passed", "sum"),
            rows_total=("rows_total", "sum"),
            median_utility_spearman=("median_utility_spearman", "median"),
            min_q25_utility_spearman=("q25_utility_spearman", "min"),
            median_utility_decile_spread=("median_utility_decile_spread", "median"),
            min_q25_utility_decile_spread=("q25_utility_decile_spread", "min"),
            mean_state_ood_share=("mean_state_ood_share", "mean"),
            mean_response_feature_coverage=("mean_response_feature_coverage", "mean"),
        )
        .reset_index()
    )
    grouped["all_heads_passed_response_quality"] = grouped["passed_heads"].eq(grouped["heads"])
    return grouped


def _parse_under_supported_folds(value: Any, *, min_fold_rows: int) -> tuple[int, str]:
    text = "" if value is None or pd.isna(value) else str(value)
    if not text:
        return 0, ""
    total_needed = 0
    parts: list[str] = []
    for item in text.split(";"):
        if not item or ":" not in item:
            continue
        fold, rows_text = item.split(":", 1)
        rows = pd.to_numeric(pd.Series([rows_text]), errors="coerce").iloc[0]
        if not np.isfinite(rows):
            continue
        needed = max(0, int(min_fold_rows) - int(rows))
        total_needed += needed
        parts.append(f"{fold}:+{needed}")
    return total_needed, ";".join(parts)


def _support_margin(value: Any, threshold: Any) -> float:
    value_num = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    threshold_num = pd.to_numeric(pd.Series([threshold]), errors="coerce").iloc[0]
    if not np.isfinite(value_num) or not np.isfinite(threshold_num):
        return float("nan")
    return float(value_num - threshold_num)


def _risk_margin(value: Any, maximum: Any) -> float:
    value_num = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    max_num = pd.to_numeric(pd.Series([maximum]), errors="coerce").iloc[0]
    if not np.isfinite(value_num) or not np.isfinite(max_num):
        return float("nan")
    return float(max_num - value_num)


def _response_blocker_summary(head_metrics: pd.DataFrame, policy: dict[str, Any]) -> pd.DataFrame:
    if head_metrics.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for _, row in head_metrics.iterrows():
        support_passed = bool(row.get("response_support_passed"))
        signal_passed = bool(row.get("response_signal_passed"))
        quality_passed = bool(row.get("response_quality_passed"))
        if quality_passed:
            blocker_type = "passed"
            next_action = "eligible_for_replay_gate"
        elif not support_passed and signal_passed:
            blocker_type = "support_only"
            next_action = (
                "rerun_or_accumulate_more_matured_rows_for_under_supported_folds; "
                "do_not_relax_quality_gate"
            )
        elif support_passed and not signal_passed:
            blocker_type = "signal_quality"
            next_action = (
                "improve_response_features_or_model_objective_before_controller_replay; "
                "support_is_sufficient"
            )
        else:
            blocker_type = "support_and_signal"
            next_action = (
                "increase_support_first_then_reassess_signal; do_not_promote_controller"
            )
        min_fold_rows = int(row.get("min_fold_rows_required", policy["min_fold_rows"]) or policy["min_fold_rows"])
        extra_fold_rows, extra_fold_rows_by_fold = _parse_under_supported_folds(
            row.get("under_supported_folds"),
            min_fold_rows=min_fold_rows,
        )
        rows_total = float(row.get("rows_total", 0.0) or 0.0)
        timestamp_total = float(row.get("timestamp_count_total", 0.0) or 0.0)
        rows.append(
            {
                "arm": str(row.get("arm")),
                "head": str(row.get("head")),
                "blocker_type": blocker_type,
                "response_quality_passed": quality_passed,
                "response_support_passed": support_passed,
                "response_signal_passed": signal_passed,
                "quality_fail_reasons": row.get("response_quality_fail_reasons", ""),
                "support_fail_reasons": row.get("response_support_fail_reasons", ""),
                "signal_fail_reasons": row.get("response_signal_fail_reasons", ""),
                "rows_total": int(rows_total),
                "min_total_rows_required": int(policy["min_total_rows"]),
                "rows_total_margin": _support_margin(rows_total, policy["min_total_rows"]),
                "timestamp_count_total": int(timestamp_total),
                "min_timestamp_count_required": int(policy["min_timestamp_count"]),
                "timestamp_count_margin": _support_margin(timestamp_total, policy["min_timestamp_count"]),
                "min_fold_rows": int(row.get("min_fold_rows", 0) or 0),
                "min_fold_rows_required": min_fold_rows,
                "min_fold_rows_margin": _support_margin(row.get("min_fold_rows"), min_fold_rows),
                "folds_below_min_rows": int(row.get("folds_below_min_rows", 0) or 0),
                "under_supported_folds": row.get("under_supported_folds", ""),
                "required_extra_rows_by_fold": extra_fold_rows_by_fold,
                "required_extra_rows_total_to_clear_support": int(extra_fold_rows),
                "mean_response_feature_coverage": row.get("mean_response_feature_coverage"),
                "response_feature_coverage_margin": _support_margin(
                    row.get("mean_response_feature_coverage"),
                    policy["min_mean_coverage"],
                ),
                "mean_state_feature_coverage": row.get("mean_state_feature_coverage"),
                "state_feature_coverage_margin": _support_margin(
                    row.get("mean_state_feature_coverage"),
                    policy["min_mean_coverage"],
                ),
                "mean_state_ood_share": row.get("mean_state_ood_share"),
                "state_ood_share_margin": _risk_margin(
                    row.get("mean_state_ood_share"),
                    policy["max_state_ood_share"],
                ),
                "median_utility_spearman": row.get("median_utility_spearman"),
                "median_utility_ic_margin": _support_margin(
                    row.get("median_utility_spearman"),
                    policy["min_median_utility_ic"],
                ),
                "q25_utility_spearman": row.get("q25_utility_spearman"),
                "q25_utility_ic_margin": _support_margin(
                    row.get("q25_utility_spearman"),
                    policy["min_q25_utility_ic"],
                ),
                "positive_utility_ic_share": row.get("positive_utility_ic_share"),
                "positive_utility_ic_share_margin": _support_margin(
                    row.get("positive_utility_ic_share"),
                    policy["min_positive_utility_ic_share"],
                ),
                "median_utility_decile_spread": row.get("median_utility_decile_spread"),
                "median_decile_spread_margin": _support_margin(
                    row.get("median_utility_decile_spread"),
                    policy["min_median_utility_decile_spread"],
                ),
                "q25_utility_decile_spread": row.get("q25_utility_decile_spread"),
                "q25_decile_spread_margin": _support_margin(
                    row.get("q25_utility_decile_spread"),
                    policy["min_q25_utility_decile_spread"],
                ),
                "median_full_sl_calibration_error": row.get("median_full_sl_calibration_error"),
                "full_sl_calibration_margin": _risk_margin(
                    row.get("median_full_sl_calibration_error"),
                    policy["max_median_full_sl_calibration_error"],
                ),
                "median_timeout_calibration_error": row.get("median_timeout_calibration_error"),
                "timeout_calibration_margin": _risk_margin(
                    row.get("median_timeout_calibration_error"),
                    policy["max_median_timeout_calibration_error"],
                ),
                "next_action": next_action,
                "promotion_waiver_allowed": False,
            }
        )
    summary = pd.DataFrame(rows)
    order = {"passed": 0, "support_only": 1, "signal_quality": 2, "support_and_signal": 3}
    summary["_order"] = summary["blocker_type"].map(order).fillna(9)
    summary = summary.sort_values(
        ["_order", "required_extra_rows_total_to_clear_support", "arm", "head"],
        ascending=[True, True, True, True],
    ).drop(columns=["_order"])
    return summary.reset_index(drop=True)


def audit_strategy_response_quality(
    artifact_dir: Path,
    output_dir: Path,
    *,
    policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    policy = {**DEFAULT_POLICY, **(policy or {})}
    output_dir.mkdir(parents=True, exist_ok=True)
    response_path = artifact_dir / "strategy_response_oof_predictions.parquet"
    failures: list[str] = []
    if not response_path.exists():
        payload = {
            "artifact_dir": str(artifact_dir),
            "output_dir": str(output_dir),
            "passed": False,
            "failures": [f"missing response predictions: {response_path}"],
        }
        (output_dir / "market_state_strategy_response_quality_gate.json").write_text(
            json.dumps(_json_safe(payload), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return payload

    response = pd.read_parquet(response_path)
    missing = sorted(REQUIRED_RESPONSE_COLUMNS.difference(response.columns))
    if missing:
        failures.append(f"response predictions missing required columns: {missing}")
        response = response.copy()
        for col in missing:
            response[col] = np.nan

    if "timestamp" in response.columns:
        response["timestamp"] = pd.to_datetime(response["timestamp"], utc=True, errors="coerce")
    duplicate_keys = ["arm", "fold", "timestamp", "strategy_id", "symbol"]
    duplicate_key_cols = [col for col in duplicate_keys if col in response.columns]
    duplicate_rows = int(response.duplicated(duplicate_key_cols).sum()) if duplicate_key_cols else 0
    if duplicate_rows:
        failures.append(f"duplicate response rows by {duplicate_key_cols}: {duplicate_rows}")

    fold_metrics = _response_fold_metrics(response, policy)
    head_metrics = _response_head_metrics(fold_metrics, policy)
    arm_metrics = _arm_summary(head_metrics)
    blocker_summary = _response_blocker_summary(head_metrics, policy)
    top_effects, effects_payload = _state_effect_summary(artifact_dir, output_dir)

    fold_metrics.to_csv(output_dir / "market_state_strategy_response_quality_by_fold.csv", index=False)
    head_metrics.to_csv(output_dir / "market_state_strategy_response_quality_by_head.csv", index=False)
    arm_metrics.to_csv(output_dir / "market_state_strategy_response_quality_by_arm.csv", index=False)
    blocker_summary.to_csv(output_dir / "market_state_strategy_response_gate_blockers.csv", index=False)

    passing_arms = (
        arm_metrics.loc[arm_metrics.get("all_heads_passed_response_quality", pd.Series(dtype=bool)).fillna(False), "arm"]
        .astype(str)
        .tolist()
        if not arm_metrics.empty
        else []
    )
    quality_passing_heads = (
        sorted(
            head_metrics.loc[
                head_metrics.get("response_quality_passed", pd.Series(dtype=bool)).fillna(False),
                "head",
            ]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )
        if not head_metrics.empty
        else []
    )
    support_blocked_heads = (
        sorted(
            head_metrics.loc[
                ~head_metrics.get("response_support_passed", pd.Series(dtype=bool)).fillna(False),
                "head",
            ]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )
        if not head_metrics.empty
        else []
    )
    signal_passing_but_support_blocked_heads = (
        sorted(
            head_metrics.loc[
                head_metrics.get("response_signal_passed", pd.Series(dtype=bool)).fillna(False)
                & ~head_metrics.get("response_support_passed", pd.Series(dtype=bool)).fillna(False),
                "head",
            ]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )
        if not head_metrics.empty
        else []
    )
    head_pass_counts = (
        head_metrics.loc[head_metrics.get("response_quality_passed", pd.Series(dtype=bool)).fillna(False), "head"]
        .dropna()
        .astype(str)
        .value_counts()
        .to_dict()
        if not head_metrics.empty
        else {}
    )
    fail_reason_counts: dict[str, int] = {}
    if not head_metrics.empty and "response_quality_fail_reasons" in head_metrics.columns:
        for reasons in head_metrics["response_quality_fail_reasons"].dropna().astype(str):
            for reason in reasons.split(";"):
                if reason:
                    fail_reason_counts[reason] = fail_reason_counts.get(reason, 0) + 1
    blocker_counts = (
        blocker_summary["blocker_type"].value_counts().to_dict()
        if not blocker_summary.empty and "blocker_type" in blocker_summary.columns
        else {}
    )
    support_only = (
        blocker_summary.loc[blocker_summary["blocker_type"].eq("support_only")]
        if not blocker_summary.empty
        else pd.DataFrame()
    )
    min_extra_support_rows = (
        int(pd.to_numeric(support_only["required_extra_rows_total_to_clear_support"], errors="coerce").min())
        if not support_only.empty
        else 0
    )

    payload = {
        "artifact_dir": str(artifact_dir),
        "output_dir": str(output_dir),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "passed": not failures,
        "structural_passed": not failures,
        "quality_gate_passed": bool(passing_arms),
        "controller_activation_allowed": bool(passing_arms) and not failures,
        "failures": failures,
        "policy": policy,
        "response_predictions_sha256": _sha256(response_path),
        "response_rows": int(len(response)),
        "arms": int(response["arm"].nunique(dropna=True)) if "arm" in response.columns else 0,
        "heads": int(response["head"].nunique(dropna=True)) if "head" in response.columns else 0,
        "folds": int(response["fold"].nunique(dropna=True)) if "fold" in response.columns else 0,
        "duplicate_rows": duplicate_rows,
        "quality_passing_arms": passing_arms,
        "quality_passing_arm_count": int(len(passing_arms)),
        "quality_passing_heads": quality_passing_heads,
        "quality_passing_head_count": int(len(quality_passing_heads)),
        "support_blocked_heads": support_blocked_heads,
        "signal_passing_but_support_blocked_heads": signal_passing_but_support_blocked_heads,
        "head_quality_pass_counts": head_pass_counts,
        "head_quality_fail_reason_counts": fail_reason_counts,
        "response_gate_blocker_counts": blocker_counts,
        "support_only_blocked_candidates": int(len(support_only)),
        "min_required_extra_rows_to_clear_support": min_extra_support_rows,
        "response_gate_blocker_summary": blocker_summary.to_dict("records"),
        "state_effect_summary": effects_payload,
    }
    (output_dir / "market_state_strategy_response_quality_gate.json").write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_report(output_dir, payload, arm_metrics, head_metrics, blocker_summary, top_effects)
    return payload


def _format_float(value: Any, digits: int = 4) -> str:
    try:
        val = float(value)
    except Exception:
        return ""
    if not np.isfinite(val):
        return ""
    return f"{val:.{digits}f}"


def _markdown_table(frame: pd.DataFrame, columns: list[str], *, max_rows: int = 20) -> str:
    if frame.empty:
        return "_No rows._\n"
    view = frame.loc[:, [col for col in columns if col in frame.columns]].head(max_rows).copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda x: _format_float(x))
    lines = ["| " + " | ".join(view.columns) + " |", "| " + " | ".join(["---"] * len(view.columns)) + " |"]
    for _, row in view.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in view.columns) + " |")
    return "\n".join(lines) + "\n"


def _write_report(
    output_dir: Path,
    payload: dict[str, Any],
    arm_metrics: pd.DataFrame,
    head_metrics: pd.DataFrame,
    blocker_summary: pd.DataFrame,
    top_effects: pd.DataFrame,
) -> None:
    lines = [
        "# Market-State Strategy-Response Quality Audit",
        "",
        f"Generated: {payload['generated_at']}",
        "",
        "## Summary",
        "",
        f"- Structural audit passed: `{payload['passed']}`",
        f"- Response quality gate passed: `{payload['quality_gate_passed']}`",
        f"- Controller activation allowed by response audit: `{payload['controller_activation_allowed']}`",
        f"- Response rows: `{payload['response_rows']}`",
        f"- Arms: `{payload['arms']}`",
        f"- Heads: `{payload['heads']}`",
        f"- Folds: `{payload['folds']}`",
        f"- Arms passing all head-level response-quality gates: `{payload['quality_passing_arm_count']}`",
        f"- Heads passing at least one response-quality gate: `{payload['quality_passing_head_count']}`",
        f"- Support-blocked heads: `{payload['support_blocked_heads']}`",
        f"- Signal-passing but support-blocked heads: `{payload['signal_passing_but_support_blocked_heads']}`",
        f"- Gate blocker counts: `{payload['response_gate_blocker_counts']}`",
        f"- Support-only blocked candidates: `{payload['support_only_blocked_candidates']}`",
        f"- Minimum extra rows needed to clear support: `{payload['min_required_extra_rows_to_clear_support']}`",
        "",
    ]
    if payload["failures"]:
        lines.extend(["## Structural Failures", ""])
        lines.extend(f"- {failure}" for failure in payload["failures"])
        lines.append("")

    lines.extend(
        [
            "## Arm-Level Quality",
            "",
            _markdown_table(
                arm_metrics.sort_values(
                    ["all_heads_passed_response_quality", "median_utility_spearman"],
                    ascending=[False, False],
                )
                if not arm_metrics.empty
                else arm_metrics,
                [
                    "arm",
                    "heads",
                    "passed_heads",
                    "rows_total",
                    "median_utility_spearman",
                    "min_q25_utility_spearman",
                    "median_utility_decile_spread",
                    "min_q25_utility_decile_spread",
                    "mean_state_ood_share",
                    "mean_response_feature_coverage",
                    "all_heads_passed_response_quality",
                ],
            ),
            "## Response Gate Blockers",
            "",
            _markdown_table(
                blocker_summary if not blocker_summary.empty else blocker_summary,
                [
                    "arm",
                    "head",
                    "blocker_type",
                    "response_support_passed",
                    "response_signal_passed",
                    "quality_fail_reasons",
                    "required_extra_rows_by_fold",
                    "required_extra_rows_total_to_clear_support",
                    "mean_response_feature_coverage",
                    "response_feature_coverage_margin",
                    "median_utility_spearman",
                    "q25_utility_spearman",
                    "median_utility_decile_spread",
                    "next_action",
                    "promotion_waiver_allowed",
                ],
                max_rows=60,
            ),
            "## Head-Level Quality",
            "",
            _markdown_table(
                head_metrics.sort_values(["arm", "head"]) if not head_metrics.empty else head_metrics,
                [
                    "arm",
                    "head",
                    "folds",
                    "rows_total",
                    "median_utility_spearman",
                    "q25_utility_spearman",
                    "positive_utility_ic_share",
                    "median_utility_decile_spread",
                    "q25_utility_decile_spread",
                    "median_full_sl_calibration_error",
                    "median_timeout_calibration_error",
                    "mean_state_ood_share",
                    "min_fold_rows_required",
                    "folds_below_min_rows",
                    "under_supported_folds",
                    "response_support_passed",
                    "response_signal_passed",
                    "response_quality_passed",
                    "response_quality_fail_reasons",
                ],
                max_rows=60,
            ),
        ]
    )
    reason_counts = payload.get("head_quality_fail_reason_counts", {})
    if reason_counts:
        reason_frame = (
            pd.DataFrame([{"reason": key, "count": value} for key, value in reason_counts.items()])
            .sort_values(["count", "reason"], ascending=[False, True])
            .reset_index(drop=True)
        )
        lines.extend(["## Fail Reason Counts", "", _markdown_table(reason_frame, ["reason", "count"])])

    if not top_effects.empty:
        lines.extend(
            [
                "## Strongest State Effects",
                "",
                _markdown_table(
                    top_effects,
                    [
                        "arm",
                        "head",
                        "state_feature",
                        "target",
                        "folds",
                        "rows_median",
                        "median_abs_spearman",
                        "median_abs_q90_minus_q10",
                    ],
                    max_rows=20,
                ),
            ]
        )
    lines.extend(
        [
            "## Interpretation",
            "",
            "This audit is not a promotion decision by itself. A passing response layer means the market-state response model is directionally learnable and calibrated enough to be considered by a replay gate. A failing response layer means threshold or priority modulation should remain logged-only, regardless of isolated replay gains.",
            "",
        ]
    )
    (output_dir / "market_state_strategy_response_quality_report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-total-rows", type=int, default=DEFAULT_POLICY["min_total_rows"])
    parser.add_argument("--min-fold-rows", type=int, default=DEFAULT_POLICY["min_fold_rows"])
    parser.add_argument("--min-timestamp-count", type=int, default=DEFAULT_POLICY["min_timestamp_count"])
    parser.add_argument("--min-mean-coverage", type=float, default=DEFAULT_POLICY["min_mean_coverage"])
    parser.add_argument("--max-state-ood-share", type=float, default=DEFAULT_POLICY["max_state_ood_share"])
    parser.add_argument("--frontier-band", type=float, default=DEFAULT_POLICY["frontier_band"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    policy = {
        "min_total_rows": args.min_total_rows,
        "min_fold_rows": args.min_fold_rows,
        "min_timestamp_count": args.min_timestamp_count,
        "min_mean_coverage": args.min_mean_coverage,
        "max_state_ood_share": args.max_state_ood_share,
        "frontier_band": args.frontier_band,
    }
    payload = audit_strategy_response_quality(args.artifact_dir, args.output_dir, policy=policy)
    print(json.dumps(_json_safe(payload), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
