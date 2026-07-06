#!/usr/bin/env python3
"""Path-risk profile for regime-conditioned label proxy candidates.

This report is intentionally model-free: it reads existing proxy monthly/weekly
outputs and asks whether a label candidate is learnable *inside* economic
execution limits before spending time on base/meta training.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_INPUT_DIR = Path("data_perp/reports/label_regime_conditioned_clean_proxy_s49_s51_q33_badmaegateonly_v1")
DEFAULT_OUTPUT_DIR = Path("data_perp/reports/label_regime_proxy_path_profile_v1")
DEFAULT_MONTHLY_FILENAME = "label_regime_conditioned_clean_proxy_monthly.csv"
DEFAULT_WEEKLY_FILENAME = "label_regime_conditioned_clean_proxy_weekly.csv"

GROUP_COLS = [
    "source",
    "arm",
    "label_arm",
    "weight_arm",
    "target_mode",
    "selection_mode",
    "mae_penalty",
    "bad_mae_penalty",
    "wide_penalty",
    "timeout_penalty",
    "mae_keep_frac",
    "wide_keep_frac",
    "timeout_keep_frac",
    "regime_feature",
    "regime_bin",
    "top_frac",
]


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        out = float(value)
        return out if math.isfinite(out) else None
    return value


def _parse_csv(value: str) -> list[str]:
    return [part.strip() for part in str(value).split(",") if part.strip()]


def _parse_float_csv(value: str) -> list[float]:
    return [float(part.strip()) for part in str(value).split(",") if part.strip()]


def _num(frame: pd.DataFrame, col: str) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[col], errors="coerce").replace([np.inf, -np.inf], np.nan)


def _safe_mean(values: Any) -> float:
    series = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    return float(series.mean()) if len(series) else float("nan")


def _safe_min(values: Any) -> float:
    series = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    return float(series.min()) if len(series) else float("nan")


def _safe_max(values: Any) -> float:
    series = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    return float(series.max()) if len(series) else float("nan")


def _safe_quantile(values: Any, q: float) -> float:
    series = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    return float(series.quantile(q)) if len(series) else float("nan")


def _weighted_mean(frame: pd.DataFrame, value_col: str, weight_col: str) -> float:
    values = _num(frame, value_col)
    weights = _num(frame, weight_col).fillna(0.0)
    mask = values.notna() & weights.gt(0.0)
    if not bool(mask.any()):
        return float("nan")
    return float(np.average(values[mask], weights=weights[mask]))


def _ratio(numerator: float, denominator: float) -> float:
    if not (math.isfinite(numerator) and math.isfinite(denominator)) or abs(denominator) < 1e-12:
        return float("nan")
    return float(numerator / denominator)


def _summarize_month_side(prefix: str, frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {
            f"{prefix}_months": 0,
            f"{prefix}_positive_months": 0,
            f"{prefix}_positive_month_rate": float("nan"),
            f"{prefix}_mean_month_u": float("nan"),
            f"{prefix}_worst_month_u": float("nan"),
            f"{prefix}_row_weighted_mean_u": float("nan"),
            f"{prefix}_selected_rows": 0,
            f"{prefix}_mean_selected_rows": float("nan"),
            f"{prefix}_min_selected_rows": 0,
            f"{prefix}_mean_mae_norm": float("nan"),
            f"{prefix}_max_month_mean_mae_norm": float("nan"),
            f"{prefix}_p90_mae_norm": float("nan"),
            f"{prefix}_max_month_p90_mae_norm": float("nan"),
            f"{prefix}_bad_mae_1r_rate": float("nan"),
            f"{prefix}_max_month_bad_mae_1r_rate": float("nan"),
            f"{prefix}_mean_mfe_norm": float("nan"),
            f"{prefix}_mfe_mae_ratio": float("nan"),
            f"{prefix}_mean_bars_to_mfe": float("nan"),
            f"{prefix}_p90_bars_to_mfe": float("nan"),
            f"{prefix}_mean_barrier": float("nan"),
            f"{prefix}_p90_barrier": float("nan"),
            f"{prefix}_wide_25bps_rate": float("nan"),
            f"{prefix}_max_month_wide_25bps_rate": float("nan"),
            f"{prefix}_timeout_rate": float("nan"),
            f"{prefix}_max_month_timeout_rate": float("nan"),
            f"{prefix}_max_top_symbol_share": float("nan"),
            f"{prefix}_max_top_timestamp_share": float("nan"),
        }
    mean_u = _num(frame, "mean_u")
    rows = _num(frame, "selected_rows").fillna(0.0)
    mean_mae = _weighted_mean(frame, "mean_mae_norm", "selected_rows")
    mean_mfe = _weighted_mean(frame, "mean_mfe_norm", "selected_rows")
    return {
        f"{prefix}_months": int(frame["period"].astype(str).nunique()),
        f"{prefix}_positive_months": int(mean_u.gt(0.0).sum()),
        f"{prefix}_positive_month_rate": float(mean_u.gt(0.0).mean()) if len(mean_u) else float("nan"),
        f"{prefix}_mean_month_u": _safe_mean(mean_u),
        f"{prefix}_worst_month_u": _safe_min(mean_u),
        f"{prefix}_row_weighted_mean_u": _weighted_mean(frame, "mean_u", "selected_rows"),
        f"{prefix}_selected_rows": int(rows.sum()),
        f"{prefix}_mean_selected_rows": _safe_mean(rows),
        f"{prefix}_min_selected_rows": int(rows.min()) if len(rows) else 0,
        f"{prefix}_mean_mae_norm": mean_mae,
        f"{prefix}_max_month_mean_mae_norm": _safe_max(_num(frame, "mean_mae_norm")),
        f"{prefix}_p90_mae_norm": _weighted_mean(frame, "p90_mae_norm", "selected_rows"),
        f"{prefix}_max_month_p90_mae_norm": _safe_max(_num(frame, "p90_mae_norm")),
        f"{prefix}_bad_mae_1r_rate": _weighted_mean(frame, "bad_mae_1r_rate", "selected_rows"),
        f"{prefix}_max_month_bad_mae_1r_rate": _safe_max(_num(frame, "bad_mae_1r_rate")),
        f"{prefix}_mean_mfe_norm": mean_mfe,
        f"{prefix}_mfe_mae_ratio": _ratio(mean_mfe, mean_mae),
        f"{prefix}_mean_bars_to_mfe": _weighted_mean(frame, "mean_bars_to_mfe", "selected_rows"),
        f"{prefix}_p90_bars_to_mfe": _weighted_mean(frame, "p90_bars_to_mfe", "selected_rows"),
        f"{prefix}_mean_barrier": _weighted_mean(frame, "mean_barrier", "selected_rows"),
        f"{prefix}_p90_barrier": _weighted_mean(frame, "p90_barrier", "selected_rows"),
        f"{prefix}_wide_25bps_rate": _weighted_mean(frame, "wide_barrier_25bps_rate", "selected_rows"),
        f"{prefix}_max_month_wide_25bps_rate": _safe_max(_num(frame, "wide_barrier_25bps_rate")),
        f"{prefix}_timeout_rate": _weighted_mean(frame, "timeout_rate", "selected_rows"),
        f"{prefix}_max_month_timeout_rate": _safe_max(_num(frame, "timeout_rate")),
        f"{prefix}_max_top_symbol_share": _safe_max(_num(frame, "top_symbol_share")),
        f"{prefix}_max_top_timestamp_share": _safe_max(_num(frame, "top_timestamp_share")),
    }


def _summarize_week_side(prefix: str, frame: pd.DataFrame, *, min_week_selected_rows: int) -> dict[str, Any]:
    if frame.empty:
        return {
            f"{prefix}_weeks": 0,
            f"{prefix}_material_weeks": 0,
            f"{prefix}_positive_week_rate": float("nan"),
            f"{prefix}_material_positive_week_rate": float("nan"),
            f"{prefix}_row_weighted_week_u": float("nan"),
            f"{prefix}_q10_week_u": float("nan"),
            f"{prefix}_q25_week_u": float("nan"),
            f"{prefix}_worst_week_u": float("nan"),
            f"{prefix}_weekly_bad_mae_1r_rate": float("nan"),
            f"{prefix}_weekly_wide_25bps_rate": float("nan"),
            f"{prefix}_weekly_timeout_rate": float("nan"),
            f"{prefix}_min_week_selected_rows": 0,
            f"{prefix}_max_week_selected_share": float("nan"),
        }
    mean_u = _num(frame, "mean_u")
    week_rows = _num(frame, "week_selected_rows").fillna(0.0)
    material = week_rows >= int(min_week_selected_rows)
    positive = mean_u.gt(0.0)
    return {
        f"{prefix}_weeks": int(len(frame)),
        f"{prefix}_material_weeks": int(material.sum()),
        f"{prefix}_positive_week_rate": float(positive.mean()) if len(positive) else float("nan"),
        f"{prefix}_material_positive_week_rate": float((positive & material).sum() / material.sum())
        if int(material.sum())
        else float("nan"),
        f"{prefix}_row_weighted_week_u": _weighted_mean(frame, "mean_u", "week_selected_rows"),
        f"{prefix}_q10_week_u": _safe_quantile(mean_u, 0.10),
        f"{prefix}_q25_week_u": _safe_quantile(mean_u, 0.25),
        f"{prefix}_worst_week_u": _safe_min(mean_u),
        f"{prefix}_weekly_bad_mae_1r_rate": _weighted_mean(frame, "bad_mae_1r_rate", "week_selected_rows"),
        f"{prefix}_weekly_wide_25bps_rate": _weighted_mean(frame, "wide_barrier_25bps_rate", "week_selected_rows"),
        f"{prefix}_weekly_timeout_rate": _weighted_mean(frame, "timeout_rate", "week_selected_rows"),
        f"{prefix}_min_week_selected_rows": int(week_rows.min()) if len(week_rows) else 0,
        f"{prefix}_max_week_selected_share": _safe_max(_num(frame, "week_selected_share")),
    }


def _decision_reasons(row: dict[str, Any], checks: list[tuple[str, bool]]) -> str:
    failed = [name for name, passed in checks if not bool(passed)]
    return "pass" if not failed else ",".join(failed)


def _add_decisions(
    row: dict[str, Any],
    *,
    fit_months: list[str],
    min_fit_material_weeks: int,
    min_holdout_material_weeks: int,
    min_fit_positive_week_rate: float,
    min_holdout_positive_week_rate: float,
    max_clean_bad_mae_1r_rate: float,
    max_clean_p90_mae_norm: float,
    max_clean_timeout_rate: float,
    max_clean_wide_25bps_rate: float,
    max_bounded_bad_mae_1r_rate: float,
    max_bounded_p90_mae_norm: float,
    min_bounded_mfe_mae_ratio: float,
    max_bounded_timeout_rate: float,
    max_bounded_wide_25bps_rate: float,
) -> None:
    fit_pos_week = float(row.get("fit_material_positive_week_rate", float("nan")))
    holdout_pos_week = float(row.get("holdout_material_positive_week_rate", float("nan")))
    fit_clean_checks = [
        ("fit_months", row.get("fit_months") == len(fit_months)),
        ("fit_month_sign", row.get("fit_positive_months") == len(fit_months) and row.get("fit_worst_month_u", -1.0) > 0.0),
        ("fit_weeks", row.get("fit_material_weeks", 0) >= min_fit_material_weeks),
        ("fit_week_sign", math.isfinite(fit_pos_week) and fit_pos_week >= min_fit_positive_week_rate),
        ("fit_bad_mae", row.get("fit_bad_mae_1r_rate", 1.0) <= max_clean_bad_mae_1r_rate),
        ("fit_p90_mae", row.get("fit_p90_mae_norm", math.inf) <= max_clean_p90_mae_norm),
        ("fit_timeout", row.get("fit_timeout_rate", 1.0) <= max_clean_timeout_rate),
        ("fit_wide", row.get("fit_wide_25bps_rate", 1.0) <= max_clean_wide_25bps_rate),
    ]
    holdout_clean_checks = [
        ("holdout_month_sign", row.get("holdout_mean_month_u", -1.0) > 0.0),
        ("holdout_weeks", row.get("holdout_material_weeks", 0) >= min_holdout_material_weeks),
        ("holdout_week_sign", math.isfinite(holdout_pos_week) and holdout_pos_week >= min_holdout_positive_week_rate),
        ("holdout_bad_mae", row.get("holdout_bad_mae_1r_rate", 1.0) <= max_clean_bad_mae_1r_rate),
        ("holdout_p90_mae", row.get("holdout_p90_mae_norm", math.inf) <= max_clean_p90_mae_norm),
        ("holdout_timeout", row.get("holdout_timeout_rate", 1.0) <= max_clean_timeout_rate),
        ("holdout_wide", row.get("holdout_wide_25bps_rate", 1.0) <= max_clean_wide_25bps_rate),
    ]
    fit_bounded_checks = [
        ("fit_months", row.get("fit_months") == len(fit_months)),
        ("fit_month_sign", row.get("fit_positive_months") == len(fit_months) and row.get("fit_worst_month_u", -1.0) > 0.0),
        ("fit_weeks", row.get("fit_material_weeks", 0) >= min_fit_material_weeks),
        ("fit_week_sign", math.isfinite(fit_pos_week) and fit_pos_week >= min_fit_positive_week_rate),
        ("fit_bad_mae", row.get("fit_bad_mae_1r_rate", 1.0) <= max_bounded_bad_mae_1r_rate),
        ("fit_p90_mae", row.get("fit_p90_mae_norm", math.inf) <= max_bounded_p90_mae_norm),
        ("fit_mfe_mae", row.get("fit_mfe_mae_ratio", -math.inf) >= min_bounded_mfe_mae_ratio),
        ("fit_timeout", row.get("fit_timeout_rate", 1.0) <= max_bounded_timeout_rate),
        ("fit_wide", row.get("fit_wide_25bps_rate", 1.0) <= max_bounded_wide_25bps_rate),
    ]
    holdout_bounded_checks = [
        ("holdout_month_sign", row.get("holdout_mean_month_u", -1.0) > 0.0),
        ("holdout_weeks", row.get("holdout_material_weeks", 0) >= min_holdout_material_weeks),
        ("holdout_week_sign", math.isfinite(holdout_pos_week) and holdout_pos_week >= min_holdout_positive_week_rate),
        ("holdout_bad_mae", row.get("holdout_bad_mae_1r_rate", 1.0) <= max_bounded_bad_mae_1r_rate),
        ("holdout_p90_mae", row.get("holdout_p90_mae_norm", math.inf) <= max_bounded_p90_mae_norm),
        ("holdout_mfe_mae", row.get("holdout_mfe_mae_ratio", -math.inf) >= min_bounded_mfe_mae_ratio),
        ("holdout_timeout", row.get("holdout_timeout_rate", 1.0) <= max_bounded_timeout_rate),
        ("holdout_wide", row.get("holdout_wide_25bps_rate", 1.0) <= max_bounded_wide_25bps_rate),
    ]
    row["fit_clean_reason"] = _decision_reasons(row, fit_clean_checks)
    row["holdout_clean_reason"] = _decision_reasons(row, holdout_clean_checks)
    row["fit_bounded_reason"] = _decision_reasons(row, fit_bounded_checks)
    row["holdout_bounded_reason"] = _decision_reasons(row, holdout_bounded_checks)
    row["fit_clean_pass"] = row["fit_clean_reason"] == "pass"
    row["holdout_clean_standalone_pass"] = row["holdout_clean_reason"] == "pass"
    row["holdout_clean_pass"] = bool(row["fit_clean_pass"] and row["holdout_clean_standalone_pass"])
    row["fit_bounded_pass"] = row["fit_bounded_reason"] == "pass"
    row["holdout_bounded_standalone_pass"] = row["holdout_bounded_reason"] == "pass"
    row["holdout_bounded_pass"] = bool(row["fit_bounded_pass"] and row["holdout_bounded_standalone_pass"])
    row["positive_dirty_holdout"] = bool(
        row.get("fit_positive_months") == len(fit_months)
        and row.get("fit_worst_month_u", -1.0) > 0.0
        and math.isfinite(fit_pos_week)
        and fit_pos_week >= min_fit_positive_week_rate
        and row.get("holdout_mean_month_u", -1.0) > 0.0
        and math.isfinite(holdout_pos_week)
        and holdout_pos_week >= min_holdout_positive_week_rate
        and not row["holdout_bounded_pass"]
    )
    row["economic_proxy_score"] = float(
        1.00 * (row.get("holdout_mean_month_u", 0.0) if pd.notna(row.get("holdout_mean_month_u")) else 0.0)
        + 0.60 * (row.get("holdout_q25_week_u", 0.0) if pd.notna(row.get("holdout_q25_week_u")) else 0.0)
        + 0.30 * (row.get("fit_worst_month_u", 0.0) if pd.notna(row.get("fit_worst_month_u")) else 0.0)
        - 0.030 * (row.get("holdout_bad_mae_1r_rate", 0.0) if pd.notna(row.get("holdout_bad_mae_1r_rate")) else 0.0)
        - 0.004 * (row.get("holdout_p90_mae_norm", 0.0) if pd.notna(row.get("holdout_p90_mae_norm")) else 0.0)
        - 0.020 * (row.get("holdout_timeout_rate", 0.0) if pd.notna(row.get("holdout_timeout_rate")) else 0.0)
        - 0.020 * (row.get("holdout_wide_25bps_rate", 0.0) if pd.notna(row.get("holdout_wide_25bps_rate")) else 0.0)
    )


def _candidate_filter(frame: pd.DataFrame, key_dict: dict[str, Any], group_cols: list[str]) -> pd.Series:
    mask = pd.Series(True, index=frame.index)
    for col in group_cols:
        value = key_dict[col]
        if pd.isna(value):
            mask &= frame[col].isna()
        else:
            mask &= frame[col].eq(value)
    return mask


def summarize_candidates(
    *,
    monthly: pd.DataFrame,
    weekly: pd.DataFrame,
    fit_months: list[str],
    holdout_month: str,
    top_fracs: list[float],
    min_week_selected_rows: int,
    min_fit_material_weeks: int,
    min_holdout_material_weeks: int,
    min_fit_positive_week_rate: float,
    min_holdout_positive_week_rate: float,
    max_clean_bad_mae_1r_rate: float,
    max_clean_p90_mae_norm: float,
    max_clean_timeout_rate: float,
    max_clean_wide_25bps_rate: float,
    max_bounded_bad_mae_1r_rate: float,
    max_bounded_p90_mae_norm: float,
    min_bounded_mfe_mae_ratio: float,
    max_bounded_timeout_rate: float,
    max_bounded_wide_25bps_rate: float,
) -> pd.DataFrame:
    group_cols = [col for col in GROUP_COLS if col in monthly.columns and col in weekly.columns]
    if "source" not in group_cols:
        raise ValueError("monthly and weekly inputs must include a source column")
    top_frac_values = {float(v) for v in top_fracs}
    monthly_subset = monthly[_num(monthly, "top_frac").isin(top_frac_values)].copy()
    weekly_subset = weekly[_num(weekly, "top_frac").isin(top_frac_values)].copy()
    rows: list[dict[str, Any]] = []
    for key, month_group in monthly_subset.groupby(group_cols, observed=True, dropna=False):
        key_dict = dict(zip(group_cols, key))
        week_group = weekly_subset[_candidate_filter(weekly_subset, key_dict, group_cols)].copy()
        fit_month = month_group[month_group["period"].astype(str).isin(fit_months)].copy()
        holdout_monthly = month_group[month_group["period"].astype(str).eq(str(holdout_month))].copy()
        fit_week = week_group[week_group["period"].astype(str).isin(fit_months)].copy()
        holdout_week = week_group[week_group["period"].astype(str).eq(str(holdout_month))].copy()
        if fit_month.empty or holdout_monthly.empty or fit_week.empty or holdout_week.empty:
            continue
        row: dict[str, Any] = dict(key_dict)
        row.update({"fit_month_list": ",".join(fit_months), "holdout_month": str(holdout_month)})
        row.update(_summarize_month_side("fit", fit_month))
        row.update(_summarize_month_side("holdout", holdout_monthly))
        row.update(_summarize_week_side("fit", fit_week, min_week_selected_rows=min_week_selected_rows))
        row.update(_summarize_week_side("holdout", holdout_week, min_week_selected_rows=min_week_selected_rows))
        _add_decisions(
            row,
            fit_months=fit_months,
            min_fit_material_weeks=min_fit_material_weeks,
            min_holdout_material_weeks=min_holdout_material_weeks,
            min_fit_positive_week_rate=min_fit_positive_week_rate,
            min_holdout_positive_week_rate=min_holdout_positive_week_rate,
            max_clean_bad_mae_1r_rate=max_clean_bad_mae_1r_rate,
            max_clean_p90_mae_norm=max_clean_p90_mae_norm,
            max_clean_timeout_rate=max_clean_timeout_rate,
            max_clean_wide_25bps_rate=max_clean_wide_25bps_rate,
            max_bounded_bad_mae_1r_rate=max_bounded_bad_mae_1r_rate,
            max_bounded_p90_mae_norm=max_bounded_p90_mae_norm,
            min_bounded_mfe_mae_ratio=min_bounded_mfe_mae_ratio,
            max_bounded_timeout_rate=max_bounded_timeout_rate,
            max_bounded_wide_25bps_rate=max_bounded_wide_25bps_rate,
        )
        rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(
        [
            "holdout_clean_pass",
            "holdout_bounded_pass",
            "positive_dirty_holdout",
            "economic_proxy_score",
            "holdout_mean_month_u",
        ],
        ascending=[False, False, False, False, False],
    )


def _read_inputs(input_dirs: list[Path], monthly_filename: str, weekly_filename: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    monthly_frames: list[pd.DataFrame] = []
    weekly_frames: list[pd.DataFrame] = []
    for input_dir in input_dirs:
        monthly_path = input_dir / monthly_filename
        weekly_path = input_dir / weekly_filename
        monthly = pd.read_csv(monthly_path)
        weekly = pd.read_csv(weekly_path)
        monthly["source"] = input_dir.name
        weekly["source"] = input_dir.name
        monthly_frames.append(monthly)
        weekly_frames.append(weekly)
    return pd.concat(monthly_frames, ignore_index=True), pd.concat(weekly_frames, ignore_index=True)


def _format_float_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].map(lambda value: f"{float(value):.4f}" if pd.notna(value) else "")
    return out


def _markdown_table(frame: pd.DataFrame, cols: list[str], limit: int = 30) -> str:
    if frame.empty:
        return "No rows."
    view = frame[[col for col in cols if col in frame.columns]].head(limit).copy()
    return _format_float_columns(view).to_markdown(index=False)


def _write_markdown(output_dir: Path, summary: pd.DataFrame, manifest: dict[str, Any]) -> Path:
    path = output_dir / "regime_proxy_path_profile.md"
    cols = [
        "source",
        "label_arm",
        "regime_feature",
        "regime_bin",
        "bad_mae_penalty",
        "mae_keep_frac",
        "top_frac",
        "economic_proxy_score",
        "fit_mean_month_u",
        "fit_worst_month_u",
        "fit_material_positive_week_rate",
        "fit_bad_mae_1r_rate",
        "fit_p90_mae_norm",
        "fit_mfe_mae_ratio",
        "holdout_mean_month_u",
        "holdout_material_positive_week_rate",
        "holdout_q25_week_u",
        "holdout_bad_mae_1r_rate",
        "holdout_p90_mae_norm",
        "holdout_mfe_mae_ratio",
        "holdout_timeout_rate",
        "holdout_wide_25bps_rate",
        "holdout_clean_reason",
        "holdout_bounded_reason",
    ]
    clean_pass = summary[summary["holdout_clean_pass"].eq(True)] if not summary.empty else summary
    bounded_pass = summary[summary["holdout_bounded_pass"].eq(True)] if not summary.empty else summary
    holdout_economic_fit_failed = (
        summary[
            (
                summary["holdout_clean_standalone_pass"].eq(True)
                | summary["holdout_bounded_standalone_pass"].eq(True)
            )
            & ~summary["holdout_clean_pass"].eq(True)
            & ~summary["holdout_bounded_pass"].eq(True)
        ].sort_values("holdout_mean_month_u", ascending=False)
        if not summary.empty
        else summary
    )
    positive_dirty = (
        summary[summary["positive_dirty_holdout"].eq(True)].sort_values("holdout_mean_month_u", ascending=False)
        if not summary.empty
        else summary
    )
    best_rejected = summary.sort_values("economic_proxy_score", ascending=False) if not summary.empty else summary
    lines = [
        "# Regime Proxy Path Profile",
        "",
        "Scope: existing proxy outputs only; no base/meta model training is run. Fit months are used for candidate selection and June is treated as holdout.",
        "",
        f"Inputs: `{', '.join(manifest['input_dirs'])}`",
        f"Fit months: `{','.join(manifest['fit_months'])}`",
        f"Holdout month: `{manifest['holdout_month']}`",
        f"Top fractions: `{','.join(str(v) for v in manifest['top_fracs'])}`",
        "",
        (
            "Clean economic gate: "
            f"bad-MAE <= `{manifest['gates']['max_clean_bad_mae_1r_rate']}`, "
            f"p90 MAE <= `{manifest['gates']['max_clean_p90_mae_norm']}`R, "
            f"timeout <= `{manifest['gates']['max_clean_timeout_rate']}`, "
            f"wide-25bps <= `{manifest['gates']['max_clean_wide_25bps_rate']}`."
        ),
        (
            "Bounded high-MAE diagnostic gate: "
            f"bad-MAE <= `{manifest['gates']['max_bounded_bad_mae_1r_rate']}`, "
            f"p90 MAE <= `{manifest['gates']['max_bounded_p90_mae_norm']}`R, "
            f"MFE/MAE >= `{manifest['gates']['min_bounded_mfe_mae_ratio']}`, "
            f"timeout <= `{manifest['gates']['max_bounded_timeout_rate']}`, "
            f"wide-25bps <= `{manifest['gates']['max_bounded_wide_25bps_rate']}`."
        ),
        "",
        "## Counts",
        "",
        f"- Candidates evaluated: `{manifest['rows']}`",
        f"- Fit clean pass: `{manifest['fit_clean_pass_rows']}`",
        f"- Holdout clean standalone pass: `{manifest['holdout_clean_standalone_pass_rows']}`",
        f"- Holdout clean pass after fit selection: `{manifest['holdout_clean_pass_rows']}`",
        f"- Fit bounded high-MAE pass: `{manifest['fit_bounded_pass_rows']}`",
        f"- Holdout bounded high-MAE standalone pass: `{manifest['holdout_bounded_standalone_pass_rows']}`",
        f"- Holdout bounded high-MAE pass after fit selection: `{manifest['holdout_bounded_pass_rows']}`",
        f"- Positive but economically dirty holdout: `{manifest['positive_dirty_holdout_rows']}`",
        "",
        "## Clean Economic Holdout Passes",
        "",
        _markdown_table(clean_pass, cols, limit=40),
        "",
        "## Bounded High-MAE Diagnostic Passes",
        "",
        _markdown_table(bounded_pass, cols, limit=40),
        "",
        "## Economically Acceptable In Holdout But Not Fit-Selectable",
        "",
        _markdown_table(holdout_economic_fit_failed, cols, limit=40),
        "",
        "## Positive But Economically Dirty Holdout",
        "",
        _markdown_table(positive_dirty, cols, limit=40),
        "",
        "## Best Rejected By Economic Proxy Score",
        "",
        _markdown_table(best_rejected, cols, limit=40),
        "",
        "## Outputs",
        "",
        f"- Summary: `{manifest['outputs']['summary']}`",
        f"- Manifest: `{manifest['outputs']['manifest']}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_report(
    *,
    input_dirs: list[Path],
    output_dir: Path,
    monthly_filename: str,
    weekly_filename: str,
    fit_months: list[str],
    holdout_month: str,
    top_fracs: list[float],
    min_week_selected_rows: int,
    min_fit_material_weeks: int,
    min_holdout_material_weeks: int,
    min_fit_positive_week_rate: float,
    min_holdout_positive_week_rate: float,
    max_clean_bad_mae_1r_rate: float,
    max_clean_p90_mae_norm: float,
    max_clean_timeout_rate: float,
    max_clean_wide_25bps_rate: float,
    max_bounded_bad_mae_1r_rate: float,
    max_bounded_p90_mae_norm: float,
    min_bounded_mfe_mae_ratio: float,
    max_bounded_timeout_rate: float,
    max_bounded_wide_25bps_rate: float,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    monthly, weekly = _read_inputs(input_dirs, monthly_filename, weekly_filename)
    summary = summarize_candidates(
        monthly=monthly,
        weekly=weekly,
        fit_months=[str(v) for v in fit_months],
        holdout_month=str(holdout_month),
        top_fracs=[float(v) for v in top_fracs],
        min_week_selected_rows=int(min_week_selected_rows),
        min_fit_material_weeks=int(min_fit_material_weeks),
        min_holdout_material_weeks=int(min_holdout_material_weeks),
        min_fit_positive_week_rate=float(min_fit_positive_week_rate),
        min_holdout_positive_week_rate=float(min_holdout_positive_week_rate),
        max_clean_bad_mae_1r_rate=float(max_clean_bad_mae_1r_rate),
        max_clean_p90_mae_norm=float(max_clean_p90_mae_norm),
        max_clean_timeout_rate=float(max_clean_timeout_rate),
        max_clean_wide_25bps_rate=float(max_clean_wide_25bps_rate),
        max_bounded_bad_mae_1r_rate=float(max_bounded_bad_mae_1r_rate),
        max_bounded_p90_mae_norm=float(max_bounded_p90_mae_norm),
        min_bounded_mfe_mae_ratio=float(min_bounded_mfe_mae_ratio),
        max_bounded_timeout_rate=float(max_bounded_timeout_rate),
        max_bounded_wide_25bps_rate=float(max_bounded_wide_25bps_rate),
    )
    paths = {
        "summary": output_dir / "regime_proxy_path_profile_summary.csv",
        "manifest": output_dir / "manifest.json",
    }
    summary.to_csv(paths["summary"], index=False)
    manifest = {
        "input_dirs": [str(path) for path in input_dirs],
        "monthly_filename": str(monthly_filename),
        "weekly_filename": str(weekly_filename),
        "output_dir": str(output_dir),
        "fit_months": [str(v) for v in fit_months],
        "holdout_month": str(holdout_month),
        "top_fracs": [float(v) for v in top_fracs],
        "gates": {
            "min_week_selected_rows": int(min_week_selected_rows),
            "min_fit_material_weeks": int(min_fit_material_weeks),
            "min_holdout_material_weeks": int(min_holdout_material_weeks),
            "min_fit_positive_week_rate": float(min_fit_positive_week_rate),
            "min_holdout_positive_week_rate": float(min_holdout_positive_week_rate),
            "max_clean_bad_mae_1r_rate": float(max_clean_bad_mae_1r_rate),
            "max_clean_p90_mae_norm": float(max_clean_p90_mae_norm),
            "max_clean_timeout_rate": float(max_clean_timeout_rate),
            "max_clean_wide_25bps_rate": float(max_clean_wide_25bps_rate),
            "max_bounded_bad_mae_1r_rate": float(max_bounded_bad_mae_1r_rate),
            "max_bounded_p90_mae_norm": float(max_bounded_p90_mae_norm),
            "min_bounded_mfe_mae_ratio": float(min_bounded_mfe_mae_ratio),
            "max_bounded_timeout_rate": float(max_bounded_timeout_rate),
            "max_bounded_wide_25bps_rate": float(max_bounded_wide_25bps_rate),
        },
        "rows": int(len(summary)),
        "fit_clean_pass_rows": int(summary["fit_clean_pass"].sum()) if not summary.empty else 0,
        "holdout_clean_standalone_pass_rows": int(summary["holdout_clean_standalone_pass"].sum())
        if not summary.empty
        else 0,
        "holdout_clean_pass_rows": int(summary["holdout_clean_pass"].sum()) if not summary.empty else 0,
        "fit_bounded_pass_rows": int(summary["fit_bounded_pass"].sum()) if not summary.empty else 0,
        "holdout_bounded_standalone_pass_rows": int(summary["holdout_bounded_standalone_pass"].sum())
        if not summary.empty
        else 0,
        "holdout_bounded_pass_rows": int(summary["holdout_bounded_pass"].sum()) if not summary.empty else 0,
        "positive_dirty_holdout_rows": int(summary["positive_dirty_holdout"].sum()) if not summary.empty else 0,
        "outputs": {key: str(value) for key, value in paths.items()},
    }
    markdown = _write_markdown(output_dir, summary, manifest)
    manifest["outputs"]["markdown"] = str(markdown)
    paths["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--monthly-filename", default=DEFAULT_MONTHLY_FILENAME)
    parser.add_argument("--weekly-filename", default=DEFAULT_WEEKLY_FILENAME)
    parser.add_argument("--fit-months", default="2026-04,2026-05")
    parser.add_argument("--holdout-month", default="2026-06")
    parser.add_argument("--top-fracs", default="0.005,0.010")
    parser.add_argument("--min-week-selected-rows", type=int, default=3)
    parser.add_argument("--min-fit-material-weeks", type=int, default=4)
    parser.add_argument("--min-holdout-material-weeks", type=int, default=2)
    parser.add_argument("--min-fit-positive-week-rate", type=float, default=0.55)
    parser.add_argument("--min-holdout-positive-week-rate", type=float, default=0.50)
    parser.add_argument("--max-clean-bad-mae-1r-rate", type=float, default=0.50)
    parser.add_argument("--max-clean-p90-mae-norm", type=float, default=3.0)
    parser.add_argument("--max-clean-timeout-rate", type=float, default=0.20)
    parser.add_argument("--max-clean-wide-25bps-rate", type=float, default=0.30)
    parser.add_argument("--max-bounded-bad-mae-1r-rate", type=float, default=0.80)
    parser.add_argument("--max-bounded-p90-mae-norm", type=float, default=4.0)
    parser.add_argument("--min-bounded-mfe-mae-ratio", type=float, default=1.25)
    parser.add_argument("--max-bounded-timeout-rate", type=float, default=0.20)
    parser.add_argument("--max-bounded-wide-25bps-rate", type=float, default=0.35)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dirs = args.input_dir or [DEFAULT_INPUT_DIR]
    manifest = run_report(
        input_dirs=input_dirs,
        output_dir=args.output_dir,
        monthly_filename=str(args.monthly_filename),
        weekly_filename=str(args.weekly_filename),
        fit_months=_parse_csv(args.fit_months),
        holdout_month=str(args.holdout_month),
        top_fracs=_parse_float_csv(args.top_fracs),
        min_week_selected_rows=int(args.min_week_selected_rows),
        min_fit_material_weeks=int(args.min_fit_material_weeks),
        min_holdout_material_weeks=int(args.min_holdout_material_weeks),
        min_fit_positive_week_rate=float(args.min_fit_positive_week_rate),
        min_holdout_positive_week_rate=float(args.min_holdout_positive_week_rate),
        max_clean_bad_mae_1r_rate=float(args.max_clean_bad_mae_1r_rate),
        max_clean_p90_mae_norm=float(args.max_clean_p90_mae_norm),
        max_clean_timeout_rate=float(args.max_clean_timeout_rate),
        max_clean_wide_25bps_rate=float(args.max_clean_wide_25bps_rate),
        max_bounded_bad_mae_1r_rate=float(args.max_bounded_bad_mae_1r_rate),
        max_bounded_p90_mae_norm=float(args.max_bounded_p90_mae_norm),
        min_bounded_mfe_mae_ratio=float(args.min_bounded_mfe_mae_ratio),
        max_bounded_timeout_rate=float(args.max_bounded_timeout_rate),
        max_bounded_wide_25bps_rate=float(args.max_bounded_wide_25bps_rate),
    )
    print(json.dumps(_json_safe(manifest), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
