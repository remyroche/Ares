#!/usr/bin/env python3
"""Locked label-candidate walk-forward comparison before production training.

This report consumes an existing source-conditioned two-head model-smoke grid.
It does not train models. It answers a narrower promotion question:

1. if the selector is locked to prior months only, which candidate is chosen;
2. how that chosen candidate compares with fixed challengers and source-level
   baselines on the next month.
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

from scripts.run_label_first_touch_execution_proxy_ablation import (  # noqa: E402
    DEFAULT_FEATURE_DIR,
    DEFAULT_FEATURE_LIST_CSV,
    _first_touch_metrics,
    _json_safe,
    _load_feature_store_columns,
    _load_labels,
    _path_metrics,
    _read_feature_list,
    _safe_mean,
    _safe_numeric,
    _safe_quantile,
)
from scripts.run_label_source_conditioned_two_head_model_smoke import (  # noqa: E402
    _select_by_fit_policy,
)
from scripts.run_label_source_conditioned_two_head_proxy import _source_fit_holdout_summary  # noqa: E402
from scripts.run_soft_label_candidate_source_ablation import (  # noqa: E402
    _build_sources,
    _source_context,
)
from scripts.run_soft_label_economic_proxy_ablation import (  # noqa: E402
    DEFAULT_EVENT_FEATURE_STORE_FEATURES,
    _event_confirmation_features,
)


DEFAULT_TWO_HEAD_DIR = Path(
    "data_perp/reports/"
    "label_source_conditioned_two_head_lgbm_smoke_quiet_mid_cleanutil_profitfloors_roundb_weights_stage17_v1"
)
DEFAULT_LABELS_PATH = Path(
    "data_perp/artifacts/20260702_120500_first_touch_c0_fast6_s10_policy_net_labels_exitaligned/labels"
)
DEFAULT_OUTPUT_DIR = Path("data_perp/reports/label_candidate_locked_walkforward_stage17_v1")
DEFAULT_SOURCE = "quiet_mid"
DEFAULT_FIXED_WEIGHTS = (
    "W6_decisive_path",
    "W7_timestamp_balanced",
    "W0_base",
    "W2_boundary_top30",
    "W8_combined_conservative",
)


def _optional_float(value: str | None) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "null", "nan"}:
        return None
    return float(text)


def _parse_csv(value: str | None, default: tuple[str, ...]) -> list[str]:
    if value is None or not str(value).strip():
        return list(default)
    return [part.strip() for part in str(value).split(",") if part.strip()]


def _format_table(frame: pd.DataFrame, cols: list[str], limit: int = 80) -> str:
    if frame.empty:
        return "No rows."
    view = frame[[col for col in cols if col in frame.columns]].head(limit).copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda value: f"{float(value):.4f}" if pd.notna(value) else "")
    return view.to_markdown(index=False)


def _finite_float(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _ensure_weight_arm(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "weight_arm" not in out.columns and "arm" in out.columns:
        parts = out["arm"].astype(str).str.split("::")
        out["weight_arm"] = parts.map(lambda item: item[1] if len(item) >= 3 else "none")
    return out


def _selection_view(row: pd.Series, *, label: str, eval_month: str, fit_months: list[str]) -> dict[str, Any]:
    return {
        "label": label,
        "eval_month": str(eval_month),
        "fit_months": ",".join(fit_months),
        "source": row.get("source", ""),
        "weight_arm": row.get("weight_arm", ""),
        "score_rule": row.get("score_rule", ""),
        "bad_threshold": _finite_float(row.get("bad_threshold")),
        "top_k": int(row.get("top_k", 0) or 0),
        "fit_profit_selection_objective": _finite_float(row.get("fit_profit_selection_objective")),
        "fit_mean_month_u": _finite_float(row.get("fit_mean_month_u")),
        "fit_worst_month_u": _finite_float(row.get("fit_worst_month_u")),
        "fit_q25_week_u": _finite_float(row.get("fit_q25_week_u")),
        "fit_selected_rows": int(row.get("fit_selected_rows", 0) or 0),
        "fit_candidate_timestamp_coverage": _finite_float(row.get("fit_candidate_timestamp_coverage")),
        "fit_first_touch_bad_mae_to_sl_rate": _finite_float(row.get("fit_first_touch_bad_mae_to_sl_rate")),
        "fit_p90_first_touch_mae_to_sl": _finite_float(row.get("fit_p90_first_touch_mae_to_sl")),
        "fit_clean_exec_actual_rate": _finite_float(row.get("fit_clean_exec_actual_rate")),
        "holdout_mean_month_u": _finite_float(row.get("holdout_mean_month_u")),
        "holdout_q25_week_u": _finite_float(row.get("holdout_q25_week_u")),
        "holdout_selected_rows": int(row.get("holdout_selected_rows", 0) or 0),
        "holdout_candidate_timestamp_coverage": _finite_float(row.get("holdout_candidate_timestamp_coverage")),
        "holdout_first_touch_bad_mae_to_sl_rate": _finite_float(row.get("holdout_first_touch_bad_mae_to_sl_rate")),
        "holdout_p90_first_touch_mae_to_sl": _finite_float(row.get("holdout_p90_first_touch_mae_to_sl")),
        "holdout_clean_exec_actual_rate": _finite_float(row.get("holdout_clean_exec_actual_rate")),
        "holdout_bounded_pass": bool(row.get("holdout_bounded_pass", False)),
    }


def _candidate_row_for_month(
    monthly: pd.DataFrame,
    *,
    eval_month: str,
    source: str,
    weight_arm: str,
    score_rule: str | None,
    bad_threshold: float | None,
    top_k: int | None,
) -> pd.Series | None:
    subset = monthly[
        monthly["period"].astype(str).eq(str(eval_month))
        & monthly["source"].astype(str).eq(str(source))
        & monthly["weight_arm"].astype(str).eq(str(weight_arm))
    ].copy()
    if score_rule:
        subset = subset[subset["score_rule"].astype(str).eq(str(score_rule))]
    if bad_threshold is not None:
        subset = subset[np.isclose(_safe_numeric(subset["bad_threshold"]), float(bad_threshold), equal_nan=False)]
    if top_k is not None:
        subset = subset[_safe_numeric(subset["top_k"]).eq(int(top_k))]
    if subset.empty:
        return None
    return subset.sort_values(["mean_u", "selected_rows"], ascending=[False, False]).iloc[0]


def _fixed_candidate_rows(
    *,
    monthly: pd.DataFrame,
    weekly: pd.DataFrame,
    months: list[str],
    source: str,
    fixed_weights: list[str],
    min_week_rows: int,
    min_selected_rows_per_fit_month: int,
    fit_min_mean_month_u: float | None,
    fit_min_worst_month_u: float | None,
    fit_min_q25_week_u: float | None,
    fit_min_candidate_timestamp_coverage: float | None,
    fit_min_material_positive_week_rate: float | None,
    fit_min_clean_exec: float | None,
    fit_max_bad_mae: float | None,
    fit_max_p90_mae: float | None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for pos, eval_month in enumerate(months):
        fit_months = months[:pos]
        if not fit_months:
            continue
        fit_holdout = _source_fit_holdout_summary(
            monthly=monthly,
            weekly=weekly,
            fit_months=fit_months,
            holdout_month=eval_month,
            min_week_rows=min_week_rows,
        )
        if fit_holdout.empty:
            continue
        fit_holdout = _ensure_weight_arm(fit_holdout)
        fit_holdout = fit_holdout[
            fit_holdout["source"].astype(str).eq(str(source))
            & fit_holdout["weight_arm"].astype(str).isin(fixed_weights)
        ].copy()
        for weight in fixed_weights:
            candidates = fit_holdout[fit_holdout["weight_arm"].astype(str).eq(str(weight))].copy()
            if candidates.empty:
                continue
            annotated, _, best = _select_by_fit_policy(
                candidates,
                selection_policy="fit_profit_floors",
                require_fit_bounded=True,
                fit_min_mean_month_u=fit_min_mean_month_u,
                fit_min_worst_month_u=fit_min_worst_month_u,
                fit_min_q25_week_u=fit_min_q25_week_u,
                fit_min_selected_rows=int(min_selected_rows_per_fit_month) * len(fit_months),
                fit_min_candidate_timestamp_coverage=fit_min_candidate_timestamp_coverage,
                fit_min_material_positive_week_rate=fit_min_material_positive_week_rate,
                fit_min_clean_exec=fit_min_clean_exec,
                fit_max_bad_mae=fit_max_bad_mae,
                fit_max_p90_mae=fit_max_p90_mae,
            )
            if not best.empty:
                row = best.iloc[0]
                selection_basis = "prior_fit_floors"
            elif not annotated.empty:
                row = annotated.sort_values(
                    ["fit_profit_selection_objective", "fit_mean_month_u", "fit_selected_rows"],
                    ascending=[False, False, False],
                ).iloc[0]
                selection_basis = "prior_fit_best_no_floor"
            else:
                continue
            rows.append(
                {
                    "label": f"prior_fit::{weight}",
                    "eval_month": str(eval_month),
                    "fit_months": ",".join(fit_months),
                    "selection_basis": selection_basis,
                    "source": source,
                    "weight_arm": weight,
                    "score_rule": str(row["score_rule"]),
                    "bad_threshold": float(row["bad_threshold"]),
                    "top_k": int(row["top_k"]),
                    "fit_profit_floor_pass": bool(row.get("fit_profit_floor_pass", False)),
                    "fit_profit_selection_objective": _finite_float(row.get("fit_profit_selection_objective")),
                    "fit_mean_month_u": _finite_float(row.get("fit_mean_month_u")),
                    "fit_worst_month_u": _finite_float(row.get("fit_worst_month_u")),
                    "fit_q25_week_u": _finite_float(row.get("fit_q25_week_u")),
                    "fit_selected_rows": int(row.get("fit_selected_rows", 0) or 0),
                    "selected_rows": int(row.get("holdout_selected_rows", 0) or 0),
                    "mean_u": _finite_float(row.get("holdout_mean_month_u")),
                    "q10_u": np.nan,
                    "hit_u": np.nan,
                    "candidate_timestamp_coverage": _finite_float(row.get("holdout_candidate_timestamp_coverage")),
                    "first_touch_bad_mae_to_sl_rate": _finite_float(row.get("holdout_first_touch_bad_mae_to_sl_rate")),
                    "p90_first_touch_mae_to_sl": _finite_float(row.get("holdout_p90_first_touch_mae_to_sl")),
                    "clean_exec_actual_rate": _finite_float(row.get("holdout_clean_exec_actual_rate")),
                    "first_touch_timeout_rate": np.nan,
                    "holdout_bounded_pass": bool(row.get("holdout_bounded_pass", False)),
                }
            )
    return pd.DataFrame(rows)


def _monthly_source_rows(
    *,
    labels_path: Path,
    feature_dir: Path,
    feature_list_csv: Path,
    max_feature_store_features: int | None,
    source: str,
    run_gap_hours: float,
) -> pd.DataFrame:
    frame = _load_labels(labels_path)
    selected_features = _read_feature_list(feature_list_csv, max_features=max_feature_store_features)
    selected_features = list(dict.fromkeys(list(selected_features) + list(DEFAULT_EVENT_FEATURE_STORE_FEATURES)))
    feature_matrix, _ = _load_feature_store_columns(
        frame,
        feature_dir=feature_dir,
        selected_features=selected_features,
    )
    if not feature_matrix.empty:
        new_cols = [col for col in feature_matrix.columns if col not in frame.columns]
        frame = pd.concat([frame.reset_index(drop=True), feature_matrix.loc[:, new_cols].reset_index(drop=True)], axis=1)
    event_features, _ = _event_confirmation_features(frame, event_features=list(DEFAULT_EVENT_FEATURE_STORE_FEATURES))
    if not event_features.empty:
        new_event_cols = [col for col in event_features.columns if col not in frame.columns]
        frame = pd.concat([frame.reset_index(drop=True), event_features.loc[:, new_event_cols].reset_index(drop=True)], axis=1)
    context = _source_context(frame)
    sources = _build_sources(frame, context, run_gap_hours=run_gap_hours)
    if source not in sources:
        raise ValueError(f"Unknown source: {source}")
    metrics = _first_touch_metrics(frame, _path_metrics(frame))
    mask = sources[source].reindex(frame.index, fill_value=False).astype(bool)
    month_series = frame["__ts__"].dt.to_period("M").astype(str)
    rows: list[dict[str, Any]] = []
    for month in sorted(month_series[mask].dropna().unique()):
        idx = mask & month_series.eq(str(month))
        selected = metrics.loc[idx]
        selected_frame = frame.loc[idx]
        rows.append(
            {
                "label": f"source_all::{source}",
                "eval_month": str(month),
                "source": source,
                "selected_rows": int(idx.sum()),
                "mean_u": _safe_mean(selected["u_policy_net"]),
                "q10_u": _safe_quantile(selected["u_policy_net"], 0.10),
                "hit_u": _safe_mean(selected["u_policy_net"] > 0.0),
                "candidate_timestamp_coverage": 1.0,
                "first_touch_bad_mae_to_sl_rate": _safe_mean(selected["first_touch_mae_to_sl"] >= 1.0),
                "p90_first_touch_mae_to_sl": _safe_quantile(selected["first_touch_mae_to_sl"], 0.90),
                "clean_exec_actual_rate": _safe_mean(selected["clean_exec_actual"]),
                "first_touch_timeout_rate": _safe_mean(selected["first_touch_timeout"]),
                "top_symbol_share": float(selected_frame["__symbol__"].value_counts(normalize=True).iloc[0])
                if len(selected_frame)
                else 0.0,
            }
        )
    return pd.DataFrame(rows)


def _rolling_selector_rows(
    *,
    monthly: pd.DataFrame,
    weekly: pd.DataFrame,
    source: str,
    months: list[str],
    min_week_rows: int,
    min_selected_rows_per_fit_month: int,
    fit_min_mean_month_u: float | None,
    fit_min_worst_month_u: float | None,
    fit_min_q25_week_u: float | None,
    fit_min_candidate_timestamp_coverage: float | None,
    fit_min_material_positive_week_rate: float | None,
    fit_min_clean_exec: float | None,
    fit_max_bad_mae: float | None,
    fit_max_p90_mae: float | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    annotated_parts: list[pd.DataFrame] = []
    selected_rows: list[dict[str, Any]] = []
    for pos, eval_month in enumerate(months):
        fit_months = months[:pos]
        if not fit_months:
            continue
        fit_holdout = _source_fit_holdout_summary(
            monthly=monthly,
            weekly=weekly,
            fit_months=fit_months,
            holdout_month=eval_month,
            min_week_rows=min_week_rows,
        )
        if fit_holdout.empty:
            continue
        fit_holdout = _ensure_weight_arm(fit_holdout)
        fit_holdout = fit_holdout[fit_holdout["source"].astype(str).eq(str(source))].copy()
        if fit_holdout.empty:
            continue
        annotated, _, best = _select_by_fit_policy(
            fit_holdout,
            selection_policy="fit_profit_floors",
            require_fit_bounded=True,
            fit_min_mean_month_u=fit_min_mean_month_u,
            fit_min_worst_month_u=fit_min_worst_month_u,
            fit_min_q25_week_u=fit_min_q25_week_u,
            fit_min_selected_rows=int(min_selected_rows_per_fit_month) * len(fit_months),
            fit_min_candidate_timestamp_coverage=fit_min_candidate_timestamp_coverage,
            fit_min_material_positive_week_rate=fit_min_material_positive_week_rate,
            fit_min_clean_exec=fit_min_clean_exec,
            fit_max_bad_mae=fit_max_bad_mae,
            fit_max_p90_mae=fit_max_p90_mae,
        )
        annotated.insert(0, "eval_month", str(eval_month))
        annotated.insert(1, "fit_months_used", ",".join(fit_months))
        annotated_parts.append(annotated)
        if not best.empty:
            selected_rows.append(_selection_view(best.iloc[0], label="rolling_prior_month_selector", eval_month=eval_month, fit_months=fit_months))
    return (
        pd.concat(annotated_parts, ignore_index=True) if annotated_parts else pd.DataFrame(),
        pd.DataFrame(selected_rows),
    )


def _combine_eval(
    *,
    rolling: pd.DataFrame,
    fixed: pd.DataFrame,
    source_baseline: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in rolling.iterrows():
        rows.append(
            {
                "label": row["label"],
                "eval_month": row["eval_month"],
                "fit_months": row.get("fit_months", ""),
                "selection_basis": "prior_fit_floors",
                "source": row.get("source", ""),
                "weight_arm": row.get("weight_arm", ""),
                "score_rule": row.get("score_rule", ""),
                "bad_threshold": row.get("bad_threshold", np.nan),
                "top_k": row.get("top_k", np.nan),
                "fit_profit_floor_pass": True,
                "fit_profit_selection_objective": row.get("fit_profit_selection_objective", np.nan),
                "fit_mean_month_u": row.get("fit_mean_month_u", np.nan),
                "fit_worst_month_u": row.get("fit_worst_month_u", np.nan),
                "fit_q25_week_u": row.get("fit_q25_week_u", np.nan),
                "selected_rows": row.get("holdout_selected_rows", 0),
                "mean_u": row.get("holdout_mean_month_u", np.nan),
                "q10_u": np.nan,
                "hit_u": np.nan,
                "candidate_timestamp_coverage": row.get("holdout_candidate_timestamp_coverage", np.nan),
                "first_touch_bad_mae_to_sl_rate": row.get("holdout_first_touch_bad_mae_to_sl_rate", np.nan),
                "p90_first_touch_mae_to_sl": row.get("holdout_p90_first_touch_mae_to_sl", np.nan),
                "clean_exec_actual_rate": row.get("holdout_clean_exec_actual_rate", np.nan),
                "holdout_bounded_pass": row.get("holdout_bounded_pass", False),
            }
        )
    rows.extend(fixed.to_dict("records") if not fixed.empty else [])
    rows.extend(source_baseline.to_dict("records") if not source_baseline.empty else [])
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    source_mean = (
        out[out["label"].astype(str).str.startswith("source_all::")]
        .set_index("eval_month")["mean_u"]
        .to_dict()
    )
    out["delta_mean_u_vs_source_all"] = out.apply(
        lambda row: _finite_float(row.get("mean_u")) - _finite_float(source_mean.get(str(row.get("eval_month")))),
        axis=1,
    )
    return out.sort_values(["eval_month", "label"]).reset_index(drop=True)


def _write_markdown(
    *,
    output_dir: Path,
    manifest: dict[str, Any],
    rolling_selected: pd.DataFrame,
    eval_comparison: pd.DataFrame,
    rolling_grid: pd.DataFrame,
) -> Path:
    path = output_dir / "label_candidate_locked_walkforward.md"
    eval_cols = [
        "label",
        "eval_month",
        "fit_months",
        "selection_basis",
        "weight_arm",
        "score_rule",
        "bad_threshold",
        "top_k",
        "fit_profit_floor_pass",
        "fit_mean_month_u",
        "fit_q25_week_u",
        "selected_rows",
        "mean_u",
        "delta_mean_u_vs_source_all",
        "candidate_timestamp_coverage",
        "first_touch_bad_mae_to_sl_rate",
        "p90_first_touch_mae_to_sl",
        "clean_exec_actual_rate",
        "holdout_bounded_pass",
    ]
    selected_cols = [
        "eval_month",
        "fit_months",
        "weight_arm",
        "score_rule",
        "bad_threshold",
        "top_k",
        "fit_profit_selection_objective",
        "fit_mean_month_u",
        "fit_worst_month_u",
        "fit_q25_week_u",
        "holdout_mean_month_u",
        "holdout_q25_week_u",
        "holdout_selected_rows",
        "holdout_bounded_pass",
    ]
    grid_cols = [
        "eval_month",
        "fit_months_used",
        "weight_arm",
        "score_rule",
        "bad_threshold",
        "top_k",
        "fit_profit_floor_pass",
        "fit_profit_selection_objective",
        "fit_mean_month_u",
        "fit_q25_week_u",
        "holdout_mean_month_u",
        "holdout_bounded_pass",
    ]
    lines = [
        "# Label Candidate Locked Walk-Forward",
        "",
        "Scope: report-only comparison using existing two-head smoke predictions. No model training, Optuna, or policy geometry optimisation is run.",
        "",
        f"Two-head smoke: `{manifest['two_head_dir']}`",
        f"Labels: `{manifest['labels_path']}`",
        f"Source: `{manifest['source']}`",
        f"Months: `{', '.join(manifest['months'])}`",
        "",
        "For each eval month after the first available month, the selector is fit only on prior months. Per-weight challengers also choose their score rule, bad threshold, and top-k from prior months only.",
        "",
        "## Rolling Prior-Month Selection",
        "",
        _format_table(rolling_selected, selected_cols, limit=80),
        "",
        "## Eval Comparison",
        "",
        _format_table(eval_comparison, eval_cols, limit=160),
        "",
        "## Rolling Grid Sample",
        "",
        _format_table(
            rolling_grid.sort_values(["eval_month", "fit_profit_selection_objective"], ascending=[True, False])
            if not rolling_grid.empty
            else rolling_grid,
            grid_cols,
            limit=120,
        ),
        "",
        "## Outputs",
        "",
        f"- Rolling selected: `{manifest['outputs']['rolling_selected']}`",
        f"- Rolling grid: `{manifest['outputs']['rolling_grid']}`",
        f"- Eval comparison: `{manifest['outputs']['eval_comparison']}`",
        f"- Source baseline monthly: `{manifest['outputs']['source_baseline_monthly']}`",
        f"- Manifest: `{manifest['outputs']['manifest']}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_report(
    *,
    two_head_dir: Path,
    labels_path: Path,
    output_dir: Path,
    feature_dir: Path,
    feature_list_csv: Path,
    max_feature_store_features: int | None,
    source: str,
    fixed_weights: list[str],
    min_week_rows: int,
    min_selected_rows_per_fit_month: int,
    run_gap_hours: float,
    fit_min_mean_month_u: float | None,
    fit_min_worst_month_u: float | None,
    fit_min_q25_week_u: float | None,
    fit_min_candidate_timestamp_coverage: float | None,
    fit_min_material_positive_week_rate: float | None,
    fit_min_clean_exec: float | None,
    fit_max_bad_mae: float | None,
    fit_max_p90_mae: float | None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    monthly = _ensure_weight_arm(pd.read_csv(two_head_dir / "source_conditioned_two_head_model_monthly.csv"))
    weekly = _ensure_weight_arm(pd.read_csv(two_head_dir / "source_conditioned_two_head_model_weekly.csv"))
    months = sorted(monthly["period"].astype(str).dropna().unique())
    rolling_grid, rolling_selected = _rolling_selector_rows(
        monthly=monthly,
        weekly=weekly,
        source=source,
        months=months,
        min_week_rows=min_week_rows,
        min_selected_rows_per_fit_month=min_selected_rows_per_fit_month,
        fit_min_mean_month_u=fit_min_mean_month_u,
        fit_min_worst_month_u=fit_min_worst_month_u,
        fit_min_q25_week_u=fit_min_q25_week_u,
        fit_min_candidate_timestamp_coverage=fit_min_candidate_timestamp_coverage,
        fit_min_material_positive_week_rate=fit_min_material_positive_week_rate,
        fit_min_clean_exec=fit_min_clean_exec,
        fit_max_bad_mae=fit_max_bad_mae,
        fit_max_p90_mae=fit_max_p90_mae,
    )
    eval_months = sorted(set(rolling_selected["eval_month"].astype(str))) if not rolling_selected.empty else months[1:]
    fixed = _fixed_candidate_rows(
        monthly=monthly,
        weekly=weekly,
        months=months,
        source=source,
        fixed_weights=fixed_weights,
        min_week_rows=min_week_rows,
        min_selected_rows_per_fit_month=min_selected_rows_per_fit_month,
        fit_min_mean_month_u=fit_min_mean_month_u,
        fit_min_worst_month_u=fit_min_worst_month_u,
        fit_min_q25_week_u=fit_min_q25_week_u,
        fit_min_candidate_timestamp_coverage=fit_min_candidate_timestamp_coverage,
        fit_min_material_positive_week_rate=fit_min_material_positive_week_rate,
        fit_min_clean_exec=fit_min_clean_exec,
        fit_max_bad_mae=fit_max_bad_mae,
        fit_max_p90_mae=fit_max_p90_mae,
    )
    source_baseline = _monthly_source_rows(
        labels_path=labels_path,
        feature_dir=feature_dir,
        feature_list_csv=feature_list_csv,
        max_feature_store_features=max_feature_store_features,
        source=source,
        run_gap_hours=run_gap_hours,
    )
    source_baseline_eval = source_baseline[source_baseline["eval_month"].astype(str).isin(eval_months)].copy()
    eval_comparison = _combine_eval(
        rolling=rolling_selected,
        fixed=fixed,
        source_baseline=source_baseline_eval,
    )

    paths = {
        "rolling_selected": output_dir / "rolling_selected.csv",
        "rolling_grid": output_dir / "rolling_grid.csv",
        "eval_comparison": output_dir / "eval_comparison.csv",
        "source_baseline_monthly": output_dir / "source_baseline_monthly.csv",
        "manifest": output_dir / "manifest.json",
    }
    rolling_selected.to_csv(paths["rolling_selected"], index=False)
    rolling_grid.to_csv(paths["rolling_grid"], index=False)
    eval_comparison.to_csv(paths["eval_comparison"], index=False)
    source_baseline.to_csv(paths["source_baseline_monthly"], index=False)

    manifest = {
        "two_head_dir": str(two_head_dir),
        "labels_path": str(labels_path),
        "output_dir": str(output_dir),
        "source": source,
        "months": months,
        "eval_months": eval_months,
        "fixed_weights": fixed_weights,
        "min_week_rows": int(min_week_rows),
        "min_selected_rows_per_fit_month": int(min_selected_rows_per_fit_month),
        "fit_floor_config": {
            "fit_min_mean_month_u": fit_min_mean_month_u,
            "fit_min_worst_month_u": fit_min_worst_month_u,
            "fit_min_q25_week_u": fit_min_q25_week_u,
            "fit_min_candidate_timestamp_coverage": fit_min_candidate_timestamp_coverage,
            "fit_min_material_positive_week_rate": fit_min_material_positive_week_rate,
            "fit_min_clean_exec": fit_min_clean_exec,
            "fit_max_bad_mae": fit_max_bad_mae,
            "fit_max_p90_mae": fit_max_p90_mae,
        },
        "outputs": {key: str(value) for key, value in paths.items()},
    }
    markdown = _write_markdown(
        output_dir=output_dir,
        manifest=manifest,
        rolling_selected=rolling_selected,
        eval_comparison=eval_comparison,
        rolling_grid=rolling_grid,
    )
    manifest["outputs"]["markdown"] = str(markdown)
    paths["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--two-head-dir", type=Path, default=DEFAULT_TWO_HEAD_DIR)
    parser.add_argument("--labels-path", type=Path, default=DEFAULT_LABELS_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--feature-list-csv", type=Path, default=DEFAULT_FEATURE_LIST_CSV)
    parser.add_argument("--max-feature-store-features", type=int, default=None)
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--fixed-weights", default=",".join(DEFAULT_FIXED_WEIGHTS))
    parser.add_argument("--min-week-rows", type=int, default=3)
    parser.add_argument("--min-selected-rows-per-fit-month", type=int, default=75)
    parser.add_argument("--run-gap-hours", type=float, default=24.0)
    parser.add_argument("--fit-min-mean-month-u", default="0.0015")
    parser.add_argument("--fit-min-worst-month-u", default="0.0")
    parser.add_argument("--fit-min-q25-week-u", default="0.0")
    parser.add_argument("--fit-min-candidate-timestamp-coverage", default="0.08")
    parser.add_argument("--fit-min-material-positive-week-rate", default="0.70")
    parser.add_argument("--fit-min-clean-exec", default="0.65")
    parser.add_argument("--fit-max-bad-mae", default="0.22")
    parser.add_argument("--fit-max-p90-mae", default="1.60")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = run_report(
        two_head_dir=args.two_head_dir,
        labels_path=args.labels_path,
        output_dir=args.output_dir,
        feature_dir=args.feature_dir,
        feature_list_csv=args.feature_list_csv,
        max_feature_store_features=args.max_feature_store_features,
        source=str(args.source),
        fixed_weights=_parse_csv(args.fixed_weights, DEFAULT_FIXED_WEIGHTS),
        min_week_rows=int(args.min_week_rows),
        min_selected_rows_per_fit_month=int(args.min_selected_rows_per_fit_month),
        run_gap_hours=float(args.run_gap_hours),
        fit_min_mean_month_u=_optional_float(args.fit_min_mean_month_u),
        fit_min_worst_month_u=_optional_float(args.fit_min_worst_month_u),
        fit_min_q25_week_u=_optional_float(args.fit_min_q25_week_u),
        fit_min_candidate_timestamp_coverage=_optional_float(args.fit_min_candidate_timestamp_coverage),
        fit_min_material_positive_week_rate=_optional_float(args.fit_min_material_positive_week_rate),
        fit_min_clean_exec=_optional_float(args.fit_min_clean_exec),
        fit_max_bad_mae=_optional_float(args.fit_max_bad_mae),
        fit_max_p90_mae=_optional_float(args.fit_max_p90_mae),
    )
    summary = {
        "output_dir": manifest["output_dir"],
        "source": manifest["source"],
        "months": manifest["months"],
        "eval_months": manifest["eval_months"],
        "fixed_weights": manifest["fixed_weights"],
        "outputs": manifest["outputs"],
    }
    print(json.dumps(_json_safe(summary), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
