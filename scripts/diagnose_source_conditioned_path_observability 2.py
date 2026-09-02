#!/usr/bin/env python3
"""Source-conditioned label/path proxy observability before model training.

This is a proxy-only label QA tool. It rebuilds decision-time source masks from
causal features, learns simple prior-window feature proxies inside each source,
and checks whether selected rows stay profitable inside execution/path limits.
No LightGBM, Optuna, policy optimisation, or final model training is run.
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

from scripts.diagnose_label_matched_clean_dirty_feature_gap import (  # noqa: E402
    DEFAULT_LABELS_PATH,
    _build_frame,
)
from scripts.report_label_proxy_quality_with_economic_limits import (  # noqa: E402
    DEFAULT_FEATURE_LIST_CSV,
    DEFAULT_PROXY_TOP_K,
    _score_proxy,
    _score_period,
    _slice_week_positions,
    _state_path_feature_columns,
    _state_path_risk_targets,
    _table,
    _top_gate,
)
from scripts.run_label_economic_proxy_ablation import _label_targets  # noqa: E402
from scripts.run_label_quality_proxy_diagnostics import (  # noqa: E402
    DEFAULT_FEATURE_DIR,
    _feature_columns,
    _json_safe,
    _rank_top_indices,
    _safe_mean,
    _safe_quantile,
    _spearman,
)
from scripts.run_soft_label_candidate_source_ablation import (  # noqa: E402
    _build_sources,
    _causal_time_edge_prior_features,
    _source_context,
    _source_summary,
)
from scripts.run_soft_label_economic_proxy_ablation import (  # noqa: E402
    DEFAULT_EVENT_FEATURE_STORE_FEATURES,
    DEFAULT_PRIOR_WINDOWS_DAYS,
    DEFAULT_STATE_PATH_PRIOR_FEATURES,
)
from scripts.run_stable_path_proxy_economic_ablation import _stable_proxy_score  # noqa: E402


DEFAULT_OUTPUT_DIR = Path("data_perp/reports/source_conditioned_path_observability_stage95_v1")
DEFAULT_MONTHS = ("2026-04", "2026-05", "2026-06")
DEFAULT_FIT_MONTHS = ("2026-04", "2026-05")
DEFAULT_HOLDOUT_MONTH = "2026-06"
DEFAULT_LABEL_ARMS = (
    "S61_tpnet_strict_adverse_veto_rank",
    "S62_tpnet_clean_dirty_contrast_rank",
    "S65_profit_inside_exec_admissible",
)
DEFAULT_PATH_TARGETS = (
    "non_timeout",
    "profit_low_mae_no_timeout",
    "decisive_profit_low_mae",
    "bounded",
    "path_quality",
)
DEFAULT_SOURCES = (
    "all",
    "quiet_mid",
    "quiet_quality",
    "confirmed_lowbarrier_quality",
    "confirmed_impulse_lowbarrier",
    "clean_breakout_quality",
    "confirmed_lowbarrier_family",
    "confirmed_lowbarrier_impulse_family",
    "confirmed_event_quality_family",
    "time_edge_prior_confirmed_family",
    "dual_prior_confirmed_lowbarrier_quality",
    "dual_prior_confirmed_impulse_lowbarrier",
    "strict_dual_prior_confirmed_lowbarrier_quality",
    "strict_dual_prior_confirmed_impulse_lowbarrier",
    "run_entry_confirmed_event_quality_family",
    "run_entry_time_edge_prior_confirmed_family",
)
DEFAULT_TOP_FRACS = (0.03, 0.05)
DEFAULT_GATE_FRACS = (0.10, 0.20)
DEFAULT_PROXY_MODE = "fit_ic"


def _parse_csv(value: str | list[str] | tuple[str, ...], default: tuple[str, ...] = ()) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(part).strip() for part in value if str(part).strip()]
    text = str(value).strip()
    if not text:
        return list(default)
    return [part.strip() for part in text.split(",") if part.strip()]


def _parse_float_csv(value: str | list[float] | tuple[float, ...]) -> list[float]:
    if isinstance(value, (list, tuple)):
        return [float(part) for part in value]
    return [float(part.strip()) for part in str(value).split(",") if part.strip()]


def _safe_numeric(values: Any) -> pd.Series:
    if isinstance(values, pd.Series):
        return pd.to_numeric(values, errors="coerce")
    return pd.to_numeric(pd.Series(values), errors="coerce")


def _weighted_mean(frame: pd.DataFrame, value_col: str, weight_col: str) -> float:
    if frame.empty or value_col not in frame.columns or weight_col not in frame.columns:
        return float("nan")
    values = pd.to_numeric(frame[value_col], errors="coerce")
    weights = pd.to_numeric(frame[weight_col], errors="coerce").fillna(0.0)
    mask = values.notna() & weights.gt(0.0)
    if not bool(mask.any()):
        return float("nan")
    return float(np.average(values[mask], weights=weights[mask]))


def _path_rate_columns(path_targets: dict[str, pd.Series]) -> list[str]:
    return [f"selected_path_{name}_rate" for name in path_targets]


def _add_selected_path_rates(
    row: dict[str, Any],
    *,
    score: pd.Series,
    path_targets: dict[str, pd.Series],
    top_frac: float,
) -> None:
    idx = _rank_top_indices(pd.to_numeric(score.reset_index(drop=True), errors="coerce"), float(top_frac))
    for name, target in path_targets.items():
        values = pd.to_numeric(target.reset_index(drop=True), errors="coerce")
        selected = values.iloc[idx] if len(idx) else values.iloc[:0]
        row[f"selected_path_{name}_rate"] = _safe_mean(selected)


def _proxy_score_for_mode(
    *,
    proxy_mode: str,
    train: pd.DataFrame,
    valid: pd.DataFrame,
    features: list[str],
    y_train: pd.Series,
    binary_target: bool,
    proxy_top_k: int,
    min_inner_folds: int,
    min_inner_train_rows: int,
    min_inner_valid_rows: int,
    min_consistency: float,
) -> tuple[pd.Series, dict[str, Any]]:
    mode = str(proxy_mode).strip().lower()
    if mode == "stable":
        return _stable_proxy_score(
            train=train,
            valid=valid,
            features=features,
            y_train=y_train,
            binary_target=binary_target,
            proxy_top_k=proxy_top_k,
            min_inner_folds=min_inner_folds,
            min_inner_train_rows=min_inner_train_rows,
            min_inner_valid_rows=min_inner_valid_rows,
            min_consistency=min_consistency,
        )
    if mode != "fit_ic":
        raise ValueError(f"Unknown proxy mode: {proxy_mode}")
    score, diag = _score_proxy(
        train=train,
        valid=valid,
        features=features,
        y_train=y_train,
        proxy_top_k=proxy_top_k,
    )
    return score, {
        "selection_mode": "fit_ic",
        "proxy_features": diag.get("proxy_features", []),
        "proxy_feature_directions": "",
        "proxy_top_abs_ic": diag.get("proxy_top_abs_ic"),
        "proxy_mean_top_abs_ic": diag.get("proxy_mean_top_abs_ic"),
    }


def _source_month_prevalence(
    *,
    frame: pd.DataFrame,
    metrics: pd.DataFrame,
    path_targets: dict[str, pd.Series],
    sources: dict[str, pd.Series],
    months: list[str],
) -> pd.DataFrame:
    month_series = frame["__ts__"].dt.to_period("M").astype(str)
    rows: list[dict[str, Any]] = []
    for source, source_mask in sources.items():
        for month in months:
            mask = source_mask & month_series.eq(str(month))
            selected_metrics = metrics.loc[mask]
            if selected_metrics.empty:
                row: dict[str, Any] = {
                    "source": source,
                    "month": month,
                    "rows": 0,
                    "symbols": 0,
                    "mean_return_net": float("nan"),
                    "mean_u": float("nan"),
                    "hit_u": float("nan"),
                    "bad_mae_1r_rate": float("nan"),
                    "p90_mae_norm": float("nan"),
                    "wide_barrier_25bps_rate": float("nan"),
                    "timeout_rate": float("nan"),
                }
            else:
                row = {
                    "source": source,
                    "month": month,
                    "rows": int(mask.sum()),
                    "symbols": int(frame.loc[mask, "__symbol__"].nunique(dropna=True)),
                    "mean_return_net": _safe_mean(selected_metrics["ret_net"]),
                    "mean_u": _safe_mean(selected_metrics["u_policy_net"]),
                    "hit_u": _safe_mean(selected_metrics["u_policy_net"] > 0.0),
                    "bad_mae_1r_rate": _safe_mean(selected_metrics["mae_norm"] >= 1.0),
                    "p90_mae_norm": _safe_quantile(selected_metrics["mae_norm"], 0.90),
                    "wide_barrier_25bps_rate": _safe_mean(selected_metrics["barrier"] > 0.025),
                    "timeout_rate": _safe_mean(selected_metrics["is_timeout"].astype(float)),
                }
            for target_name, target in path_targets.items():
                row[f"path_{target_name}_rate"] = _safe_mean(target.loc[mask]) if int(mask.sum()) else float("nan")
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["source", "month"], ascending=[True, True])


def _period_rows_for_selector(
    *,
    valid: pd.DataFrame,
    valid_metrics: pd.DataFrame,
    target: pd.DataFrame,
    score: pd.Series,
    month: str,
    source: str,
    selector: str,
    label_arm: str,
    economic_arm: str,
    top_fracs: list[float],
    label_score: pd.Series | None,
    economic_score: pd.Series | None,
    economic_target: pd.Series | None,
    label_proxy_features: str,
    economic_proxy_features: str,
    selected_path_targets: dict[str, pd.Series],
    train_rows: int,
    valid_rows: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    valid = valid.reset_index(drop=True)
    valid_metrics = valid_metrics.reset_index(drop=True)
    target = target.reset_index(drop=True)
    score = pd.to_numeric(score.reset_index(drop=True), errors="coerce")
    label_score = pd.to_numeric(label_score.reset_index(drop=True), errors="coerce") if label_score is not None else None
    economic_score = (
        pd.to_numeric(economic_score.reset_index(drop=True), errors="coerce")
        if economic_score is not None
        else None
    )
    economic_target = (
        pd.to_numeric(economic_target.reset_index(drop=True), errors="coerce")
        if economic_target is not None
        else None
    )
    period_slices = [("month", month, np.arange(len(valid), dtype=np.int64))]
    period_slices.extend(("week", week, pos) for week, pos in _slice_week_positions(valid))
    for period_type, period, pos in period_slices:
        local_targets = {name: values.iloc[pos].reset_index(drop=True) for name, values in selected_path_targets.items()}
        for top_frac in top_fracs:
            row = _score_period(
                frame=valid.iloc[pos].reset_index(drop=True),
                metrics=valid_metrics.iloc[pos].reset_index(drop=True),
                target=target.iloc[pos].reset_index(drop=True),
                score=score.iloc[pos].reset_index(drop=True),
                period_type=period_type,
                period=period,
                month=month,
                selector=selector,
                label_arm=label_arm,
                economic_arm=economic_arm,
                top_frac=float(top_frac),
                label_score=label_score.iloc[pos].reset_index(drop=True) if label_score is not None else None,
                economic_score=economic_score.iloc[pos].reset_index(drop=True) if economic_score is not None else None,
                economic_target=economic_target.iloc[pos].reset_index(drop=True) if economic_target is not None else None,
                label_proxy_features=label_proxy_features,
                economic_proxy_features=economic_proxy_features,
            )
            row["source"] = source
            row["source_train_rows"] = int(train_rows)
            row["source_valid_rows"] = int(valid_rows)
            _add_selected_path_rates(
                row,
                score=score.iloc[pos].reset_index(drop=True),
                path_targets=local_targets,
                top_frac=float(top_frac),
            )
            rows.append(row)
    return rows


def _fit_holdout_summary(
    period_rows: pd.DataFrame,
    *,
    fit_months: list[str],
    holdout_month: str,
    min_week_rows: int,
    max_timeout_rate: float | None,
) -> pd.DataFrame:
    if period_rows.empty:
        return pd.DataFrame()
    monthly = period_rows[period_rows["period_type"].eq("month")].copy()
    weekly = period_rows[period_rows["period_type"].eq("week")].copy()
    if monthly.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    group_cols = ["source", "selector", "label_arm", "economic_arm", "top_frac"]
    for key, group in monthly.groupby(group_cols, observed=True, dropna=False):
        source, selector, label_arm, economic_arm, top_frac = key
        week_group = weekly[
            weekly["source"].astype(str).eq(str(source))
            & weekly["selector"].astype(str).eq(str(selector))
            & weekly["label_arm"].astype(str).eq(str(label_arm))
            & weekly["economic_arm"].astype(str).eq(str(economic_arm))
            & pd.to_numeric(weekly["top_frac"], errors="coerce").eq(float(top_frac))
        ].copy()
        fit_month = group[group["month"].astype(str).isin(fit_months)].copy()
        holdout_monthly = group[group["month"].astype(str).eq(str(holdout_month))].copy()
        fit_week = week_group[week_group["month"].astype(str).isin(fit_months)].copy()
        holdout_week = week_group[week_group["month"].astype(str).eq(str(holdout_month))].copy()
        if fit_month.empty or holdout_monthly.empty:
            continue

        def week_stats(frame: pd.DataFrame) -> tuple[int, float, float]:
            returns = pd.to_numeric(frame["mean_return_net"], errors="coerce")
            selected = pd.to_numeric(frame["selected_rows"], errors="coerce").fillna(0.0)
            material = selected.ge(int(min_week_rows))
            if not bool(material.any()):
                return 0, float("nan"), float("nan")
            return (
                int(material.sum()),
                float((returns.gt(0.0) & material).sum() / material.sum()),
                _safe_quantile(returns[material], 0.25),
            )

        fit_returns = pd.to_numeric(fit_month["mean_return_net"], errors="coerce")
        holdout_returns = pd.to_numeric(holdout_monthly["mean_return_net"], errors="coerce")
        fit_material_weeks, fit_week_rate, fit_q25_week = week_stats(fit_week)
        holdout_material_weeks, holdout_week_rate, holdout_q25_week = week_stats(holdout_week)
        row: dict[str, Any] = {
            "source": str(source),
            "selector": str(selector),
            "label_arm": str(label_arm),
            "economic_arm": str(economic_arm),
            "top_frac": float(top_frac),
            "fit_mean_return_net": _safe_mean(fit_returns),
            "fit_worst_return_net": _safe_quantile(fit_returns, 0.0),
            "holdout_mean_return_net": _safe_mean(holdout_returns),
            "holdout_worst_return_net": _safe_quantile(holdout_returns, 0.0),
            "fit_positive_months": int(fit_returns.gt(0.0).sum()),
            "holdout_positive_months": int(holdout_returns.gt(0.0).sum()),
            "fit_material_weeks": fit_material_weeks,
            "holdout_material_weeks": holdout_material_weeks,
            "fit_material_positive_week_rate": fit_week_rate,
            "holdout_material_positive_week_rate": holdout_week_rate,
            "fit_q25_week_return_net": fit_q25_week,
            "holdout_q25_week_return_net": holdout_q25_week,
            "fit_bad_mae_1r_rate": _weighted_mean(fit_month, "bad_mae_1r_rate", "selected_rows"),
            "holdout_bad_mae_1r_rate": _weighted_mean(holdout_monthly, "bad_mae_1r_rate", "selected_rows"),
            "fit_p90_mae_norm": _weighted_mean(fit_month, "p90_mae_norm", "selected_rows"),
            "holdout_p90_mae_norm": _weighted_mean(holdout_monthly, "p90_mae_norm", "selected_rows"),
            "fit_wide_barrier_25bps_rate": _weighted_mean(fit_month, "wide_barrier_25bps_rate", "selected_rows"),
            "holdout_wide_barrier_25bps_rate": _weighted_mean(
                holdout_monthly,
                "wide_barrier_25bps_rate",
                "selected_rows",
            ),
            "fit_timeout_rate": _weighted_mean(fit_month, "timeout_rate", "selected_rows"),
            "holdout_timeout_rate": _weighted_mean(holdout_monthly, "timeout_rate", "selected_rows"),
            "fit_score_ic_u": _safe_mean(fit_month["score_ic_u"]),
            "holdout_score_ic_u": _safe_mean(holdout_monthly["score_ic_u"]),
            "fit_selected_rows": int(pd.to_numeric(fit_month["selected_rows"], errors="coerce").sum(skipna=True)),
            "holdout_selected_rows": int(
                pd.to_numeric(holdout_monthly["selected_rows"], errors="coerce").sum(skipna=True)
            ),
        }
        for col in _path_rate_columns({c.replace("selected_path_", "").replace("_rate", ""): None for c in []}):
            row[col] = float("nan")
        for col in [c for c in period_rows.columns if c.startswith("selected_path_") and c.endswith("_rate")]:
            row[f"fit_{col}"] = _weighted_mean(fit_month, col, "selected_rows")
            row[f"holdout_{col}"] = _weighted_mean(holdout_monthly, col, "selected_rows")

        needs_positive_ic = not str(selector).startswith("oracle_")
        fit_sign = (
            row["fit_positive_months"] == len(fit_months)
            and row["fit_worst_return_net"] > 0.0
            and row["fit_material_weeks"] >= 4
            and row["fit_material_positive_week_rate"] >= 0.55
        )
        holdout_sign = (
            row["holdout_positive_months"] >= 1
            and row["holdout_mean_return_net"] > 0.0
            and row["holdout_material_weeks"] >= 2
            and row["holdout_material_positive_week_rate"] >= 0.50
        )
        fit_timeout_ok = (
            True
            if max_timeout_rate is None
            else math.isfinite(float(row["fit_timeout_rate"]))
            and float(row["fit_timeout_rate"]) <= float(max_timeout_rate)
        )
        holdout_timeout_ok = (
            True
            if max_timeout_rate is None
            else math.isfinite(float(row["holdout_timeout_rate"]))
            and float(row["holdout_timeout_rate"]) <= float(max_timeout_rate)
        )
        fit_economic = (
            fit_sign
            and (row["fit_score_ic_u"] > 0.0 if needs_positive_ic else True)
            and row["fit_bad_mae_1r_rate"] <= 0.40
            and row["fit_p90_mae_norm"] <= 4.00
            and row["fit_wide_barrier_25bps_rate"] <= 0.05
            and fit_timeout_ok
        )
        holdout_economic = (
            holdout_sign
            and (row["holdout_score_ic_u"] > 0.0 if needs_positive_ic else True)
            and row["holdout_bad_mae_1r_rate"] <= 0.40
            and row["holdout_p90_mae_norm"] <= 4.00
            and row["holdout_wide_barrier_25bps_rate"] <= 0.05
            and holdout_timeout_ok
        )
        row["fit_sign_pass"] = bool(fit_sign)
        row["holdout_sign_pass"] = bool(holdout_sign)
        row["fit_economic_pass"] = bool(fit_economic)
        row["holdout_economic_pass"] = bool(holdout_economic)
        row["trainworthy_pass"] = bool(fit_economic and holdout_economic)
        rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(
        ["trainworthy_pass", "holdout_economic_pass", "fit_economic_pass", "holdout_mean_return_net"],
        ascending=[False, False, False, False],
    )


def _write_markdown(
    *,
    output_dir: Path,
    source_summary: pd.DataFrame,
    prevalence: pd.DataFrame,
    period_rows: pd.DataFrame,
    fit_holdout: pd.DataFrame,
    proxy_ic: pd.DataFrame,
    manifest: dict[str, Any],
) -> Path:
    path = output_dir / "source_conditioned_path_observability.md"
    fit_cols = [
        "trainworthy_pass",
        "fit_economic_pass",
        "holdout_economic_pass",
        "fit_sign_pass",
        "holdout_sign_pass",
        "source",
        "selector",
        "label_arm",
        "economic_arm",
        "top_frac",
        "fit_mean_return_net",
        "holdout_mean_return_net",
        "fit_bad_mae_1r_rate",
        "holdout_bad_mae_1r_rate",
        "fit_p90_mae_norm",
        "holdout_p90_mae_norm",
        "fit_timeout_rate",
        "holdout_timeout_rate",
        "fit_score_ic_u",
        "holdout_score_ic_u",
        "fit_selected_rows",
        "holdout_selected_rows",
        "holdout_selected_path_profit_low_mae_no_timeout_rate",
        "holdout_selected_path_decisive_profit_low_mae_rate",
    ]
    prevalence_cols = [
        "source",
        "month",
        "rows",
        "mean_return_net",
        "mean_u",
        "bad_mae_1r_rate",
        "p90_mae_norm",
        "timeout_rate",
        "path_profit_low_mae_no_timeout_rate",
        "path_decisive_profit_low_mae_rate",
        "path_bounded_rate",
    ]
    june_cols = [
        "source",
        "month",
        "period_type",
        "selector",
        "label_arm",
        "economic_arm",
        "top_frac",
        "selected_rows",
        "mean_return_net",
        "score_ic_u",
        "bad_mae_1r_rate",
        "p90_mae_norm",
        "timeout_rate",
        "selected_path_profit_low_mae_no_timeout_rate",
        "selected_path_decisive_profit_low_mae_rate",
        "selected_path_bounded_rate",
    ]
    lines = [
        "# Source-Conditioned Path Observability",
        "",
        "Scope: proxy-only label QA. Source masks are rebuilt from decision-time features; proxy scores are fitted only on earlier months and evaluated OOS by month/week.",
        "",
        f"Months: `{', '.join(manifest['months'])}`. Fit: `{', '.join(manifest['fit_months'])}`. Holdout: `{manifest['holdout_month']}`.",
        f"Labels: `{', '.join(manifest['label_arms'])}`.",
        f"Path targets: `{', '.join(manifest['path_targets'])}`.",
        f"Proxy top-k: `{manifest['proxy_top_k']}`. Top fractions: `{manifest['top_fracs']}`. Gate fractions: `{manifest['gate_fracs']}`.",
        "",
        "## Fit / Holdout",
        "",
        _table(fit_holdout, fit_cols, limit=80),
        "",
        "## Source Month Prevalence",
        "",
        _table(
            prevalence.sort_values(["month", "rows"], ascending=[True, False]),
            prevalence_cols,
            limit=80,
        ),
        "",
        "## June Month Selections",
        "",
        _table(
            period_rows[
                period_rows["period_type"].eq("month")
                & period_rows["month"].astype(str).eq(str(manifest["holdout_month"]))
            ].sort_values(["mean_return_net", "selected_path_decisive_profit_low_mae_rate"], ascending=[False, False]),
            june_cols,
            limit=80,
        ),
        "",
        "## June Proxy IC",
        "",
        _table(
            proxy_ic[proxy_ic["month"].astype(str).eq(str(manifest["holdout_month"]))].sort_values(
                ["source", "oos_ic_u"],
                ascending=[True, False],
            ),
            [
                "source",
                "month",
                "score_name",
                "label_arm",
                "path_target",
                "train_rows",
                "valid_rows",
                "oos_ic_score_target",
                "oos_ic_u",
                "oos_ic_bad_mae",
                "oos_ic_timeout",
                "proxy_features",
            ],
            limit=80,
        ),
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
                "timeout_rate",
                "mean_event_quality",
                "mean_time_edge_prior_quality",
                "mean_adverse_prior_safety",
                "mean_dual_prior_quality",
            ],
            limit=80,
        ),
        "",
        "## Outputs",
        "",
        f"- Source summary: `{manifest['outputs']['source_summary']}`",
        f"- Source month prevalence: `{manifest['outputs']['source_month_prevalence']}`",
        f"- Period rows: `{manifest['outputs']['period_rows']}`",
        f"- Fit/holdout: `{manifest['outputs']['fit_holdout']}`",
        f"- Proxy IC: `{manifest['outputs']['proxy_ic']}`",
        f"- Manifest: `{manifest['outputs']['manifest']}`",
    ]
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
    path_targets: list[str],
    sources: list[str],
    months: list[str],
    fit_months: list[str],
    holdout_month: str,
    top_fracs: list[float],
    gate_fracs: list[float],
    combine_label_weight: float,
    proxy_mode: str,
    proxy_top_k: int,
    min_train_source_rows: int,
    min_valid_source_rows: int,
    min_inner_folds: int,
    min_inner_train_rows: int,
    min_inner_valid_rows: int,
    min_consistency: float,
    min_week_rows: int,
    max_timeout_rate: float | None,
    run_gap_hours: float,
    include_causal_outcome_priors: bool,
    include_causal_state_path_priors: bool,
    include_causal_time_edge_priors: bool,
    include_event_confirmation_features: bool,
    include_adverse_path_composites: bool,
    prior_windows_days: list[float],
    prior_embargo_hours: float,
    state_path_prior_features: list[str],
    event_feature_store_features: list[str],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame, metrics, reports = _build_frame(
        labels_path=labels_path,
        feature_dir=feature_dir,
        feature_list_csv=feature_list_csv,
        max_feature_store_features=max_feature_store_features,
        include_causal_outcome_priors=include_causal_outcome_priors,
        include_causal_state_path_priors=include_causal_state_path_priors,
        include_event_confirmation_features=include_event_confirmation_features,
        include_adverse_path_composites=include_adverse_path_composites,
        prior_windows_days=prior_windows_days,
        prior_embargo_hours=prior_embargo_hours,
        state_path_prior_features=state_path_prior_features,
        event_feature_store_features=event_feature_store_features,
    )
    frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True, errors="coerce")
    reports["causal_time_edge_priors"] = {"enabled": False}
    if include_causal_time_edge_priors:
        time_edge_features, reports["causal_time_edge_priors"] = _causal_time_edge_prior_features(
            frame,
            metrics,
            windows_days=prior_windows_days,
            embargo_hours=prior_embargo_hours,
        )
        frame = pd.concat([frame, time_edge_features.astype(np.float32, copy=False)], axis=1).copy()

    context = _source_context(frame)
    frame = pd.concat([frame, context.astype(np.float32, copy=False)], axis=1).copy()
    source_masks_all = _build_sources(frame, context, run_gap_hours=run_gap_hours)
    missing_sources = sorted(set(sources).difference(source_masks_all))
    if missing_sources:
        raise ValueError(f"Unknown source(s): {missing_sources}")
    source_masks = {name: source_masks_all[name].reindex(frame.index, fill_value=False).astype(bool) for name in sources}

    features = _feature_columns(frame)
    path_features = _state_path_feature_columns(features)
    if not path_features:
        path_features = [feature for feature in features if feature.startswith("prior_")]

    label_target_map = _label_targets(frame, metrics)
    path_target_map_all = _state_path_risk_targets(metrics)
    missing_labels = sorted(set(label_arms).difference(label_target_map))
    missing_path = sorted(set(path_targets).difference(path_target_map_all))
    if missing_labels:
        raise ValueError(f"Unknown label arm(s): {missing_labels}")
    if missing_path:
        raise ValueError(f"Unknown path target(s): {missing_path}")
    label_target_map = {name: label_target_map[name] for name in label_arms}
    path_target_map = {name: path_target_map_all[name] for name in path_targets}

    month_series = frame["__ts__"].dt.to_period("M").astype(str)
    source_summary = _source_summary(frame=frame, metrics=metrics, context=context, sources=source_masks)
    prevalence = _source_month_prevalence(
        frame=frame,
        metrics=metrics,
        path_targets=path_target_map,
        sources=source_masks,
        months=months,
    )

    period_rows: list[dict[str, Any]] = []
    proxy_ic_rows: list[dict[str, Any]] = []
    for source_name, source_mask in source_masks.items():
        for month in months:
            train_mask = month_series.lt(str(month)) & source_mask
            valid_mask = month_series.eq(str(month)) & source_mask
            train_rows = int(train_mask.sum())
            valid_rows = int(valid_mask.sum())
            if train_rows < int(min_train_source_rows) or valid_rows < int(min_valid_source_rows):
                continue
            train = frame.loc[train_mask].copy()
            valid = frame.loc[valid_mask].copy()
            valid_metrics = metrics.loc[valid_mask].copy()

            label_scores: dict[str, pd.Series] = {}
            label_feature_diag: dict[str, str] = {}
            for label_arm in label_arms:
                score, diag = _proxy_score_for_mode(
                    proxy_mode=proxy_mode,
                    train=train,
                    valid=valid,
                    features=features,
                    y_train=label_target_map[label_arm].loc[train_mask, "target_soft"],
                    binary_target=False,
                    proxy_top_k=proxy_top_k,
                    min_inner_folds=min_inner_folds,
                    min_inner_train_rows=min_inner_train_rows,
                    min_inner_valid_rows=min_inner_valid_rows,
                    min_consistency=min_consistency,
                )
                label_scores[label_arm] = score.reset_index(drop=True)
                label_feature_diag[label_arm] = (
                    f"{diag['selection_mode']}|{','.join(diag.get('proxy_features', []))}|"
                    f"{diag.get('proxy_feature_directions', '')}"
                )
                target_valid = label_target_map[label_arm].loc[valid_mask, "target_soft"].reset_index(drop=True)
                proxy_ic_rows.append(
                    {
                        "source": source_name,
                        "month": month,
                        "score_name": "label_proxy",
                        "label_arm": label_arm,
                        "path_target": "",
                        "train_rows": train_rows,
                        "valid_rows": valid_rows,
                        "oos_ic_score_target": _spearman(score.reset_index(drop=True), target_valid),
                        "oos_ic_u": _spearman(score.reset_index(drop=True), valid_metrics["u_policy_net"].reset_index(drop=True)),
                        "oos_ic_bad_mae": _spearman(
                            score.reset_index(drop=True),
                            (valid_metrics["mae_norm"].reset_index(drop=True) >= 1.0).astype(float),
                        ),
                        "oos_ic_timeout": _spearman(
                            score.reset_index(drop=True),
                            valid_metrics["is_timeout"].astype(float).reset_index(drop=True),
                        ),
                        "proxy_features": label_feature_diag[label_arm],
                    }
                )

            path_scores: dict[str, pd.Series] = {}
            path_feature_diag: dict[str, str] = {}
            for path_name in path_targets:
                score, diag = _proxy_score_for_mode(
                    proxy_mode=proxy_mode,
                    train=train,
                    valid=valid,
                    features=path_features,
                    y_train=path_target_map[path_name].loc[train_mask],
                    binary_target=path_name != "path_quality",
                    proxy_top_k=proxy_top_k,
                    min_inner_folds=min_inner_folds,
                    min_inner_train_rows=min_inner_train_rows,
                    min_inner_valid_rows=min_inner_valid_rows,
                    min_consistency=min_consistency,
                )
                path_scores[path_name] = score.reset_index(drop=True)
                path_feature_diag[path_name] = (
                    f"{diag['selection_mode']}|{','.join(diag.get('proxy_features', []))}|"
                    f"{diag.get('proxy_feature_directions', '')}"
                )
                path_valid = path_target_map[path_name].loc[valid_mask].reset_index(drop=True)
                proxy_ic_rows.append(
                    {
                        "source": source_name,
                        "month": month,
                        "score_name": "path_proxy",
                        "label_arm": "",
                        "path_target": path_name,
                        "train_rows": train_rows,
                        "valid_rows": valid_rows,
                        "oos_ic_score_target": _spearman(score.reset_index(drop=True), path_valid),
                        "oos_ic_u": _spearman(score.reset_index(drop=True), valid_metrics["u_policy_net"].reset_index(drop=True)),
                        "oos_ic_bad_mae": _spearman(
                            score.reset_index(drop=True),
                            (valid_metrics["mae_norm"].reset_index(drop=True) >= 1.0).astype(float),
                        ),
                        "oos_ic_timeout": _spearman(
                            score.reset_index(drop=True),
                            valid_metrics["is_timeout"].astype(float).reset_index(drop=True),
                        ),
                        "proxy_features": path_feature_diag[path_name],
                    }
                )

            selected_path_targets = {
                name: target.loc[valid_mask].reset_index(drop=True) for name, target in path_target_map.items()
            }
            for label_arm in label_arms:
                target_valid = label_target_map[label_arm].loc[valid_mask].reset_index(drop=True)
                label_proxy = label_scores[label_arm]
                selector_prefix = "stable" if str(proxy_mode).strip().lower() == "stable" else "fit_ic"
                specs: list[dict[str, Any]] = [
                    {
                        "selector": "oracle_label_sort",
                        "economic_arm": "none",
                        "score": target_valid["target_soft"],
                        "economic_score": None,
                        "economic_target": None,
                        "economic_features": "",
                    },
                    {
                        "selector": f"{selector_prefix}_label_proxy_oos",
                        "economic_arm": "none",
                        "score": label_proxy,
                        "economic_score": None,
                        "economic_target": None,
                        "economic_features": "",
                    },
                ]
                for path_name, path_proxy in path_scores.items():
                    path_valid = selected_path_targets[path_name]
                    specs.extend(
                        [
                            {
                                "selector": f"oracle_path_{path_name}_sort",
                                "economic_arm": f"path::{path_name}",
                                "score": path_valid,
                                "economic_score": path_valid,
                                "economic_target": path_valid,
                                "economic_features": "oracle_path_target",
                            },
                            {
                                "selector": f"{selector_prefix}_path_{path_name}_proxy_oos",
                                "economic_arm": f"path::{path_name}",
                                "score": path_proxy,
                                "economic_score": path_proxy,
                                "economic_target": path_valid,
                                "economic_features": path_feature_diag[path_name],
                            },
                            {
                                "selector": (
                                    f"{selector_prefix}_combined_l{combine_label_weight:.2f}"
                                    f"_label_path_{path_name}_oos"
                                ),
                                "economic_arm": f"path::{path_name}",
                                "score": combine_label_weight * label_proxy + (1.0 - combine_label_weight) * path_proxy,
                                "economic_score": path_proxy,
                                "economic_target": path_valid,
                                "economic_features": path_feature_diag[path_name],
                            },
                        ]
                    )
                    for gate_frac in gate_fracs:
                        path_gate = _top_gate(path_proxy, gate_frac)
                        label_gate = _top_gate(label_proxy, gate_frac)
                        specs.extend(
                            [
                                {
                                    "selector": (
                                        f"{selector_prefix}_path_{path_name}_gate{gate_frac:.2f}"
                                        "_then_label_oos"
                                    ),
                                    "economic_arm": f"path::{path_name}",
                                    "score": label_proxy.where(path_gate),
                                    "economic_score": path_proxy,
                                    "economic_target": path_valid,
                                    "economic_features": path_feature_diag[path_name],
                                },
                                {
                                    "selector": (
                                        f"{selector_prefix}_dual_label_path_{path_name}"
                                        f"_gate{gate_frac:.2f}_oos"
                                    ),
                                    "economic_arm": f"path::{path_name}",
                                    "score": label_proxy.where(label_gate & path_gate),
                                    "economic_score": path_proxy,
                                    "economic_target": path_valid,
                                    "economic_features": path_feature_diag[path_name],
                                },
                            ]
                        )

                for spec in specs:
                    period_rows.extend(
                        _period_rows_for_selector(
                            valid=valid,
                            valid_metrics=valid_metrics,
                            target=target_valid,
                            score=spec["score"],
                            month=month,
                            source=source_name,
                            selector=spec["selector"],
                            label_arm=label_arm,
                            economic_arm=spec["economic_arm"],
                            top_fracs=top_fracs,
                            label_score=target_valid["target_soft"],
                            economic_score=spec["economic_score"],
                            economic_target=spec["economic_target"],
                            label_proxy_features=label_feature_diag[label_arm],
                            economic_proxy_features=spec["economic_features"],
                            selected_path_targets=selected_path_targets,
                            train_rows=train_rows,
                            valid_rows=valid_rows,
                        )
                    )

    period_frame = pd.DataFrame(period_rows)
    proxy_ic = pd.DataFrame(proxy_ic_rows)
    fit_holdout = _fit_holdout_summary(
        period_frame,
        fit_months=fit_months,
        holdout_month=holdout_month,
        min_week_rows=min_week_rows,
        max_timeout_rate=max_timeout_rate,
    )

    paths = {
        "source_summary": output_dir / "source_conditioned_source_summary.csv",
        "source_month_prevalence": output_dir / "source_conditioned_month_prevalence.csv",
        "period_rows": output_dir / "source_conditioned_period_rows.csv",
        "fit_holdout": output_dir / "source_conditioned_fit_holdout.csv",
        "proxy_ic": output_dir / "source_conditioned_proxy_ic.csv",
        "manifest": output_dir / "manifest.json",
    }
    source_summary.to_csv(paths["source_summary"], index=False)
    prevalence.to_csv(paths["source_month_prevalence"], index=False)
    period_frame.to_csv(paths["period_rows"], index=False)
    fit_holdout.to_csv(paths["fit_holdout"], index=False)
    proxy_ic.to_csv(paths["proxy_ic"], index=False)

    manifest = {
        "scope": "source_conditioned_path_observability",
        "labels_path": str(labels_path),
        "output_dir": str(output_dir),
        "rows": int(len(frame)),
        "timestamp_min": frame["__ts__"].min(),
        "timestamp_max": frame["__ts__"].max(),
        "symbols": int(frame["__symbol__"].nunique(dropna=True)),
        "feature_dir": str(feature_dir),
        "feature_list_csv": str(feature_list_csv),
        "max_feature_store_features": max_feature_store_features,
        "feature_count": int(len(features)),
        "path_feature_count": int(len(path_features)),
        "label_arms": list(label_arms),
        "path_targets": list(path_targets),
        "sources": list(sources),
        "months": list(months),
        "fit_months": list(fit_months),
        "holdout_month": str(holdout_month),
        "top_fracs": [float(v) for v in top_fracs],
        "gate_fracs": [float(v) for v in gate_fracs],
        "combine_label_weight": float(combine_label_weight),
        "proxy_mode": str(proxy_mode),
        "proxy_top_k": int(proxy_top_k),
        "min_train_source_rows": int(min_train_source_rows),
        "min_valid_source_rows": int(min_valid_source_rows),
        "min_inner_folds": int(min_inner_folds),
        "min_inner_train_rows": int(min_inner_train_rows),
        "min_inner_valid_rows": int(min_inner_valid_rows),
        "min_consistency": float(min_consistency),
        "min_week_rows": int(min_week_rows),
        "max_timeout_rate": max_timeout_rate,
        "run_gap_hours": float(run_gap_hours),
        "include_causal_outcome_priors": bool(include_causal_outcome_priors),
        "include_causal_state_path_priors": bool(include_causal_state_path_priors),
        "include_causal_time_edge_priors": bool(include_causal_time_edge_priors),
        "include_event_confirmation_features": bool(include_event_confirmation_features),
        "include_adverse_path_composites": bool(include_adverse_path_composites),
        "prior_windows_days": [float(v) for v in prior_windows_days],
        "prior_embargo_hours": float(prior_embargo_hours),
        "reports": reports,
        "outputs": {key: str(value) for key, value in paths.items()},
    }
    markdown = _write_markdown(
        output_dir=output_dir,
        source_summary=source_summary,
        prevalence=prevalence,
        period_rows=period_frame,
        fit_holdout=fit_holdout,
        proxy_ic=proxy_ic,
        manifest={**manifest, "outputs": {**manifest["outputs"], "markdown": str(output_dir / "source_conditioned_path_observability.md")}},
    )
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
    parser.add_argument("--label-arms", type=lambda value: _parse_csv(value, DEFAULT_LABEL_ARMS), default=list(DEFAULT_LABEL_ARMS))
    parser.add_argument("--path-targets", type=lambda value: _parse_csv(value, DEFAULT_PATH_TARGETS), default=list(DEFAULT_PATH_TARGETS))
    parser.add_argument("--sources", type=lambda value: _parse_csv(value, DEFAULT_SOURCES), default=list(DEFAULT_SOURCES))
    parser.add_argument("--months", type=lambda value: _parse_csv(value, DEFAULT_MONTHS), default=list(DEFAULT_MONTHS))
    parser.add_argument("--fit-months", type=lambda value: _parse_csv(value, DEFAULT_FIT_MONTHS), default=list(DEFAULT_FIT_MONTHS))
    parser.add_argument("--holdout-month", type=str, default=DEFAULT_HOLDOUT_MONTH)
    parser.add_argument("--top-fracs", type=lambda value: _parse_float_csv(value), default=list(DEFAULT_TOP_FRACS))
    parser.add_argument("--gate-fracs", type=lambda value: _parse_float_csv(value), default=list(DEFAULT_GATE_FRACS))
    parser.add_argument("--combine-label-weight", type=float, default=0.50)
    parser.add_argument("--proxy-mode", choices=("fit_ic", "stable"), default=DEFAULT_PROXY_MODE)
    parser.add_argument("--proxy-top-k", type=int, default=DEFAULT_PROXY_TOP_K)
    parser.add_argument("--min-train-source-rows", type=int, default=300)
    parser.add_argument("--min-valid-source-rows", type=int, default=30)
    parser.add_argument("--min-inner-folds", type=int, default=1)
    parser.add_argument("--min-inner-train-rows", type=int, default=250)
    parser.add_argument("--min-inner-valid-rows", type=int, default=30)
    parser.add_argument("--min-consistency", type=float, default=0.50)
    parser.add_argument("--min-week-rows", type=int, default=10)
    parser.add_argument("--max-timeout-rate", type=float, default=0.50)
    parser.add_argument("--run-gap-hours", type=float, default=24.0)
    parser.add_argument("--include-causal-outcome-priors", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-causal-state-path-priors", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-causal-time-edge-priors", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-event-confirmation-features", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-adverse-path-composites", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prior-windows-days", type=lambda value: _parse_float_csv(value), default=list(DEFAULT_PRIOR_WINDOWS_DAYS))
    parser.add_argument("--prior-embargo-hours", type=float, default=24.0)
    parser.add_argument(
        "--state-path-prior-features",
        type=lambda value: _parse_csv(value, tuple(DEFAULT_STATE_PATH_PRIOR_FEATURES)),
        default=list(DEFAULT_STATE_PATH_PRIOR_FEATURES),
    )
    parser.add_argument(
        "--event-feature-store-features",
        type=lambda value: _parse_csv(value, tuple(DEFAULT_EVENT_FEATURE_STORE_FEATURES)),
        default=list(DEFAULT_EVENT_FEATURE_STORE_FEATURES),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = run_report(
        labels_path=args.labels_path,
        output_dir=args.output_dir,
        feature_dir=args.feature_dir,
        feature_list_csv=args.feature_list_csv,
        max_feature_store_features=args.max_feature_store_features,
        label_arms=list(args.label_arms),
        path_targets=list(args.path_targets),
        sources=list(args.sources),
        months=list(args.months),
        fit_months=list(args.fit_months),
        holdout_month=str(args.holdout_month),
        top_fracs=list(args.top_fracs),
        gate_fracs=list(args.gate_fracs),
        combine_label_weight=float(args.combine_label_weight),
        proxy_mode=str(args.proxy_mode),
        proxy_top_k=int(args.proxy_top_k),
        min_train_source_rows=int(args.min_train_source_rows),
        min_valid_source_rows=int(args.min_valid_source_rows),
        min_inner_folds=int(args.min_inner_folds),
        min_inner_train_rows=int(args.min_inner_train_rows),
        min_inner_valid_rows=int(args.min_inner_valid_rows),
        min_consistency=float(args.min_consistency),
        min_week_rows=int(args.min_week_rows),
        max_timeout_rate=args.max_timeout_rate,
        run_gap_hours=float(args.run_gap_hours),
        include_causal_outcome_priors=bool(args.include_causal_outcome_priors),
        include_causal_state_path_priors=bool(args.include_causal_state_path_priors),
        include_causal_time_edge_priors=bool(args.include_causal_time_edge_priors),
        include_event_confirmation_features=bool(args.include_event_confirmation_features),
        include_adverse_path_composites=bool(args.include_adverse_path_composites),
        prior_windows_days=list(args.prior_windows_days),
        prior_embargo_hours=float(args.prior_embargo_hours),
        state_path_prior_features=list(args.state_path_prior_features),
        event_feature_store_features=list(args.event_feature_store_features),
    )
    print(json.dumps(_json_safe({"output_dir": manifest["output_dir"], "outputs": manifest["outputs"]}), indent=2))


if __name__ == "__main__":
    main()
