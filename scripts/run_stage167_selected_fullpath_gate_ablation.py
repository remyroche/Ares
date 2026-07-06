#!/usr/bin/env python3
"""Causal post-selection gate sweep for the Stage167 full-path tail.

This diagnostic starts from the Stage167 month-forward selected-row ledger.
For each holdout month it fits simple gates on earlier selected rows only,
then applies the selected gate to the holdout month. It is deliberately small
and transparent: the goal is to see whether current causal features can reduce
full-path adverse excursion after Stage167 has already selected a trade.
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

from scripts.run_first_touch_label_training_smoke import _table  # noqa: E402
from scripts.run_label_quality_proxy_diagnostics import (  # noqa: E402
    DEFAULT_FEATURE_DIR,
    DEFAULT_FEATURE_LIST_CSV,
    _json_safe,
    _load_feature_store_columns,
    _read_feature_list,
    _safe_mean,
    _safe_quantile,
)


DEFAULT_LEDGER_CSV = Path(
    "data_perp/reports/stage167_full_path_tail_feature_gap_v1/"
    "stage167_full_path_tail_selected_ledger.csv"
)
DEFAULT_OUTPUT_DIR = Path("data_perp/reports/stage169_stage167_selected_fullpath_gate_sum_delta_no_score_v1")
DEFAULT_MONTHS = ("2026-04", "2026-05", "2026-06")


@dataclass(frozen=True)
class Condition:
    column: str
    op: str
    threshold: float


@dataclass(frozen=True)
class Gate:
    name: str
    source: str
    conditions: tuple[Condition, ...]


def _safe_numeric(values: Any) -> pd.Series:
    if isinstance(values, pd.Series):
        return pd.to_numeric(values, errors="coerce")
    return pd.to_numeric(pd.Series(values), errors="coerce")


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


def _load_ledger(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    required = {
        "__ts__",
        "__symbol__",
        "period",
        "first_touch_net",
        "clean_first_touch_exec",
        "first_touch_timeout",
        "first_touch_mae_to_sl",
        "full_path_mae_to_sl",
        "score",
        "utility_pred",
        "support_pred",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    frame["__ts__"] = pd.to_datetime(frame["__ts__"], errors="coerce")
    frame["period"] = frame["period"].astype(str)
    frame["__symbol__"] = frame["__symbol__"].astype(str)
    return frame.sort_values(["__ts__", "__symbol__"], kind="mergesort").reset_index(drop=True)


def _attach_features(
    ledger: pd.DataFrame,
    *,
    feature_dir: Path,
    feature_list_csv: Path,
    max_gate_features: int,
) -> tuple[pd.DataFrame, dict[str, Any], list[str]]:
    requested = _read_feature_list(feature_list_csv, max_features=max_gate_features)
    feature_frame = pd.DataFrame(
        {
            "__ts__": pd.to_datetime(ledger["__ts__"], errors="coerce"),
            "__symbol__": ledger["__symbol__"].astype(str),
        },
    )
    matrix, manifest = _load_feature_store_columns(
        feature_frame,
        feature_dir=feature_dir,
        selected_features=requested,
    )
    out = ledger.reset_index(drop=True).copy()
    if not matrix.empty:
        out = pd.concat([out, matrix.reset_index(drop=True)], axis=1)
    usable = [feature for feature in requested if feature in out.columns]
    return out, manifest, usable


def _full_path_safe_mask(
    frame: pd.DataFrame,
    *,
    first_touch_clean_r: float,
    full_path_safe_r: float,
) -> pd.Series:
    return (
        (_safe_numeric(frame["clean_first_touch_exec"]) >= 0.5)
        & (_safe_numeric(frame["first_touch_timeout"]) < 0.5)
        & (_safe_numeric(frame["first_touch_net"]) > 0.0)
        & (_safe_numeric(frame["first_touch_mae_to_sl"]) <= float(first_touch_clean_r))
        & (_safe_numeric(frame["full_path_mae_to_sl"]) <= float(full_path_safe_r))
    ).fillna(False)


def _week_summary(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {
            "weeks": 0,
            "positive_weeks": 0,
            "positive_week_rate": float("nan"),
            "worst_week_first_touch_net": float("nan"),
        }
    weeks = pd.to_datetime(frame["__ts__"], errors="coerce").dt.to_period("W-SUN").astype(str)
    weekly = (
        frame.assign(week=weeks)
        .groupby("week", observed=True)
        .agg(rows=("first_touch_net", "size"), mean_first_touch_net=("first_touch_net", "mean"))
        .reset_index()
    )
    positive = _safe_numeric(weekly["mean_first_touch_net"]) > 0.0
    return {
        "weeks": int(len(weekly)),
        "positive_weeks": int(positive.sum()),
        "positive_week_rate": float(positive.mean()) if len(positive) else float("nan"),
        "worst_week_first_touch_net": float(_safe_numeric(weekly["mean_first_touch_net"]).min())
        if len(weekly)
        else float("nan"),
    }


def _metrics(
    frame: pd.DataFrame,
    *,
    first_touch_clean_r: float,
    full_path_safe_r: float,
    full_path_dirty_r: float,
) -> dict[str, Any]:
    rows = int(len(frame))
    weeks = _week_summary(frame)
    full_mae = _safe_numeric(frame.get("full_path_mae_to_sl"))
    first_mae = _safe_numeric(frame.get("first_touch_mae_to_sl"))
    timeout = _safe_numeric(frame.get("first_touch_timeout"))
    clean = _safe_numeric(frame.get("clean_first_touch_exec"))
    net = _safe_numeric(frame.get("first_touch_net"))
    full_safe = _full_path_safe_mask(
        frame,
        first_touch_clean_r=first_touch_clean_r,
        full_path_safe_r=full_path_safe_r,
    )
    top_symbol_share = (
        float(frame["__symbol__"].astype(str).value_counts(normalize=True).iloc[0]) if rows else float("nan")
    )
    out = {
        "rows": rows,
        "mean_first_touch_net": _safe_mean(net),
        "sum_first_touch_net": float(net.sum()) if rows else 0.0,
        "hit_first_touch_net_rate": _safe_mean(net > 0.0),
        "clean_first_touch_exec_rate": _safe_mean(clean),
        "first_touch_timeout_rate": _safe_mean(timeout >= 0.5),
        "bad_first_touch_mae_to_sl_rate": _safe_mean(first_mae >= float(first_touch_clean_r)),
        "p90_first_touch_mae_to_sl": _safe_quantile(first_mae, 0.90),
        "full_path_safe_rate": _safe_mean(full_safe),
        "bad_full_path_mae_3r_rate": _safe_mean(full_mae >= float(full_path_dirty_r)),
        "p90_full_path_mae_to_sl": _safe_quantile(full_mae, 0.90),
        "top_symbol_share": top_symbol_share,
    }
    out.update(weeks)
    return out


def _objective(
    metrics: dict[str, Any],
    *,
    full_path_dirty_r: float,
    full_path_bad_weight: float,
    full_path_p90_weight: float,
    first_touch_bad_weight: float,
    timeout_weight: float,
) -> float:
    if int(metrics.get("rows", 0) or 0) <= 0:
        return -1.0e9
    mean_net = float(metrics.get("mean_first_touch_net", float("nan")))
    if not math.isfinite(mean_net):
        return -1.0e9
    bad_full = float(metrics.get("bad_full_path_mae_3r_rate", 1.0) or 0.0)
    p90_full = float(metrics.get("p90_full_path_mae_to_sl", full_path_dirty_r) or full_path_dirty_r)
    bad_first = float(metrics.get("bad_first_touch_mae_to_sl_rate", 1.0) or 0.0)
    timeout = float(metrics.get("first_touch_timeout_rate", 1.0) or 0.0)
    return (
        mean_net
        - float(full_path_bad_weight) * bad_full
        - float(full_path_p90_weight) * max(0.0, p90_full - float(full_path_dirty_r))
        - float(first_touch_bad_weight) * bad_first
        - float(timeout_weight) * timeout
    )


def _portfolio_objective(
    metrics: dict[str, Any],
    *,
    full_path_dirty_r: float,
    full_path_bad_weight: float,
    full_path_p90_weight: float,
    first_touch_bad_weight: float,
    timeout_weight: float,
) -> float:
    rows = int(metrics.get("rows", 0) or 0)
    if rows <= 0:
        return -1.0e9
    total_net = float(metrics.get("sum_first_touch_net", float("nan")))
    if not math.isfinite(total_net):
        return -1.0e9
    bad_full = float(metrics.get("bad_full_path_mae_3r_rate", 1.0) or 0.0)
    p90_full = float(metrics.get("p90_full_path_mae_to_sl", full_path_dirty_r) or full_path_dirty_r)
    bad_first = float(metrics.get("bad_first_touch_mae_to_sl_rate", 1.0) or 0.0)
    timeout = float(metrics.get("first_touch_timeout_rate", 1.0) or 0.0)
    per_row_penalty = (
        float(full_path_bad_weight) * bad_full
        + float(full_path_p90_weight) * max(0.0, p90_full - float(full_path_dirty_r))
        + float(first_touch_bad_weight) * bad_first
        + float(timeout_weight) * timeout
    )
    return total_net - (float(rows) * per_row_penalty)


def _apply_gate(frame: pd.DataFrame, gate: Gate) -> pd.Series:
    if not gate.conditions:
        return pd.Series(True, index=frame.index)
    mask = pd.Series(True, index=frame.index)
    for condition in gate.conditions:
        values = _safe_numeric(frame[condition.column])
        if condition.op == "<=":
            mask &= values <= float(condition.threshold)
        elif condition.op == ">=":
            mask &= values >= float(condition.threshold)
        else:
            raise ValueError(f"Unsupported gate op: {condition.op}")
    return mask.fillna(False).astype(bool)


def _format_threshold(value: float) -> str:
    if not math.isfinite(float(value)):
        return "nan"
    return f"{float(value):.6g}"


def _candidate_gates(
    fit: pd.DataFrame,
    *,
    feature_columns: list[str],
    quantiles: list[float],
    include_score_gates: bool,
) -> list[Gate]:
    gates = [Gate("no_gate", "baseline", ())]
    score_columns = ["score", "utility_pred", "support_pred"] if include_score_gates else []
    for source, columns in (("feature", feature_columns), ("selector_score", score_columns)):
        for column in columns:
            if column not in fit.columns:
                continue
            values = _safe_numeric(fit[column]).replace([np.inf, -np.inf], np.nan).dropna()
            if len(values) < 12 or int(values.nunique(dropna=True)) < 4:
                continue
            seen: set[tuple[str, float]] = set()
            for quantile in quantiles:
                threshold = float(values.quantile(float(quantile)))
                if not math.isfinite(threshold):
                    continue
                for op in ("<=", ">="):
                    key = (op, round(threshold, 12))
                    if key in seen:
                        continue
                    seen.add(key)
                    q_label = f"q{int(float(quantile) * 100)}"
                    name = f"{column}{op}fit_{q_label}({_format_threshold(threshold)})"
                    gates.append(Gate(name, source, (Condition(column, op, threshold),)))
    return gates


def _summarize_weekly(frame: pd.DataFrame, *, period: str, variant: str) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    weeks = pd.to_datetime(frame["__ts__"], errors="coerce").dt.to_period("W-SUN").astype(str)
    rows: list[dict[str, Any]] = []
    for week, group in frame.assign(week=weeks).groupby("week", observed=True):
        metrics = _metrics(
            group,
            first_touch_clean_r=1.0,
            full_path_safe_r=3.0,
            full_path_dirty_r=3.0,
        )
        rows.append(
            {
                "period": str(period),
                "week": str(week),
                "variant": str(variant),
                "rows": int(metrics["rows"]),
                "mean_first_touch_net": metrics["mean_first_touch_net"],
                "sum_first_touch_net": metrics["sum_first_touch_net"],
                "clean_first_touch_exec_rate": metrics["clean_first_touch_exec_rate"],
                "first_touch_timeout_rate": metrics["first_touch_timeout_rate"],
                "bad_first_touch_mae_to_sl_rate": metrics["bad_first_touch_mae_to_sl_rate"],
                "bad_full_path_mae_3r_rate": metrics["bad_full_path_mae_3r_rate"],
                "p90_full_path_mae_to_sl": metrics["p90_full_path_mae_to_sl"],
            }
        )
    return rows


def _gate_row(
    *,
    period: str,
    gate: Gate,
    fit: pd.DataFrame,
    holdout: pd.DataFrame,
    min_fit_rows: int,
    min_fit_keep_frac: float,
    max_fit_keep_frac: float,
    objective_kwargs: dict[str, float],
    baseline_fit_metrics: dict[str, Any],
    baseline_holdout_metrics: dict[str, Any],
    selection_mode: str,
    first_touch_clean_r: float,
    full_path_safe_r: float,
    full_path_dirty_r: float,
) -> dict[str, Any]:
    fit_mask = _apply_gate(fit, gate)
    holdout_mask = _apply_gate(holdout, gate)
    fit_kept = fit.loc[fit_mask].copy()
    holdout_kept = holdout.loc[holdout_mask].copy()
    fit_metrics = _metrics(
        fit_kept,
        first_touch_clean_r=first_touch_clean_r,
        full_path_safe_r=full_path_safe_r,
        full_path_dirty_r=full_path_dirty_r,
    )
    holdout_metrics = _metrics(
        holdout_kept,
        first_touch_clean_r=first_touch_clean_r,
        full_path_safe_r=full_path_safe_r,
        full_path_dirty_r=full_path_dirty_r,
    )
    fit_keep_frac = float(len(fit_kept) / len(fit)) if len(fit) else float("nan")
    holdout_keep_frac = float(len(holdout_kept) / len(holdout)) if len(holdout) else float("nan")
    fit_objective = _objective(fit_metrics, **objective_kwargs)
    holdout_objective = _objective(holdout_metrics, **objective_kwargs)
    fit_portfolio_objective = _portfolio_objective(fit_metrics, **objective_kwargs)
    holdout_portfolio_objective = _portfolio_objective(holdout_metrics, **objective_kwargs)
    baseline_fit_objective = _objective(baseline_fit_metrics, **objective_kwargs)
    baseline_holdout_objective = _objective(baseline_holdout_metrics, **objective_kwargs)
    baseline_fit_portfolio_objective = _portfolio_objective(baseline_fit_metrics, **objective_kwargs)
    baseline_holdout_portfolio_objective = _portfolio_objective(baseline_holdout_metrics, **objective_kwargs)
    if selection_mode == "quality":
        fit_selection_score = fit_objective
        holdout_selection_score = holdout_objective
    elif selection_mode == "portfolio_delta":
        fit_selection_score = fit_portfolio_objective - baseline_fit_portfolio_objective
        holdout_selection_score = holdout_portfolio_objective - baseline_holdout_portfolio_objective
    elif selection_mode == "sum_delta":
        fit_selection_score = float(fit_metrics.get("sum_first_touch_net", 0.0)) - float(
            baseline_fit_metrics.get("sum_first_touch_net", 0.0),
        )
        holdout_selection_score = float(holdout_metrics.get("sum_first_touch_net", 0.0)) - float(
            baseline_holdout_metrics.get("sum_first_touch_net", 0.0),
        )
    else:
        raise ValueError(f"Unknown selection mode: {selection_mode}")
    eligible = (
        gate.name == "no_gate"
        or (
            fit_metrics["rows"] >= int(min_fit_rows)
            and fit_keep_frac >= float(min_fit_keep_frac)
            and fit_keep_frac <= float(max_fit_keep_frac)
            and float(fit_metrics["mean_first_touch_net"]) > 0.0
            and (selection_mode not in {"portfolio_delta", "sum_delta"} or fit_selection_score > 0.0)
        )
    )
    row: dict[str, Any] = {
        "period": str(period),
        "gate_name": gate.name,
        "gate_source": gate.source,
        "gate_condition_count": int(len(gate.conditions)),
        "fit_rows_available": int(len(fit)),
        "holdout_rows_available": int(len(holdout)),
        "fit_keep_frac": fit_keep_frac,
        "holdout_keep_frac": holdout_keep_frac,
        "fit_objective": fit_objective,
        "holdout_objective": holdout_objective,
        "fit_portfolio_objective": fit_portfolio_objective,
        "holdout_portfolio_objective": holdout_portfolio_objective,
        "baseline_fit_objective": baseline_fit_objective,
        "baseline_holdout_objective": baseline_holdout_objective,
        "baseline_fit_portfolio_objective": baseline_fit_portfolio_objective,
        "baseline_holdout_portfolio_objective": baseline_holdout_portfolio_objective,
        "fit_selection_score": fit_selection_score,
        "holdout_selection_score": holdout_selection_score,
        "fit_eligible_for_selection": bool(eligible),
    }
    for prefix, metrics in (("fit", fit_metrics), ("holdout", holdout_metrics)):
        for key, value in metrics.items():
            row[f"{prefix}_{key}"] = value
    return row


def _write_markdown(
    *,
    output_dir: Path,
    monthly: pd.DataFrame,
    candidates: pd.DataFrame,
    weekly: pd.DataFrame,
    manifest: dict[str, Any],
) -> Path:
    path = output_dir / "stage169_stage167_selected_fullpath_gate.md"
    cols = [
        "period",
        "variant",
        "gate_name",
        "rows",
        "keep_frac",
        "mean_first_touch_net",
        "sum_first_touch_net",
        "clean_first_touch_exec_rate",
        "first_touch_timeout_rate",
        "bad_first_touch_mae_to_sl_rate",
        "bad_full_path_mae_3r_rate",
        "p90_full_path_mae_to_sl",
        "positive_week_rate",
        "worst_week_first_touch_net",
    ]
    cand_cols = [
        "period",
        "gate_name",
        "gate_source",
        "fit_rows",
        "fit_keep_frac",
        "fit_selection_score",
        "fit_objective",
        "fit_portfolio_objective",
        "fit_mean_first_touch_net",
        "fit_bad_full_path_mae_3r_rate",
        "fit_p90_full_path_mae_to_sl",
        "holdout_rows",
        "holdout_keep_frac",
        "holdout_selection_score",
        "holdout_objective",
        "holdout_portfolio_objective",
        "holdout_mean_first_touch_net",
        "holdout_bad_full_path_mae_3r_rate",
        "holdout_p90_full_path_mae_to_sl",
    ]
    weekly_cols = [
        "period",
        "week",
        "variant",
        "rows",
        "mean_first_touch_net",
        "sum_first_touch_net",
        "bad_full_path_mae_3r_rate",
        "p90_full_path_mae_to_sl",
    ]
    selected_candidates = candidates[candidates.get("selected_gate", False).astype(bool)] if not candidates.empty else candidates
    lines = [
        "# Stage169 Stage167 Selected Full-Path Gate",
        "",
        "Scope: causal diagnostic. Gates are fitted only on earlier Stage167 selected rows and applied to the next month. April is pass-through because this selected-ledger artifact has no prior selected month.",
        "",
        f"Selected ledger: `{manifest['ledger_csv']}`",
        f"Feature dir: `{manifest['feature_dir']}`",
        f"Feature list: `{manifest['feature_list_csv']}`",
        f"Gate feature count: `{manifest['gate_feature_count']}`",
        f"Gate search breadth: `{manifest['gate_search_breadth']}`",
        f"Selection mode: `{manifest['selection_mode']}`",
        f"Objective: `{manifest['objective']}`",
        "",
        "## Monthly Baseline vs Causal Gate",
        "",
        _table(monthly, cols, limit=80),
        "",
        "## Selected Gate Evidence",
        "",
        _table(selected_candidates, cand_cols, limit=80),
        "",
        "## Weekly Rows",
        "",
        _table(weekly, weekly_cols, limit=120),
        "",
        "## Outputs",
        "",
        f"- Monthly: `{manifest['outputs']['monthly']}`",
        f"- Candidates: `{manifest['outputs']['candidates']}`",
        f"- Weekly: `{manifest['outputs']['weekly']}`",
        f"- Manifest: `{manifest['outputs']['manifest']}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_ablation(
    *,
    ledger_csv: Path,
    output_dir: Path,
    feature_dir: Path,
    feature_list_csv: Path,
    max_gate_features: int,
    months: list[str],
    quantiles: list[float],
    min_fit_rows: int,
    min_fit_keep_frac: float,
    max_fit_keep_frac: float,
    include_score_gates: bool,
    selection_mode: str,
    first_touch_clean_r: float,
    full_path_safe_r: float,
    full_path_dirty_r: float,
    full_path_bad_weight: float,
    full_path_p90_weight: float,
    first_touch_bad_weight: float,
    timeout_weight: float,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    ledger = _load_ledger(ledger_csv)
    ledger, feature_manifest, gate_features = _attach_features(
        ledger,
        feature_dir=feature_dir,
        feature_list_csv=feature_list_csv,
        max_gate_features=max_gate_features,
    )
    objective_kwargs = {
        "full_path_dirty_r": float(full_path_dirty_r),
        "full_path_bad_weight": float(full_path_bad_weight),
        "full_path_p90_weight": float(full_path_p90_weight),
        "first_touch_bad_weight": float(first_touch_bad_weight),
        "timeout_weight": float(timeout_weight),
    }

    monthly_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    weekly_rows: list[dict[str, Any]] = []
    search_breadth: dict[str, int] = {}

    for period in months:
        fit = ledger[ledger["period"].astype(str) < str(period)].copy()
        holdout = ledger[ledger["period"].astype(str).eq(str(period))].copy()
        if holdout.empty:
            continue
        baseline_metrics = _metrics(
            holdout,
            first_touch_clean_r=first_touch_clean_r,
            full_path_safe_r=full_path_safe_r,
            full_path_dirty_r=full_path_dirty_r,
        )
        monthly_rows.append(
            {
                "period": str(period),
                "variant": "stage167_baseline",
                "gate_name": "no_gate",
                "rows": baseline_metrics["rows"],
                "keep_frac": 1.0,
                **baseline_metrics,
            }
        )
        weekly_rows.extend(_summarize_weekly(holdout, period=str(period), variant="stage167_baseline"))

        if len(fit) < int(min_fit_rows):
            gate = Gate("no_prior_selected_rows_pass_through", "no_prior", ())
            gate_metrics = baseline_metrics
            monthly_rows.append(
                {
                    "period": str(period),
                    "variant": "causal_gate",
                    "gate_name": gate.name,
                    "rows": gate_metrics["rows"],
                    "keep_frac": 1.0,
                    **gate_metrics,
                }
            )
            weekly_rows.extend(_summarize_weekly(holdout, period=str(period), variant="causal_gate"))
            candidate_rows.append(
                {
                    "period": str(period),
                    "gate_name": gate.name,
                    "gate_source": gate.source,
                    "selected_gate": True,
                    "fit_rows_available": int(len(fit)),
                    "holdout_rows_available": int(len(holdout)),
                    "fit_eligible_for_selection": False,
                    "reason": "insufficient_prior_selected_rows",
                }
            )
            search_breadth[str(period)] = 0
            continue

        gates = _candidate_gates(
            fit,
            feature_columns=gate_features,
            quantiles=quantiles,
            include_score_gates=include_score_gates,
        )
        baseline_fit_metrics = _metrics(
            fit,
            first_touch_clean_r=first_touch_clean_r,
            full_path_safe_r=full_path_safe_r,
            full_path_dirty_r=full_path_dirty_r,
        )
        baseline_holdout_metrics = baseline_metrics
        search_breadth[str(period)] = len(gates)
        rows_for_month: list[dict[str, Any]] = []
        for gate in gates:
            rows_for_month.append(
                _gate_row(
                    period=str(period),
                    gate=gate,
                    fit=fit,
                    holdout=holdout,
                    min_fit_rows=min_fit_rows,
                    min_fit_keep_frac=min_fit_keep_frac,
                    max_fit_keep_frac=max_fit_keep_frac,
                    objective_kwargs=objective_kwargs,
                    baseline_fit_metrics=baseline_fit_metrics,
                    baseline_holdout_metrics=baseline_holdout_metrics,
                    selection_mode=selection_mode,
                    first_touch_clean_r=first_touch_clean_r,
                    full_path_safe_r=full_path_safe_r,
                    full_path_dirty_r=full_path_dirty_r,
                )
            )
        month_candidates = pd.DataFrame(rows_for_month)
        eligible = month_candidates[month_candidates["fit_eligible_for_selection"].astype(bool)].copy()
        if eligible.empty:
            selected = month_candidates[month_candidates["gate_name"].eq("no_gate")].iloc[0]
        else:
            selected = eligible.sort_values(
                [
                    "fit_selection_score",
                    "fit_portfolio_objective",
                    "fit_mean_first_touch_net",
                    "fit_bad_full_path_mae_3r_rate",
                    "fit_rows",
                ],
                ascending=[False, False, False, True, False],
            ).iloc[0]
        selected_gate_name = str(selected["gate_name"])
        for row in rows_for_month:
            row["selected_gate"] = row["gate_name"] == selected_gate_name
            candidate_rows.append(row)
        selected_gate = next(gate for gate in gates if gate.name == selected_gate_name)
        gated_holdout = holdout.loc[_apply_gate(holdout, selected_gate)].copy()
        gated_metrics = _metrics(
            gated_holdout,
            first_touch_clean_r=first_touch_clean_r,
            full_path_safe_r=full_path_safe_r,
            full_path_dirty_r=full_path_dirty_r,
        )
        monthly_rows.append(
            {
                "period": str(period),
                "variant": "causal_gate",
                "gate_name": selected_gate_name,
                "rows": gated_metrics["rows"],
                "keep_frac": float(len(gated_holdout) / len(holdout)) if len(holdout) else float("nan"),
                **gated_metrics,
            }
        )
        weekly_rows.extend(_summarize_weekly(gated_holdout, period=str(period), variant="causal_gate"))

    monthly = pd.DataFrame(monthly_rows)
    candidates = pd.DataFrame(candidate_rows)
    weekly = pd.DataFrame(weekly_rows)
    paths = {
        "monthly": output_dir / "stage169_monthly_baseline_vs_selected.csv",
        "candidates": output_dir / "stage169_gate_candidates.csv",
        "weekly": output_dir / "stage169_weekly_baseline_vs_selected.csv",
        "manifest": output_dir / "manifest.json",
    }
    monthly.to_csv(paths["monthly"], index=False)
    candidates.to_csv(paths["candidates"], index=False)
    weekly.to_csv(paths["weekly"], index=False)
    objective = (
        "mean_first_touch_net"
        f" - {full_path_bad_weight}*bad_full_path_mae_3r_rate"
        f" - {full_path_p90_weight}*max(p90_full_path_mae_to_sl-{full_path_dirty_r},0)"
        f" - {first_touch_bad_weight}*bad_first_touch_mae_to_sl_rate"
        f" - {timeout_weight}*first_touch_timeout_rate"
    )
    manifest = {
        "scope": "stage169_stage167_selected_fullpath_gate",
        "ledger_csv": str(ledger_csv),
        "output_dir": str(output_dir),
        "feature_dir": str(feature_dir),
        "feature_list_csv": str(feature_list_csv),
        "feature_store": feature_manifest,
        "gate_feature_count": int(len(gate_features)),
        "gate_features": gate_features,
        "months": list(months),
        "quantiles": list(quantiles),
        "min_fit_rows": int(min_fit_rows),
        "min_fit_keep_frac": float(min_fit_keep_frac),
        "max_fit_keep_frac": float(max_fit_keep_frac),
        "include_score_gates": bool(include_score_gates),
        "selection_mode": str(selection_mode),
        "first_touch_clean_r": float(first_touch_clean_r),
        "full_path_safe_r": float(full_path_safe_r),
        "full_path_dirty_r": float(full_path_dirty_r),
        "objective": objective,
        "gate_search_breadth": search_breadth,
        "outputs": {key: str(value) for key, value in paths.items()},
    }
    paths["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    markdown = _write_markdown(output_dir=output_dir, monthly=monthly, candidates=candidates, weekly=weekly, manifest=manifest)
    manifest["outputs"]["markdown"] = str(markdown)
    paths["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger-csv", type=Path, default=DEFAULT_LEDGER_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--feature-list-csv", type=Path, default=DEFAULT_FEATURE_LIST_CSV)
    parser.add_argument("--max-gate-features", type=int, default=80)
    parser.add_argument("--months", default=",".join(DEFAULT_MONTHS))
    parser.add_argument("--quantiles", default="0.20,0.35,0.50,0.65,0.80")
    parser.add_argument("--min-fit-rows", type=int, default=20)
    parser.add_argument("--min-fit-keep-frac", type=float, default=0.25)
    parser.add_argument("--max-fit-keep-frac", type=float, default=0.95)
    parser.add_argument("--include-score-gates", action="store_true", default=False)
    parser.add_argument("--no-score-gates", dest="include_score_gates", action="store_false")
    parser.add_argument("--selection-mode", choices=("quality", "portfolio_delta", "sum_delta"), default="sum_delta")
    parser.add_argument("--first-touch-clean-r", type=float, default=1.0)
    parser.add_argument("--full-path-safe-r", type=float, default=3.0)
    parser.add_argument("--full-path-dirty-r", type=float, default=3.0)
    parser.add_argument("--full-path-bad-weight", type=float, default=0.010)
    parser.add_argument("--full-path-p90-weight", type=float, default=0.0015)
    parser.add_argument("--first-touch-bad-weight", type=float, default=0.004)
    parser.add_argument("--timeout-weight", type=float, default=0.004)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = run_ablation(
        ledger_csv=args.ledger_csv,
        output_dir=args.output_dir,
        feature_dir=args.feature_dir,
        feature_list_csv=args.feature_list_csv,
        max_gate_features=int(args.max_gate_features),
        months=_parse_csv(str(args.months), default=DEFAULT_MONTHS),
        quantiles=_parse_float_csv(str(args.quantiles)),
        min_fit_rows=int(args.min_fit_rows),
        min_fit_keep_frac=float(args.min_fit_keep_frac),
        max_fit_keep_frac=float(args.max_fit_keep_frac),
        include_score_gates=bool(args.include_score_gates),
        selection_mode=str(args.selection_mode),
        first_touch_clean_r=float(args.first_touch_clean_r),
        full_path_safe_r=float(args.full_path_safe_r),
        full_path_dirty_r=float(args.full_path_dirty_r),
        full_path_bad_weight=float(args.full_path_bad_weight),
        full_path_p90_weight=float(args.full_path_p90_weight),
        first_touch_bad_weight=float(args.first_touch_bad_weight),
        timeout_weight=float(args.timeout_weight),
    )
    print(json.dumps(_json_safe(manifest), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
