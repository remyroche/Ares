#!/usr/bin/env python3
"""Diagnostic source-gated evaluation of existing prediction-time scores.

This script is intentionally read/eval-only: it does not train a model, tune
policy geometry, or integrate source tags into production training. It answers
whether causal source gates improve the realized profile of an already
materialized prediction/proxy score.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_label_quality_proxy_diagnostics import (  # noqa: E402
    _effective_n,
    _json_safe,
    _path_metrics,
    _rank_top_indices,
    _safe_mean,
    _safe_numeric,
    _safe_quantile,
    _spearman,
)


DEFAULT_JOINED_SUBSET = Path(
    "data_perp/reports/source_quality_label_walkforward_ablation_v1/"
    "source_quality_clean_joined_subset_july_refresh_basegateoff_v1/"
    "source_quality_clean_joined_subset.parquet"
)
DEFAULT_OUTPUT_DIR = Path(
    "data_perp/reports/source_quality_label_walkforward_ablation_v1/"
    "source_gated_vanilla_diagnostic_july_refresh_basegateoff_v1"
)
DEFAULT_SCORE_COLS = (
    "oof_pred",
    "oof_meta_clf",
    "oof_base_clf",
    "pred_H10_pred_mean",
    "base_H10_pred_mean",
    "base_rank_pct",
)
DEFAULT_SOURCE_GATES = (
    "train_include_risk_adjusted_capture_candidate_v4",
    "train_include_compression_capture_candidate_v3",
)
DEFAULT_MONTHS = ("2026-04", "2026-05", "2026-06")
DEFAULT_TOP_FRACS = (0.03, 0.10)


@dataclass(frozen=True)
class GateSpec:
    name: str
    any_columns: tuple[str, ...] = ()
    all_columns: tuple[str, ...] = ()
    max_barrier: float | None = None
    description: str = ""


def _parse_csv(value: str | None, default: tuple[str, ...]) -> list[str]:
    if value is None or not str(value).strip():
        return list(default)
    return [part.strip() for part in str(value).split(",") if part.strip()]


def _parse_float_csv(value: str | None, default: tuple[float, ...]) -> list[float]:
    if value is None or not str(value).strip():
        return list(default)
    return [float(part.strip()) for part in str(value).split(",") if part.strip()]


def _load_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() in {".csv", ".gz"}:
        frame = pd.read_csv(path)
    else:
        frame = pd.read_parquet(path)
    required = {"__ts__", "__symbol__", "__barrier_pct__", "__mfe_ret__", "__mae_ret__", "__u_policy_net__"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} missing required joined-subset columns: {missing}")
    frame = frame.copy()
    frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True, errors="coerce")
    if frame["__ts__"].isna().any():
        raise ValueError(f"{path} contains unparsable __ts__ values")
    return frame.sort_values(["__ts__", "__symbol__"]).reset_index(drop=True)


def _bool_mask(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(False, index=frame.index)
    values = frame[column]
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False).astype(bool)
    if pd.api.types.is_numeric_dtype(values):
        return pd.to_numeric(values, errors="coerce").fillna(0.0).ne(0.0)
    return values.astype(str).str.lower().isin({"1", "true", "yes", "y"})


def _short_gate_name(column: str) -> str:
    name = str(column)
    for prefix in ("train_include_", "tag_"):
        if name.startswith(prefix):
            name = name[len(prefix) :]
    name = re.sub(r"_v\d+$", "", name)
    name = re.sub(r"_score_top\d+(_v\d+)?$", "", name)
    return name


def _available_score_columns(frame: pd.DataFrame, requested: list[str], min_non_null: int) -> list[str]:
    cols: list[str] = []
    for col in requested:
        if col not in frame.columns:
            continue
        score = _safe_numeric(frame[col])
        if int(score.notna().sum()) >= int(min_non_null) and score.nunique(dropna=True) >= 3:
            cols.append(col)
    return cols


def _build_gate_specs(
    frame: pd.DataFrame,
    *,
    gate_columns: list[str],
    dirty_excluded_column: str,
    barrier_guards: list[float],
) -> list[GateSpec]:
    specs: list[GateSpec] = [GateSpec(name="all_rows", description="No source gate.")]
    if dirty_excluded_column in frame.columns:
        specs.append(
            GateSpec(
                name="dirty_excluded",
                all_columns=(dirty_excluded_column,),
                description=f"{dirty_excluded_column} == true",
            )
        )

    available_gates = [col for col in gate_columns if col in frame.columns]
    for col in available_gates:
        base = _short_gate_name(col)
        specs.append(GateSpec(name=base, all_columns=(col,), description=f"{col} == true"))
        if dirty_excluded_column in frame.columns:
            specs.append(
                GateSpec(
                    name=f"{base}__dirty_excluded",
                    all_columns=(col, dirty_excluded_column),
                    description=f"{col} == true and {dirty_excluded_column} == true",
                )
            )
        for barrier in barrier_guards:
            suffix = f"barrier{int(round(barrier * 10000)):04d}bps"
            specs.append(
                GateSpec(
                    name=f"{base}__{suffix}",
                    all_columns=(col,),
                    max_barrier=float(barrier),
                    description=f"{col} == true and barrier <= {barrier:.4f}",
                )
            )
            if dirty_excluded_column in frame.columns:
                specs.append(
                    GateSpec(
                        name=f"{base}__dirty_excluded__{suffix}",
                        all_columns=(col, dirty_excluded_column),
                        max_barrier=float(barrier),
                        description=f"{col} == true and {dirty_excluded_column} == true and barrier <= {barrier:.4f}",
                    )
                )

    if len(available_gates) >= 2:
        specs.append(
            GateSpec(
                name="capture_candidate_union",
                any_columns=tuple(available_gates),
                description="Any configured capture-candidate source gate.",
            )
        )
        if dirty_excluded_column in frame.columns:
            specs.append(
                GateSpec(
                    name="capture_candidate_union__dirty_excluded",
                    any_columns=tuple(available_gates),
                    all_columns=(dirty_excluded_column,),
                    description="Any configured capture-candidate source gate and dirty-excluded.",
                )
            )
        for barrier in barrier_guards:
            suffix = f"barrier{int(round(barrier * 10000)):04d}bps"
            specs.append(
                GateSpec(
                    name=f"capture_candidate_union__{suffix}",
                    any_columns=tuple(available_gates),
                    max_barrier=float(barrier),
                    description=f"Any configured capture-candidate source gate and barrier <= {barrier:.4f}.",
                )
            )
            if dirty_excluded_column in frame.columns:
                specs.append(
                    GateSpec(
                        name=f"capture_candidate_union__dirty_excluded__{suffix}",
                        any_columns=tuple(available_gates),
                        all_columns=(dirty_excluded_column,),
                        max_barrier=float(barrier),
                        description=(
                            "Any configured capture-candidate source gate, dirty-excluded, "
                            f"and barrier <= {barrier:.4f}."
                        ),
                    )
                )
    # Preserve order but drop duplicate names that can occur with repeated CLI gates.
    deduped: list[GateSpec] = []
    seen: set[str] = set()
    for spec in specs:
        if spec.name in seen:
            continue
        seen.add(spec.name)
        deduped.append(spec)
    return deduped


def _gate_mask(frame: pd.DataFrame, metrics: pd.DataFrame, spec: GateSpec) -> pd.Series:
    mask = pd.Series(True, index=frame.index)
    if spec.any_columns:
        any_mask = pd.Series(False, index=frame.index)
        for col in spec.any_columns:
            any_mask = any_mask | _bool_mask(frame, col)
        mask = mask & any_mask
    for col in spec.all_columns:
        mask = mask & _bool_mask(frame, col)
    if spec.max_barrier is not None:
        mask = mask & _safe_numeric(metrics["barrier"]).le(float(spec.max_barrier))
    return mask.fillna(False)


def _period_values(frame: pd.DataFrame, period: str) -> pd.Series:
    if period == "month":
        return frame["__ts__"].dt.to_period("M").astype(str)
    if period == "week":
        return frame["__ts__"].dt.to_period("W-SUN").astype(str)
    raise ValueError(f"Unsupported period: {period}")


def _mfe_mae_ratio(metrics: pd.DataFrame) -> pd.Series:
    if metrics.empty:
        return pd.Series(dtype=float)
    return (metrics["mfe_norm"] / metrics["mae_norm"].clip(lower=0.25)).replace([np.inf, -np.inf], np.nan).clip(upper=10.0)


def _profile_row(
    *,
    frame: pd.DataFrame,
    metrics: pd.DataFrame,
    score: pd.Series | None,
    gate: GateSpec,
    period: str,
    period_rows: int,
    score_col: str,
    top_frac: float | None,
    selector: str,
) -> dict[str, Any]:
    utility = metrics["u_policy_net"] if "u_policy_net" in metrics.columns else pd.Series(dtype=float)
    ratio = _mfe_mae_ratio(metrics)
    symbols = frame.get("__symbol__", pd.Series(dtype=object))
    timestamps = frame.get("__ts__", pd.Series(dtype="datetime64[ns]"))
    score_valid = _safe_numeric(score).notna() if score is not None else pd.Series(False, index=frame.index)
    return {
        "selector": selector,
        "period": period,
        "gate": gate.name,
        "gate_description": gate.description,
        "score_col": score_col,
        "top_frac": float(top_frac) if top_frac is not None else float("nan"),
        "period_rows": int(period_rows),
        "rows": int(len(frame)),
        "coverage": float(len(frame) / period_rows) if period_rows else 0.0,
        "score_valid_rows": int(score_valid.sum()) if len(score_valid) else 0,
        "score_valid_rate": float(score_valid.mean()) if len(score_valid) else 0.0,
        "mean_score": _safe_mean(score) if score is not None else float("nan"),
        "score_ic_u": _spearman(score, utility) if score is not None else float("nan"),
        "mean_u": _safe_mean(utility),
        "median_u": _safe_quantile(utility, 0.50),
        "p25_u": _safe_quantile(utility, 0.25),
        "q10_u": _safe_quantile(utility, 0.10),
        "hit_u": _safe_mean(utility > 0.0),
        "mean_return_net": _safe_mean(metrics.get("ret_net")),
        "hit_return_net": _safe_mean(_safe_numeric(metrics.get("ret_net")) > 0.0),
        "bad_mae_1r_rate": _safe_mean(_safe_numeric(metrics.get("mae_norm")) >= 1.0),
        "bad_mae_negative_rate": _safe_mean((_safe_numeric(metrics.get("mae_norm")) >= 1.0) & (utility <= 0.0)),
        "p90_mae_norm": _safe_quantile(metrics.get("mae_norm"), 0.90),
        "mean_mae_norm": _safe_mean(metrics.get("mae_norm")),
        "mean_mfe_norm": _safe_mean(metrics.get("mfe_norm")),
        "mean_mfe_mae_ratio": _safe_mean(ratio),
        "timeout_rate": _safe_mean(metrics.get("is_timeout").astype(float) if "is_timeout" in metrics.columns else pd.Series(dtype=float)),
        "wide_barrier_25bps_rate": _safe_mean(_safe_numeric(metrics.get("barrier")) > 0.025),
        "wide_barrier_35bps_rate": _safe_mean(_safe_numeric(metrics.get("barrier")) > 0.035),
        "mean_barrier": _safe_mean(metrics.get("barrier")),
        "p90_barrier": _safe_quantile(metrics.get("barrier"), 0.90),
        "clean_row_rate": _safe_mean(
            (utility > 0.0)
            & (_safe_numeric(metrics.get("mae_norm")) <= 1.0)
            & (_safe_numeric(metrics.get("barrier")) <= 0.025)
            & (_safe_numeric(metrics.get("is_timeout")).fillna(0.0) <= 0.0)
        ),
        "bounded_row_rate": _safe_mean(
            (utility > 0.0)
            & (_safe_numeric(metrics.get("mae_norm")) <= 1.0)
            & (_safe_numeric(metrics.get("barrier")) <= 0.035)
            & (ratio >= 1.25)
            & (_safe_numeric(metrics.get("is_timeout")).fillna(0.0) <= 0.0)
        ),
        "symbol_effective_n": _effective_n(symbols),
        "unique_symbols": int(symbols.nunique(dropna=True)) if len(symbols) else 0,
        "top_symbol": str(symbols.value_counts(dropna=False).index[0]) if len(symbols) else "",
        "top_symbol_share": float(symbols.value_counts(normalize=True, dropna=False).iloc[0]) if len(symbols) else 0.0,
        "timestamp_effective_n": _effective_n(timestamps.astype(str)) if len(timestamps) else 0.0,
        "top_timestamp_share": float(timestamps.astype(str).value_counts(normalize=True, dropna=False).iloc[0]) if len(timestamps) else 0.0,
    }


def _select_top(
    *,
    frame: pd.DataFrame,
    metrics: pd.DataFrame,
    score: pd.Series,
    top_frac: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    idx = _rank_top_indices(score, top_frac)
    if len(idx) == 0:
        return frame.iloc[:0].copy(), metrics.iloc[:0].copy(), score.iloc[:0].copy()
    return frame.iloc[idx].copy(), metrics.iloc[idx].copy(), score.iloc[idx].copy()


def _add_deltas(rows: pd.DataFrame, *, baseline_gate: str = "all_rows") -> pd.DataFrame:
    if rows.empty:
        return rows
    keys = ["selector", "period", "score_col", "top_frac"]
    baseline = rows[rows["gate"].eq(baseline_gate)].copy()
    cols = [
        "mean_u",
        "hit_u",
        "q10_u",
        "bad_mae_1r_rate",
        "p90_mae_norm",
        "timeout_rate",
        "wide_barrier_25bps_rate",
        "wide_barrier_35bps_rate",
        "clean_row_rate",
        "bounded_row_rate",
        "top_symbol_share",
    ]
    baseline = baseline[keys + cols].rename(columns={col: f"all_rows_{col}" for col in cols})
    out = rows.merge(baseline, on=keys, how="left", validate="many_to_one")
    for col in cols:
        out[f"delta_{col}_vs_all_rows"] = (
            pd.to_numeric(out[col], errors="coerce") - pd.to_numeric(out[f"all_rows_{col}"], errors="coerce")
        )
    return out


def _aggregate(monthly: pd.DataFrame, *, expected_months: int) -> pd.DataFrame:
    if monthly.empty:
        return monthly
    rows: list[dict[str, Any]] = []
    group_cols = ["gate", "gate_description", "score_col", "top_frac"]
    for key, group in monthly.groupby(group_cols, dropna=False, observed=True):
        gate, description, score_col, top_frac = key
        months = int(group["period"].nunique())
        mean_u = _safe_numeric(group["mean_u"])
        delta = _safe_numeric(group.get("delta_mean_u_vs_all_rows"))
        mean_delta = _safe_mean(delta)
        mean_bad_mae = _safe_mean(group["bad_mae_1r_rate"])
        mean_baseline_bad_mae = _safe_mean(group.get("all_rows_bad_mae_1r_rate"))
        mean_timeout = _safe_mean(group["timeout_rate"])
        mean_baseline_timeout = _safe_mean(group.get("all_rows_timeout_rate"))
        mean_wide = _safe_mean(group["wide_barrier_25bps_rate"])
        mean_baseline_wide = _safe_mean(group.get("all_rows_wide_barrier_25bps_rate"))
        positive_all = (
            months >= int(expected_months)
            and int((mean_u > 0.0).sum()) >= int(expected_months)
            and math.isfinite(mean_delta)
            and mean_delta > 0.0
            and float(mean_u.min()) > 0.0
        )
        risk_not_worse = (
            math.isfinite(mean_bad_mae)
            and math.isfinite(mean_baseline_bad_mae)
            and mean_bad_mae <= mean_baseline_bad_mae
            and math.isfinite(mean_timeout)
            and math.isfinite(mean_baseline_timeout)
            and mean_timeout <= mean_baseline_timeout
            and math.isfinite(mean_wide)
            and math.isfinite(mean_baseline_wide)
            and mean_wide <= mean_baseline_wide
        )
        if gate == "all_rows":
            decision = "baseline"
        elif positive_all and risk_not_worse:
            decision = "candidate_gate_cleaner"
        elif positive_all:
            decision = "candidate_gate_risk_review"
        else:
            decision = "diagnostic_only"
        rows.append(
            {
                "decision": decision,
                "gate": gate,
                "gate_description": description,
                "score_col": score_col,
                "top_frac": float(top_frac),
                "months": months,
                "positive_months": int((mean_u > 0.0).sum()),
                "positive_delta_months_vs_all_rows": int((delta > 0.0).sum()),
                "mean_u": _safe_mean(mean_u),
                "worst_month_mean_u": float(mean_u.min()) if len(mean_u.dropna()) else float("nan"),
                "delta_mean_u_vs_all_rows": mean_delta,
                "hit_u": _safe_mean(group["hit_u"]),
                "q10_u": _safe_mean(group["q10_u"]),
                "bad_mae_1r_rate": mean_bad_mae,
                "delta_bad_mae_1r_rate_vs_all_rows": _safe_mean(group.get("delta_bad_mae_1r_rate_vs_all_rows")),
                "p90_mae_norm": _safe_mean(group["p90_mae_norm"]),
                "timeout_rate": mean_timeout,
                "delta_timeout_rate_vs_all_rows": _safe_mean(group.get("delta_timeout_rate_vs_all_rows")),
                "wide_barrier_25bps_rate": mean_wide,
                "delta_wide_barrier_25bps_rate_vs_all_rows": _safe_mean(group.get("delta_wide_barrier_25bps_rate_vs_all_rows")),
                "clean_row_rate": _safe_mean(group["clean_row_rate"]),
                "delta_clean_row_rate_vs_all_rows": _safe_mean(group.get("delta_clean_row_rate_vs_all_rows")),
                "bounded_row_rate": _safe_mean(group["bounded_row_rate"]),
                "mean_selected_rows": _safe_mean(group["rows"]),
                "min_selected_rows": int(pd.to_numeric(group["rows"], errors="coerce").min()),
                "mean_score_valid_rate": _safe_mean(group["score_valid_rate"]),
                "score_ic_u": _safe_mean(group["score_ic_u"]),
                "top_symbol_share": _safe_mean(group["top_symbol_share"]),
                "symbol_effective_n": _safe_mean(group["symbol_effective_n"]),
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values(
        ["top_frac", "decision", "delta_mean_u_vs_all_rows", "mean_u"],
        ascending=[True, True, False, False],
        na_position="last",
    )


def _evaluate(
    frame: pd.DataFrame,
    *,
    metrics: pd.DataFrame,
    score_cols: list[str],
    gates: list[GateSpec],
    months: list[str],
    top_fracs: list[float],
    min_score_rows: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    month_key = _period_values(frame, "month")
    week_key = _period_values(frame, "week")
    monthly_rows: list[dict[str, Any]] = []
    weekly_rows: list[dict[str, Any]] = []
    profile_rows: list[dict[str, Any]] = []

    def eval_period(period_value: str, period_mask: pd.Series, period_kind: str) -> None:
        period_rows = int(period_mask.sum())
        if period_rows <= 0:
            return
        for gate in gates:
            gate_mask = _gate_mask(frame, metrics, gate) & period_mask
            period_frame = frame.loc[gate_mask].reset_index(drop=True)
            period_metrics = metrics.loc[gate_mask].reset_index(drop=True)
            profile_rows.append(
                _profile_row(
                    frame=period_frame,
                    metrics=period_metrics,
                    score=None,
                    gate=gate,
                    period=period_value,
                    period_rows=period_rows,
                    score_col="",
                    top_frac=None,
                    selector=f"gate_profile_{period_kind}",
                )
            )
            for score_col in score_cols:
                period_score = _safe_numeric(period_frame[score_col]) if score_col in period_frame.columns else pd.Series(dtype=float)
                score_valid_mask = period_score.notna()
                score_frame = period_frame.loc[score_valid_mask].reset_index(drop=True)
                score_metrics = period_metrics.loc[score_valid_mask].reset_index(drop=True)
                score_values = period_score.loc[score_valid_mask].reset_index(drop=True)
                if int(len(score_values)) < int(min_score_rows):
                    continue
                for top_frac in top_fracs:
                    selected_frame, selected_metrics, selected_score = _select_top(
                        frame=score_frame,
                        metrics=score_metrics,
                        score=score_values,
                        top_frac=top_frac,
                    )
                    row = _profile_row(
                        frame=selected_frame,
                        metrics=selected_metrics,
                        score=selected_score,
                        gate=gate,
                        period=period_value,
                        period_rows=period_rows,
                        score_col=score_col,
                        top_frac=top_frac,
                        selector=f"score_topk_{period_kind}",
                    )
                    if period_kind == "month":
                        monthly_rows.append(row)
                    else:
                        weekly_rows.append(row)

    for month in months:
        eval_period(month, month_key.eq(str(month)), "month")
    for week, idx in pd.Series(np.arange(len(frame)), index=frame.index).groupby(week_key, dropna=False):
        week_mask = pd.Series(False, index=frame.index)
        week_mask.iloc[idx.to_numpy(dtype=np.int64)] = True
        week_months = set(month_key.loc[week_mask].astype(str).unique())
        if months and not week_months.intersection(set(months)):
            continue
        eval_period(str(week), week_mask, "week")

    monthly = _add_deltas(pd.DataFrame(monthly_rows))
    weekly = _add_deltas(pd.DataFrame(weekly_rows))
    profile = pd.DataFrame(profile_rows)
    return monthly, weekly, profile


def _table(frame: pd.DataFrame, cols: list[str], limit: int = 12) -> str:
    if frame.empty:
        return "_No rows._"
    view = frame[[col for col in cols if col in frame.columns]].head(limit).copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.4f}")
    return view.to_markdown(index=False)


def _write_report(output_dir: Path, aggregate: pd.DataFrame, monthly: pd.DataFrame, manifest: dict[str, Any]) -> Path:
    path = output_dir / "source_gated_vanilla_diagnostic_report.md"
    top = aggregate[aggregate["decision"].ne("baseline")].copy()
    top = top.sort_values(["decision", "top_frac", "delta_mean_u_vs_all_rows"], ascending=[True, True, False])
    lines = [
        "# Source-Gated Vanilla Diagnostic",
        "",
        "Scope: diagnostic-only evaluation of existing prediction-time score columns inside causal source gates.",
        "No model training, Optuna, policy geometry optimization, or production integration is performed.",
        "",
        "## Inputs",
        "",
        f"- Joined subset: `{manifest['joined_subset_path']}`",
        f"- Rows: `{manifest['rows']}`",
        f"- Date range: `{manifest['timestamp_min']}` to `{manifest['timestamp_max']}`",
        f"- Months: `{', '.join(manifest['months'])}`",
        f"- Score columns: `{', '.join(manifest['score_cols'])}`",
        "",
        "## Aggregate Candidates",
        "",
        _table(
            top,
            [
                "decision",
                "gate",
                "score_col",
                "top_frac",
                "months",
                "positive_months",
                "mean_u",
                "worst_month_mean_u",
                "delta_mean_u_vs_all_rows",
                "bad_mae_1r_rate",
                "timeout_rate",
                "wide_barrier_25bps_rate",
                "mean_selected_rows",
            ],
            limit=20,
        ),
        "",
        "## Baselines",
        "",
        _table(
            aggregate[aggregate["gate"].eq("all_rows")],
            [
                "gate",
                "score_col",
                "top_frac",
                "mean_u",
                "worst_month_mean_u",
                "bad_mae_1r_rate",
                "timeout_rate",
                "wide_barrier_25bps_rate",
                "mean_selected_rows",
            ],
            limit=20,
        ),
        "",
        "## Monthly Detail",
        "",
        _table(
            monthly.sort_values(["period", "top_frac", "delta_mean_u_vs_all_rows"], ascending=[True, True, False]),
            [
                "period",
                "gate",
                "score_col",
                "top_frac",
                "rows",
                "mean_u",
                "delta_mean_u_vs_all_rows",
                "bad_mae_1r_rate",
                "timeout_rate",
                "wide_barrier_25bps_rate",
                "top_symbol_share",
            ],
            limit=30,
        ),
        "",
        "## Interpretation",
        "",
        "- `candidate_gate_cleaner` means the gate beat all-rows on utility in every requested month and did not worsen bad-MAE, timeout, or 25 bps wide-barrier rates.",
        "- `candidate_gate_risk_review` means the gate beat all-rows on utility in every requested month but worsened at least one risk metric.",
        "- These are research gates only; they are not production policy decisions.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def run_report(
    *,
    joined_subset_path: Path,
    output_dir: Path,
    score_cols: list[str],
    gate_columns: list[str],
    dirty_excluded_column: str,
    barrier_guards: list[float],
    months: list[str],
    top_fracs: list[float],
    min_score_rows: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = _load_frame(joined_subset_path)
    metrics = _path_metrics(frame)
    available_scores = _available_score_columns(frame, score_cols, min_non_null=min_score_rows)
    if not available_scores:
        raise ValueError(f"No requested score columns have at least {min_score_rows} non-null rows")
    gates = _build_gate_specs(
        frame,
        gate_columns=gate_columns,
        dirty_excluded_column=dirty_excluded_column,
        barrier_guards=barrier_guards,
    )
    monthly, weekly, profile = _evaluate(
        frame,
        metrics=metrics,
        score_cols=available_scores,
        gates=gates,
        months=months,
        top_fracs=top_fracs,
        min_score_rows=min_score_rows,
    )
    aggregate = _aggregate(monthly, expected_months=len(months))

    paths = {
        "monthly": output_dir / "source_gated_vanilla_monthly.csv",
        "weekly": output_dir / "source_gated_vanilla_weekly.csv",
        "gate_profile": output_dir / "source_gated_vanilla_gate_profile_by_period.csv",
        "aggregate": output_dir / "source_gated_vanilla_aggregate.csv",
        "manifest": output_dir / "manifest.json",
    }
    monthly.to_csv(paths["monthly"], index=False)
    weekly.to_csv(paths["weekly"], index=False)
    profile.to_csv(paths["gate_profile"], index=False)
    aggregate.to_csv(paths["aggregate"], index=False)

    manifest: dict[str, Any] = {
        "scope": "diagnostic_source_gated_vanilla_existing_scores",
        "joined_subset_path": str(joined_subset_path),
        "output_dir": str(output_dir),
        "rows": int(len(frame)),
        "timestamp_min": frame["__ts__"].min().isoformat(),
        "timestamp_max": frame["__ts__"].max().isoformat(),
        "symbols": int(frame["__symbol__"].nunique(dropna=True)),
        "months": months,
        "top_fracs": [float(v) for v in top_fracs],
        "score_cols": available_scores,
        "requested_score_cols": score_cols,
        "gate_columns": gate_columns,
        "dirty_excluded_column": dirty_excluded_column if dirty_excluded_column in frame.columns else "",
        "barrier_guards": [float(v) for v in barrier_guards],
        "gate_specs": [
            {
                "name": gate.name,
                "any_columns": list(gate.any_columns),
                "all_columns": list(gate.all_columns),
                "max_barrier": gate.max_barrier,
                "description": gate.description,
            }
            for gate in gates
        ],
        "outputs": {key: str(value) for key, value in paths.items()},
        "metric_report": {
            "utility_source": metrics.attrs.get("utility_source"),
            "mae_encoding": metrics.attrs.get("mae_encoding"),
            "selection": "descending score within each month/week and gate",
            "baseline_delta": "same score_col/top_frac on all_rows",
        },
    }
    report_path = _write_report(output_dir, aggregate, monthly, manifest)
    manifest["outputs"]["report"] = str(report_path)
    paths["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--joined-subset-path", type=Path, default=DEFAULT_JOINED_SUBSET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--score-cols", default=",".join(DEFAULT_SCORE_COLS))
    parser.add_argument("--source-gates", default=",".join(DEFAULT_SOURCE_GATES))
    parser.add_argument("--dirty-excluded-column", default="train_include_dirty_excluded_v0")
    parser.add_argument("--barrier-guards", default="0.025")
    parser.add_argument("--months", default=",".join(DEFAULT_MONTHS))
    parser.add_argument("--top-fracs", default=",".join(str(v) for v in DEFAULT_TOP_FRACS))
    parser.add_argument("--min-score-rows", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = run_report(
        joined_subset_path=args.joined_subset_path,
        output_dir=args.output_dir,
        score_cols=_parse_csv(args.score_cols, DEFAULT_SCORE_COLS),
        gate_columns=_parse_csv(args.source_gates, DEFAULT_SOURCE_GATES),
        dirty_excluded_column=str(args.dirty_excluded_column),
        barrier_guards=_parse_float_csv(args.barrier_guards, (0.025,)),
        months=_parse_csv(args.months, DEFAULT_MONTHS),
        top_fracs=_parse_float_csv(args.top_fracs, DEFAULT_TOP_FRACS),
        min_score_rows=int(args.min_score_rows),
    )
    print(json.dumps(_json_safe(manifest), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
