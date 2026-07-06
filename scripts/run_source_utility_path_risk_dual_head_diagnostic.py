#!/usr/bin/env python3
"""Dual-head utility plus path-risk diagnostics for source-aware labels.

This script tests the next diagnostic question after utility/source gates:
can a separate path-risk head reduce bad-MAE, timeout, and wide-barrier rows
without destroying the OOS utility signal?

It is diagnostic-only. Utility and path-risk targets use realized outcomes on
training months, then both heads score future months from causal features only.
No production training artifact is modified.
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
from scripts.run_source_utility_label_rework_diagnostic import (  # noqa: E402
    DEFAULT_TOP_FRACS,
    _build_target,
    _safe_numeric,
)
from scripts.run_source_utility_risk_gate_diagnostic import (  # noqa: E402
    _assert_gate_columns_causal,
    _gate_mask,
    _gate_specs_by_name,
    _label_specs_by_name,
)


DEFAULT_OUTPUT_DIR = Path(
    "data_perp/reports/source_quality_label_walkforward_ablation_v1_sideaware_20260702/utility_path_risk_dual_head"
)
DEFAULT_LABELS = ("utility_linear_source_q80_v1",)
DEFAULT_RISK_TARGETS = (
    "path_risk_composite_v1",
    "path_failure_strict_v1",
)
DEFAULT_FEATURE_SETS = ("base_plus_source",)
DEFAULT_SOURCE_BUCKETS = ("all_rows", "risk_adjusted_capture_candidate")
DEFAULT_CAUSAL_GATES = ("no_gate", "low_barrier_pressure_q50")
DEFAULT_SELECTIONS = (
    "utility_only",
    "utility_minus_risk_0p50",
    "utility_minus_risk_1p00",
    "utility_risk_q50",
    "utility_risk_q35",
    "utility_risk_rank_q35",
)


@dataclass(frozen=True)
class RiskTargetSpec:
    name: str
    bad_mae_weight: float
    timeout_weight: float
    wide25_weight: float
    wide35_weight: float
    mae_severity_weight: float
    barrier_severity_weight: float
    timeout_negative_only: bool = False


RISK_TARGET_SPECS = (
    RiskTargetSpec(
        name="path_risk_composite_v1",
        bad_mae_weight=0.38,
        timeout_weight=0.16,
        wide25_weight=0.16,
        wide35_weight=0.08,
        mae_severity_weight=0.16,
        barrier_severity_weight=0.06,
    ),
    RiskTargetSpec(
        name="path_failure_strict_v1",
        bad_mae_weight=0.55,
        timeout_weight=0.20,
        wide25_weight=0.15,
        wide35_weight=0.05,
        mae_severity_weight=0.05,
        barrier_severity_weight=0.00,
        timeout_negative_only=True,
    ),
    RiskTargetSpec(
        name="barrier_timeout_risk_v1",
        bad_mae_weight=0.25,
        timeout_weight=0.25,
        wide25_weight=0.25,
        wide35_weight=0.10,
        mae_severity_weight=0.05,
        barrier_severity_weight=0.10,
    ),
)


@dataclass(frozen=True)
class SelectionSpec:
    name: str
    mode: str
    penalty: float = 0.0
    risk_q: float | None = None


SELECTION_SPECS = (
    SelectionSpec("utility_only", "utility"),
    SelectionSpec("utility_minus_risk_0p50", "penalty", penalty=0.50),
    SelectionSpec("utility_minus_risk_1p00", "penalty", penalty=1.00),
    SelectionSpec("utility_minus_risk_1p50", "penalty", penalty=1.50),
    SelectionSpec("utility_risk_q50", "risk_gate", risk_q=0.50),
    SelectionSpec("utility_risk_q35", "risk_gate", risk_q=0.35),
    SelectionSpec("utility_risk_q25", "risk_gate", risk_q=0.25),
    SelectionSpec("utility_risk_rank_q50", "risk_rank_gate", risk_q=0.50),
    SelectionSpec("utility_risk_rank_q35", "risk_rank_gate", risk_q=0.35),
    SelectionSpec("utility_risk_rank_q25", "risk_rank_gate", risk_q=0.25),
)


def _risk_targets_by_name(names: list[str]) -> list[RiskTargetSpec]:
    available = {spec.name: spec for spec in RISK_TARGET_SPECS}
    missing = sorted(set(names) - set(available))
    if missing:
        raise ValueError(f"Unknown risk target(s): {missing}; available={sorted(available)}")
    return [available[name] for name in names]


def _selection_specs_by_name(names: list[str]) -> list[SelectionSpec]:
    available = {spec.name: spec for spec in SELECTION_SPECS}
    missing = sorted(set(names) - set(available))
    if missing:
        raise ValueError(f"Unknown selection(s): {missing}; available={sorted(available)}")
    return [available[name] for name in names]


def _bool_series(frame: pd.DataFrame, col: str) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(False, index=frame.index)
    values = frame[col]
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False).astype(bool)
    lowered = values.astype(str).str.lower()
    return lowered.isin({"1", "true", "t", "yes", "y"})


def _source_bucket_mask(frame: pd.DataFrame, source_bucket: str) -> pd.Series:
    if source_bucket == "all_rows":
        return pd.Series(True, index=frame.index)
    if "primary_source_tag" in frame.columns:
        primary = frame["primary_source_tag"].fillna("").astype(str).eq(str(source_bucket))
        if bool(primary.any()):
            return primary
    tag_col = f"tag_{source_bucket}"
    if tag_col in frame.columns:
        return _bool_series(frame, tag_col)
    return pd.Series(False, index=frame.index)


def _rank_top_fraction_indices(score: pd.Series, top_frac: float) -> np.ndarray:
    score_s = _safe_numeric(score).reset_index(drop=True)
    valid = score_s.notna().to_numpy()
    if not bool(valid.any()):
        return np.array([], dtype=np.int64)
    valid_idx = np.flatnonzero(valid)
    k = min(max(1, int(math.ceil(float(top_frac) * len(valid_idx)))), len(valid_idx))
    order = np.argsort(-score_s.iloc[valid_idx].to_numpy(dtype=np.float64), kind="mergesort")
    return valid_idx[order[:k]].astype(np.int64, copy=False)


def _build_path_risk_target(metrics: pd.DataFrame, spec: RiskTargetSpec) -> tuple[pd.DataFrame, pd.Series]:
    utility = _safe_numeric(metrics["u_policy_net"])
    mae_norm = _safe_numeric(metrics["mae_norm"]).fillna(0.0)
    barrier = _safe_numeric(metrics["barrier"]).fillna(0.0)
    timeout = metrics["is_timeout"].astype(float).fillna(0.0)
    if spec.timeout_negative_only:
        timeout_component = timeout.where(utility.le(0.0), 0.0)
    else:
        timeout_component = timeout
    bad_mae = mae_norm.ge(1.0).astype(float)
    wide25 = barrier.gt(0.025).astype(float)
    wide35 = barrier.gt(0.035).astype(float)
    mae_severity = (mae_norm / 4.0).clip(0.0, 1.0)
    barrier_severity = (barrier / 0.050).clip(0.0, 1.0)
    risk = (
        spec.bad_mae_weight * bad_mae
        + spec.timeout_weight * timeout_component
        + spec.wide25_weight * wide25
        + spec.wide35_weight * wide35
        + spec.mae_severity_weight * mae_severity
        + spec.barrier_severity_weight * barrier_severity
    )
    denom = (
        spec.bad_mae_weight
        + spec.timeout_weight
        + spec.wide25_weight
        + spec.wide35_weight
        + spec.mae_severity_weight
        + spec.barrier_severity_weight
    )
    risk_soft = (risk / max(denom, 1e-8)).clip(0.0, 1.0)
    risk_hard = ((mae_norm >= 1.0) | (barrier > 0.025) | (timeout_component > 0.5)).astype(float)
    weights = (0.50 + 1.50 * risk_soft).clip(0.50, 2.00).astype(np.float32)
    target = pd.DataFrame({"target_soft": risk_soft, "target_hard": risk_hard}, index=metrics.index)
    return target, weights


def _week_start(ts: pd.Series) -> pd.Series:
    return (
        pd.to_datetime(ts, utc=True, errors="coerce")
        .dt.tz_convert(None)
        .dt.to_period("W-SUN")
        .apply(lambda value: value.start_time.date().isoformat() if pd.notna(value) else "")
    )


def _path_summary(metrics: pd.DataFrame) -> dict[str, Any]:
    if metrics.empty:
        return {
            "rows": 0,
            "mean_u": float("nan"),
            "median_u": float("nan"),
            "q10_u": float("nan"),
            "hit_u": float("nan"),
            "bad_mae_1r_rate": float("nan"),
            "p90_mae_norm": float("nan"),
            "timeout_rate": float("nan"),
            "wide_barrier_25bps_rate": float("nan"),
            "wide_barrier_35bps_rate": float("nan"),
        }
    return {
        "rows": int(len(metrics)),
        "mean_u": _safe_mean(metrics["u_policy_net"]),
        "median_u": _safe_quantile(metrics["u_policy_net"], 0.50),
        "q10_u": _safe_quantile(metrics["u_policy_net"], 0.10),
        "hit_u": _safe_mean(metrics["u_policy_net"] > 0.0),
        "bad_mae_1r_rate": _safe_mean(metrics["mae_norm"] >= 1.0),
        "p90_mae_norm": _safe_quantile(metrics["mae_norm"], 0.90),
        "timeout_rate": _safe_mean(metrics["is_timeout"].astype(float)),
        "wide_barrier_25bps_rate": _safe_mean(metrics["barrier"] > 0.025),
        "wide_barrier_35bps_rate": _safe_mean(metrics["barrier"] > 0.035),
    }


def _selected_frame(
    *,
    frame: pd.DataFrame,
    metrics: pd.DataFrame,
    utility_target: pd.DataFrame,
    risk_target: pd.DataFrame,
    utility_pred: pd.Series,
    risk_pred: pd.Series,
    selected_idx: np.ndarray,
    context: dict[str, Any],
) -> pd.DataFrame:
    if not len(selected_idx):
        return pd.DataFrame()
    ledger_cols = ["__ts__", "__symbol__"]
    for col in ("side", "side_name", "__side__", "timeframe", "candidate_id", "primary_source_tag"):
        if col in frame.columns:
            ledger_cols.append(col)
    selected = frame.iloc[selected_idx][ledger_cols].copy()
    if "side" not in selected.columns and "side" in metrics.columns:
        selected["side"] = metrics["side"].iloc[selected_idx].to_numpy(dtype=np.int8, copy=False)
    if "side_name" not in selected.columns and "side" in selected.columns:
        selected["side_name"] = np.where(_safe_numeric(selected["side"]) < 0.0, "short", "long")
    selected.insert(0, "candidate", context["candidate"])
    for key, value in context.items():
        if key == "candidate":
            continue
        selected[key] = value
    selected["week_start"] = _week_start(selected["__ts__"])
    selected["utility_pred"] = utility_pred.iloc[selected_idx].to_numpy(dtype=np.float32, copy=False)
    selected["risk_pred"] = risk_pred.iloc[selected_idx].to_numpy(dtype=np.float32, copy=False)
    selected["utility_target_soft"] = utility_target["target_soft"].iloc[selected_idx].to_numpy()
    selected["risk_target_soft"] = risk_target["target_soft"].iloc[selected_idx].to_numpy()
    selected["u_policy_net"] = metrics["u_policy_net"].iloc[selected_idx].to_numpy()
    selected["mae_norm"] = metrics["mae_norm"].iloc[selected_idx].to_numpy()
    selected["barrier"] = metrics["barrier"].iloc[selected_idx].to_numpy()
    selected["is_timeout"] = metrics["is_timeout"].iloc[selected_idx].to_numpy()
    return selected


def _select_indices(
    *,
    utility_pred: pd.Series,
    risk_pred: pd.Series,
    risk_threshold: float | None,
    selection: SelectionSpec,
    top_frac: float,
) -> np.ndarray:
    if selection.mode == "utility":
        score = utility_pred
        eligible = utility_pred.notna()
    elif selection.mode == "penalty":
        score = utility_pred - float(selection.penalty) * risk_pred
        eligible = score.notna()
    elif selection.mode == "risk_gate":
        threshold = risk_threshold
        if threshold is None or not math.isfinite(float(threshold)):
            threshold = float(risk_pred.dropna().quantile(float(selection.risk_q or 0.50)))
        eligible = utility_pred.notna() & risk_pred.le(float(threshold))
        score = utility_pred.where(eligible)
    elif selection.mode == "risk_rank_gate":
        threshold = float(risk_pred.dropna().quantile(float(selection.risk_q or 0.50)))
        eligible = utility_pred.notna() & risk_pred.le(float(threshold))
        score = utility_pred.where(eligible)
    else:
        raise ValueError(f"Unsupported selection mode: {selection.mode}")
    if int(eligible.sum()) == 0:
        return np.array([], dtype=np.int64)
    return _rank_top_fraction_indices(score, top_frac)


def _weekly_summary(selected: pd.DataFrame) -> pd.DataFrame:
    if selected.empty:
        return pd.DataFrame()
    group_cols = [
        "candidate",
        "period",
        "week_start",
        "label",
        "risk_target",
        "feature_set",
        "source_bucket",
        "causal_gate",
        "selection",
        "top_frac",
    ]
    rows: list[dict[str, Any]] = []
    for key, group in selected.groupby(group_cols, dropna=False, observed=True):
        context = dict(zip(group_cols, key, strict=False))
        metrics = pd.DataFrame(
            {
                "u_policy_net": group["u_policy_net"],
                "mae_norm": group["mae_norm"],
                "barrier": group["barrier"],
                "is_timeout": group["is_timeout"],
            }
        )
        side = _safe_numeric(group["side"]) if "side" in group.columns else pd.Series(dtype=float)
        side_name = side.map(lambda value: "short" if value < 0.0 else "long")
        side_top_share = (
            float(side_name.value_counts(normalize=True).iloc[0]) if len(side_name) else float("nan")
        )
        rows.append(
            {
                **context,
                **_path_summary(metrics),
                "top_symbol_share": (
                    float(group["__symbol__"].value_counts(normalize=True).iloc[0]) if len(group) else float("nan")
                ),
                "unique_symbols": int(group["__symbol__"].nunique()),
                "long_share": _safe_mean(side > 0.0) if len(side) else float("nan"),
                "short_share": _safe_mean(side < 0.0) if len(side) else float("nan"),
                "side_top_share": side_top_share,
                "mean_utility_pred": _safe_mean(group["utility_pred"]),
                "mean_risk_pred": _safe_mean(group["risk_pred"]),
                "mean_risk_target_soft": _safe_mean(group["risk_target_soft"]),
            }
        )
    return pd.DataFrame(rows).sort_values(["candidate", "period", "week_start"], kind="mergesort")


def _add_utility_only_deltas(monthly: pd.DataFrame) -> pd.DataFrame:
    if monthly.empty:
        return monthly
    keys = ["period", "label", "risk_target", "feature_set", "source_bucket", "causal_gate", "top_frac"]
    baseline_cols = [
        "selected_rows",
        "mean_u",
        "hit_u",
        "q10_u",
        "bad_mae_1r_rate",
        "p90_mae_norm",
        "timeout_rate",
        "wide_barrier_25bps_rate",
    ]
    base = monthly[monthly["selection"].eq("utility_only")][keys + baseline_cols].rename(
        columns={col: f"utility_only_{col}" for col in baseline_cols}
    )
    out = monthly.merge(base, on=keys, how="left", validate="many_to_one")
    for col in baseline_cols:
        out[f"delta_{col}_vs_utility_only"] = _safe_numeric(out[col]) - _safe_numeric(out[f"utility_only_{col}"])
    return out


def _aggregate(monthly: pd.DataFrame, weekly: pd.DataFrame, *, expected_months: int) -> pd.DataFrame:
    if monthly.empty:
        return pd.DataFrame()
    group_cols = [
        "label",
        "risk_target",
        "feature_set",
        "source_bucket",
        "causal_gate",
        "selection",
        "top_frac",
    ]
    weekly_map: dict[tuple[Any, ...], pd.DataFrame] = {}
    if not weekly.empty:
        for key, group in weekly.groupby(group_cols, dropna=False, observed=True):
            weekly_map[key] = group
    rows: list[dict[str, Any]] = []
    for key, group in monthly.groupby(group_cols, dropna=False, observed=True):
        label, risk_target, feature_set, source_bucket, causal_gate, selection, top_frac = key
        month_count = int(group["period"].nunique())
        positive_months = int((_safe_numeric(group["mean_u"]) > 0.0).sum())
        ic_u_positive_months = int((_safe_numeric(group["utility_score_ic_u_scope"]) > 0.0).sum())
        risk_ic_positive_months = int((_safe_numeric(group["risk_score_ic_risk_scope"]) > 0.0).sum())
        min_selected = _safe_quantile(group["selected_rows"], 0.0)
        mean_bad = _safe_mean(group["bad_mae_1r_rate"])
        mean_timeout = _safe_mean(group["timeout_rate"])
        mean_wide = _safe_mean(group["wide_barrier_25bps_rate"])
        weekly_group = weekly_map.get(key, pd.DataFrame())
        weeks = int(len(weekly_group))
        positive_weeks = int((_safe_numeric(weekly_group.get("mean_u", pd.Series(dtype=float))) > 0.0).sum())
        q25_week_u = _safe_quantile(weekly_group.get("mean_u", pd.Series(dtype=float)), 0.25)
        worst_week_u = _safe_quantile(weekly_group.get("mean_u", pd.Series(dtype=float)), 0.0)
        mean_week_bad = _safe_mean(weekly_group.get("bad_mae_1r_rate", pd.Series(dtype=float)))
        max_week_bad = _safe_quantile(weekly_group.get("bad_mae_1r_rate", pd.Series(dtype=float)), 1.0)
        max_top_symbol_share = _safe_quantile(weekly_group.get("top_symbol_share", pd.Series(dtype=float)), 1.0)
        max_week_side_top_share = _safe_quantile(weekly_group.get("side_top_share", pd.Series(dtype=float)), 1.0)
        monthly_ok = (
            month_count >= int(expected_months)
            and positive_months >= int(expected_months)
            and ic_u_positive_months >= int(expected_months)
            and risk_ic_positive_months >= int(expected_months)
            and _safe_quantile(group["mean_u"], 0.0) > 0.0
            and math.isfinite(min_selected)
            and min_selected >= 25.0
        )
        path_ok = (
            math.isfinite(mean_bad)
            and mean_bad <= 0.50
            and math.isfinite(mean_timeout)
            and mean_timeout <= 0.18
            and math.isfinite(mean_wide)
            and mean_wide <= 0.15
        )
        weekly_ok = (
            weeks >= 10
            and positive_weeks >= math.ceil(0.75 * weeks)
            and math.isfinite(q25_week_u)
            and q25_week_u > 0.0
            and math.isfinite(max_top_symbol_share)
            and max_top_symbol_share <= 0.75
        )
        risk_reduced = (
            _safe_mean(group.get("delta_bad_mae_1r_rate_vs_utility_only", pd.Series(dtype=float))) < -0.05
            or _safe_mean(group.get("delta_timeout_rate_vs_utility_only", pd.Series(dtype=float))) < -0.03
            or _safe_mean(group.get("delta_wide_barrier_25bps_rate_vs_utility_only", pd.Series(dtype=float))) < -0.03
        )
        if monthly_ok and path_ok and weekly_ok:
            decision = "candidate_for_label_ablation"
        elif monthly_ok and path_ok and not weekly_ok:
            decision = "monthly_positive_weekly_unstable"
        elif monthly_ok and not path_ok:
            decision = "monthly_positive_path_limits_fail"
        elif risk_reduced:
            decision = "risk_reduced_not_enough"
        else:
            decision = "diagnostic_only"
        rows.append(
            {
                "decision": decision,
                "label": label,
                "risk_target": risk_target,
                "feature_set": feature_set,
                "source_bucket": source_bucket,
                "causal_gate": causal_gate,
                "selection": selection,
                "top_frac": float(top_frac),
                "months": month_count,
                "positive_months": positive_months,
                "utility_ic_u_positive_months": ic_u_positive_months,
                "risk_ic_positive_months": risk_ic_positive_months,
                "mean_u": _safe_mean(group["mean_u"]),
                "worst_month_u": _safe_quantile(group["mean_u"], 0.0),
                "q10_u": _safe_mean(group["q10_u"]),
                "hit_u": _safe_mean(group["hit_u"]),
                "bad_mae_1r_rate": mean_bad,
                "p90_mae_norm": _safe_mean(group["p90_mae_norm"]),
                "timeout_rate": mean_timeout,
                "wide_barrier_25bps_rate": mean_wide,
                "mean_selected_rows": _safe_mean(group["selected_rows"]),
                "min_selected_rows": min_selected,
                "utility_score_ic_u_scope": _safe_mean(group["utility_score_ic_u_scope"]),
                "risk_score_ic_risk_scope": _safe_mean(group["risk_score_ic_risk_scope"]),
                "delta_mean_u_vs_utility_only": _safe_mean(group.get("delta_mean_u_vs_utility_only")),
                "delta_bad_mae_1r_rate_vs_utility_only": _safe_mean(
                    group.get("delta_bad_mae_1r_rate_vs_utility_only")
                ),
                "delta_timeout_rate_vs_utility_only": _safe_mean(group.get("delta_timeout_rate_vs_utility_only")),
                "delta_wide_barrier_25bps_rate_vs_utility_only": _safe_mean(
                    group.get("delta_wide_barrier_25bps_rate_vs_utility_only")
                ),
                "weeks": weeks,
                "positive_weeks": positive_weeks,
                "q25_week_u": q25_week_u,
                "worst_week_u": worst_week_u,
                "mean_week_bad_mae_1r_rate": mean_week_bad,
                "max_week_bad_mae_1r_rate": max_week_bad,
                "max_top_symbol_share": max_top_symbol_share,
                "max_week_side_top_share": max_week_side_top_share,
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values(
        ["decision", "mean_u", "q25_week_u", "bad_mae_1r_rate"],
        ascending=[True, False, False, True],
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


def _write_report(output_dir: Path, aggregate: pd.DataFrame, weekly: pd.DataFrame, manifest: dict[str, Any]) -> Path:
    path = output_dir / "source_utility_path_risk_dual_head_report.md"
    cols = [
        "decision",
        "label",
        "risk_target",
        "feature_set",
        "source_bucket",
        "causal_gate",
        "selection",
        "top_frac",
        "months",
        "positive_months",
        "utility_ic_u_positive_months",
        "risk_ic_positive_months",
        "mean_u",
        "worst_month_u",
        "bad_mae_1r_rate",
        "timeout_rate",
        "wide_barrier_25bps_rate",
        "mean_selected_rows",
        "min_selected_rows",
        "delta_mean_u_vs_utility_only",
        "delta_bad_mae_1r_rate_vs_utility_only",
        "delta_timeout_rate_vs_utility_only",
        "delta_wide_barrier_25bps_rate_vs_utility_only",
        "weeks",
        "positive_weeks",
        "q25_week_u",
        "worst_week_u",
        "max_top_symbol_share",
    ]
    candidate = aggregate[aggregate["decision"].eq("candidate_for_label_ablation")]
    unstable = aggregate[aggregate["decision"].eq("monthly_positive_weekly_unstable")]
    path_fail = aggregate[aggregate["decision"].eq("monthly_positive_path_limits_fail")]
    risk_reduced = aggregate[aggregate["decision"].eq("risk_reduced_not_enough")]
    top = aggregate[aggregate["top_frac"].isin([0.01, 0.03, 0.10])].sort_values(
        ["mean_u", "q25_week_u"], ascending=[False, False]
    )
    weekly_cols = [
        "candidate",
        "period",
        "week_start",
        "rows",
        "mean_u",
        "q10_u",
        "hit_u",
        "bad_mae_1r_rate",
        "timeout_rate",
        "wide_barrier_25bps_rate",
        "top_symbol_share",
        "unique_symbols",
    ]
    lines = [
        "# Source Utility Path-Risk Dual-Head Diagnostic",
        "",
        "Scope: utility prediction plus separate path-risk prediction. Training uses prior months only.",
        "",
        f"Rows joined to label ledger: `{manifest['rows']}`",
        f"Utility source: `{manifest.get('utility_source', '')}`",
        f"Months: `{', '.join(manifest['months'])}`",
        f"Labels: `{', '.join(manifest['labels'])}`",
        f"Risk targets: `{', '.join(manifest['risk_targets'])}`",
        "",
        "## Candidates",
        "",
        _table(candidate, cols, limit=80),
        "",
        "## Monthly Positive But Weekly Unstable",
        "",
        _table(unstable, cols, limit=80),
        "",
        "## Monthly Positive But Path Limits Fail",
        "",
        _table(path_fail, cols, limit=80),
        "",
        "## Risk Reduced But Not Enough",
        "",
        _table(risk_reduced, cols, limit=80),
        "",
        "## Best Rows By Mean Utility",
        "",
        _table(top, cols, limit=120),
        "",
        "## Worst Weekly Rows",
        "",
        _table(weekly.sort_values("mean_u"), weekly_cols, limit=80),
        "",
        "## Outputs",
        "",
        f"- Monthly: `{manifest['outputs']['monthly']}`",
        f"- Weekly: `{manifest['outputs']['weekly']}`",
        f"- Aggregate: `{manifest['outputs']['aggregate']}`",
        f"- Selected rows: `{manifest['outputs']['selected_rows_parquet']}`",
        f"- Manifest: `{manifest['outputs']['manifest']}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _score_month(
    *,
    frame: pd.DataFrame,
    metrics: pd.DataFrame,
    month_period: pd.Series,
    month: str,
    labels: dict[str, Any],
    risk_targets: list[RiskTargetSpec],
    gates: dict[str, Any],
    feature_map: dict[str, list[str]],
    labels_requested: list[str],
    feature_sets: list[str],
    source_buckets: list[str],
    causal_gates: list[str],
    selections: list[SelectionSpec],
    top_fracs: list[float],
    seeds: list[int],
    train_lookback_months: int | None,
    min_train_rows: int,
    min_valid_rows: int,
    min_scope_rows: int,
) -> tuple[list[dict[str, Any]], list[pd.DataFrame], list[dict[str, Any]]]:
    valid_mask = month_period.eq(month)
    train_mask = month_period < month
    if train_lookback_months is not None and int(train_lookback_months) > 0:
        prior_months = sorted(month_period[train_mask].dropna().unique())
        keep = set(prior_months[-int(train_lookback_months) :])
        train_mask = train_mask & month_period.isin(keep)
    if int(valid_mask.sum()) < int(min_valid_rows):
        return [], [], [{"period": month, "skipped": True, "reason": "too_few_valid_rows"}]

    rows: list[dict[str, Any]] = []
    selected_frames: list[pd.DataFrame] = []
    diagnostics: list[dict[str, Any]] = []
    risk_target_cache = {spec.name: _build_path_risk_target(metrics, spec) for spec in risk_targets}
    gate_cache = {gate_name: _gate_mask(frame, train_mask, gates[gate_name]) for gate_name in causal_gates}
    for label_name in labels_requested:
        utility_target, utility_weights, _label_report = _build_target(
            frame=frame,
            metrics=metrics,
            train_mask=train_mask,
            valid_mask=valid_mask,
            spec=labels[label_name],
        )
        train_utility_mask = train_mask & utility_target["target_soft"].notna() & utility_weights.gt(0.0)
        if int(train_utility_mask.sum()) < int(min_train_rows):
            diagnostics.append(
                {
                    "period": month,
                    "label": label_name,
                    "skipped": True,
                    "reason": "too_few_utility_train_rows",
                    "train_rows": int(train_utility_mask.sum()),
                }
            )
            continue
        for risk_spec in risk_targets:
            risk_target, risk_weights = risk_target_cache[risk_spec.name]
            train_risk_mask = train_mask & risk_target["target_soft"].notna() & risk_weights.gt(0.0)
            train_model_mask = train_utility_mask & train_risk_mask
            if int(train_model_mask.sum()) < int(min_train_rows):
                diagnostics.append(
                    {
                        "period": month,
                        "label": label_name,
                        "risk_target": risk_spec.name,
                        "skipped": True,
                        "reason": "too_few_joint_train_rows",
                        "train_rows": int(train_model_mask.sum()),
                    }
                )
                continue
            risk_thresholds = {
                0.25: float(risk_target.loc[train_model_mask, "target_soft"].quantile(0.25)),
                0.35: float(risk_target.loc[train_model_mask, "target_soft"].quantile(0.35)),
                0.50: float(risk_target.loc[train_model_mask, "target_soft"].quantile(0.50)),
            }
            for feature_set in feature_sets:
                features = feature_map.get(feature_set)
                if not features:
                    continue
                x_train, x_valid = _month_model_frame(
                    frame,
                    train_mask=train_model_mask,
                    valid_mask=valid_mask,
                    features=features,
                )
                utility_matrix = np.vstack(
                    [
                        _fit_predict(
                            x_train=x_train,
                            y_train=utility_target.loc[train_model_mask, "target_soft"],
                            w_train=utility_weights.loc[train_model_mask],
                            x_valid=x_valid,
                            seed=seed,
                        )
                        for seed in seeds
                    ]
                )
                risk_matrix = np.vstack(
                    [
                        _fit_predict(
                            x_train=x_train,
                            y_train=risk_target.loc[train_model_mask, "target_soft"],
                            w_train=risk_weights.loc[train_model_mask],
                            x_valid=x_valid,
                            seed=seed + 10007,
                        )
                        for seed in seeds
                    ]
                )
                utility_pred = pd.Series(np.nan, index=frame.index, dtype=np.float32)
                risk_pred = pd.Series(np.nan, index=frame.index, dtype=np.float32)
                utility_pred.loc[valid_mask] = np.mean(utility_matrix, axis=0).astype(np.float32)
                risk_pred.loc[valid_mask] = np.mean(risk_matrix, axis=0).astype(np.float32)
                for source_bucket in source_buckets:
                    bucket_mask = valid_mask & _source_bucket_mask(frame, source_bucket)
                    for causal_gate in causal_gates:
                        gate_mask, gate_report = gate_cache[causal_gate]
                        scope_mask = bucket_mask & gate_mask
                        scope_rows = int(scope_mask.sum())
                        if scope_rows < int(min_scope_rows):
                            continue
                        scope_idx = np.flatnonzero(scope_mask.to_numpy())
                        scope_frame = frame.iloc[scope_idx].reset_index(drop=True)
                        scope_metrics = metrics.iloc[scope_idx].reset_index(drop=True)
                        scope_utility_target = utility_target.iloc[scope_idx].reset_index(drop=True)
                        scope_risk_target = risk_target.iloc[scope_idx].reset_index(drop=True)
                        scope_utility_pred = utility_pred.iloc[scope_idx].reset_index(drop=True)
                        scope_risk_pred = risk_pred.iloc[scope_idx].reset_index(drop=True)
                        scope_diag = {
                            "period": month,
                            "label": label_name,
                            "risk_target": risk_spec.name,
                            "feature_set": feature_set,
                            "source_bucket": source_bucket,
                            "causal_gate": causal_gate,
                            "train_rows": int(train_model_mask.sum()),
                            "valid_rows": int(valid_mask.sum()),
                            "scope_rows": scope_rows,
                            "gate_missing_columns": ",".join(gate_report.get("missing_gate_columns", [])),
                            "gate_thresholds_json": json.dumps(
                                _json_safe(gate_report.get("thresholds", {})), sort_keys=True
                            ),
                            "utility_score_ic_u_scope": _spearman(
                                scope_utility_pred, scope_metrics["u_policy_net"]
                            ),
                            "utility_score_ic_label_scope": _spearman(
                                scope_utility_pred, scope_utility_target["target_soft"]
                            ),
                            "risk_score_ic_risk_scope": _spearman(
                                scope_risk_pred, scope_risk_target["target_soft"]
                            ),
                            "risk_score_ic_badmae_scope": _spearman(
                                scope_risk_pred, scope_metrics["mae_norm"] >= 1.0
                            ),
                            "risk_score_ic_timeout_scope": _spearman(
                                scope_risk_pred, scope_metrics["is_timeout"].astype(float)
                            ),
                            "risk_score_ic_wide25_scope": _spearman(
                                scope_risk_pred, scope_metrics["barrier"] > 0.025
                            ),
                        }
                        diagnostics.append({**scope_diag, "skipped": False})
                        for top_frac in top_fracs:
                            for selection in selections:
                                risk_threshold = None
                                if selection.risk_q is not None:
                                    risk_threshold = risk_thresholds.get(float(selection.risk_q))
                                if selection.mode == "risk_rank_gate":
                                    risk_threshold = float(
                                        scope_risk_pred.dropna().quantile(float(selection.risk_q or 0.50))
                                    )
                                selected_local = _select_indices(
                                    utility_pred=scope_utility_pred,
                                    risk_pred=scope_risk_pred,
                                    risk_threshold=risk_threshold,
                                    selection=selection,
                                    top_frac=top_frac,
                                )
                                selected_metrics = (
                                    scope_metrics.iloc[selected_local].copy()
                                    if len(selected_local)
                                    else scope_metrics.iloc[:0].copy()
                                )
                                selected_risk_pred = (
                                    scope_risk_pred.iloc[selected_local].copy()
                                    if len(selected_local)
                                    else pd.Series(dtype=float)
                                )
                                row = {
                                    **scope_diag,
                                    "selection": selection.name,
                                    "top_frac": float(top_frac),
                                    "risk_threshold": float(risk_threshold)
                                    if risk_threshold is not None and math.isfinite(float(risk_threshold))
                                    else float("nan"),
                                    "eligible_rows": int(
                                        scope_risk_pred.le(float(risk_threshold)).sum()
                                        if risk_threshold is not None and math.isfinite(float(risk_threshold))
                                        else scope_rows
                                    ),
                                    "selected_rows": int(len(selected_metrics)),
                                    **_path_summary(selected_metrics),
                                    "mean_utility_pred": _safe_mean(scope_utility_pred.iloc[selected_local]),
                                    "mean_risk_pred": _safe_mean(selected_risk_pred),
                                    "mean_risk_target_soft": _safe_mean(
                                        scope_risk_target["target_soft"].iloc[selected_local]
                                    ),
                                }
                                rows.append(row)
                                context = {
                                    "candidate": (
                                        f"{label_name}__{risk_spec.name}__{feature_set}__"
                                        f"{source_bucket}__{causal_gate}__{selection.name}__top{top_frac}"
                                    ),
                                    "period": month,
                                    "label": label_name,
                                    "risk_target": risk_spec.name,
                                    "feature_set": feature_set,
                                    "source_bucket": source_bucket,
                                    "causal_gate": causal_gate,
                                    "selection": selection.name,
                                    "top_frac": float(top_frac),
                                    "scope_rows": scope_rows,
                                }
                                selected_frames.append(
                                    _selected_frame(
                                        frame=scope_frame,
                                        metrics=scope_metrics,
                                        utility_target=scope_utility_target,
                                        risk_target=scope_risk_target,
                                        utility_pred=scope_utility_pred,
                                        risk_pred=scope_risk_pred,
                                        selected_idx=selected_local,
                                        context=context,
                                    )
                                )
    return rows, selected_frames, diagnostics


def run_diagnostic(
    *,
    quality_labels_path: Path,
    labels_path: Path,
    output_dir: Path,
    feature_dir: Path,
    feature_list_csv: Path,
    max_feature_store_features: int | None,
    months: list[str],
    labels_requested: list[str],
    risk_target_names: list[str],
    feature_sets: list[str],
    source_buckets: list[str],
    causal_gate_names: list[str],
    selection_names: list[str],
    top_fracs: list[float],
    seeds: list[int],
    train_lookback_months: int | None,
    min_train_rows: int,
    min_valid_rows: int,
    min_scope_rows: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = {spec.name: spec for spec in _label_specs_by_name(labels_requested)}
    risk_targets = _risk_targets_by_name(risk_target_names)
    gates = {spec.name: spec for spec in _gate_specs_by_name(causal_gate_names)}
    selections = _selection_specs_by_name(selection_names)
    _assert_gate_columns_causal(list(gates.values()))

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
    base_features = list(feature_matrix.columns)
    source_features = _source_feature_columns(frame)
    feature_map = {
        "base": base_features,
        "base_plus_source": list(dict.fromkeys(base_features + source_features)),
    }
    month_period = frame["__ts__"].dt.to_period("M").astype(str)

    rows: list[dict[str, Any]] = []
    selected_frames: list[pd.DataFrame] = []
    diagnostics: list[dict[str, Any]] = []
    for month in months:
        month_rows, month_selected, month_diag = _score_month(
            frame=frame,
            metrics=metrics,
            month_period=month_period,
            month=month,
            labels=labels,
            risk_targets=risk_targets,
            gates=gates,
            feature_map=feature_map,
            labels_requested=labels_requested,
            feature_sets=feature_sets,
            source_buckets=source_buckets,
            causal_gates=causal_gate_names,
            selections=selections,
            top_fracs=top_fracs,
            seeds=seeds,
            train_lookback_months=train_lookback_months,
            min_train_rows=min_train_rows,
            min_valid_rows=min_valid_rows,
            min_scope_rows=min_scope_rows,
        )
        rows.extend(month_rows)
        selected_frames.extend(month_selected)
        diagnostics.extend(month_diag)

    monthly = _add_utility_only_deltas(pd.DataFrame(rows))
    selected = pd.concat([frame for frame in selected_frames if not frame.empty], ignore_index=True) if selected_frames else pd.DataFrame()
    weekly = _weekly_summary(selected)
    aggregate = _aggregate(monthly, weekly, expected_months=len(months))
    diagnostics_frame = pd.DataFrame(diagnostics)

    paths = {
        "monthly": output_dir / "source_utility_path_risk_dual_head_monthly.csv",
        "weekly": output_dir / "source_utility_path_risk_dual_head_weekly.csv",
        "aggregate": output_dir / "source_utility_path_risk_dual_head_aggregate.csv",
        "diagnostics": output_dir / "source_utility_path_risk_dual_head_diagnostics.csv",
        "selected_rows_parquet": output_dir / "source_utility_path_risk_dual_head_selected_rows.parquet",
        "selected_rows_csv": output_dir / "source_utility_path_risk_dual_head_selected_rows.csv",
        "manifest": output_dir / "manifest.json",
    }
    monthly.to_csv(paths["monthly"], index=False)
    weekly.to_csv(paths["weekly"], index=False)
    aggregate.to_csv(paths["aggregate"], index=False)
    diagnostics_frame.to_csv(paths["diagnostics"], index=False)
    selected.to_parquet(paths["selected_rows_parquet"], index=False)
    selected.to_csv(paths["selected_rows_csv"], index=False)
    manifest = {
        "scope": "source_utility_path_risk_dual_head_diagnostic",
        "quality_labels_path": str(quality_labels_path),
        "labels_path": str(labels_path),
        "output_dir": str(output_dir),
        "rows": int(len(frame)),
        "utility_source": metrics.attrs.get("utility_source"),
        "months": list(months),
        "labels": labels_requested,
        "risk_targets": risk_target_names,
        "feature_sets": feature_sets,
        "source_buckets": source_buckets,
        "causal_gates": causal_gate_names,
        "selections": selection_names,
        "top_fracs": [float(v) for v in top_fracs],
        "seeds": [int(seed) for seed in seeds],
        "join_report": join_report,
        "feature_store": feature_report,
        "base_feature_count": int(len(base_features)),
        "source_feature_count": int(len(source_features)),
        "outputs": {key: str(path) for key, path in paths.items()},
    }
    paths["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    markdown = _write_report(output_dir, aggregate, weekly, manifest)
    manifest["outputs"]["markdown"] = str(markdown)
    paths["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quality-labels-path", type=Path, default=DEFAULT_QUALITY_LABELS)
    parser.add_argument("--labels-path", type=Path, default=DEFAULT_LABELS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--feature-list-csv", type=Path, default=DEFAULT_FEATURE_LIST_CSV)
    parser.add_argument("--max-feature-store-features", type=int, default=96)
    parser.add_argument("--months", type=str, default=",".join(DEFAULT_MONTHS))
    parser.add_argument("--labels", type=str, default=",".join(DEFAULT_LABELS))
    parser.add_argument("--risk-targets", type=str, default=",".join(DEFAULT_RISK_TARGETS))
    parser.add_argument("--feature-sets", type=str, default=",".join(DEFAULT_FEATURE_SETS))
    parser.add_argument("--source-buckets", type=str, default=",".join(DEFAULT_SOURCE_BUCKETS))
    parser.add_argument("--causal-gates", type=str, default=",".join(DEFAULT_CAUSAL_GATES))
    parser.add_argument("--selections", type=str, default=",".join(DEFAULT_SELECTIONS))
    parser.add_argument("--top-fracs", type=str, default=",".join(str(v) for v in DEFAULT_TOP_FRACS))
    parser.add_argument("--seeds", type=str, default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--train-lookback-months", type=int, default=None)
    parser.add_argument("--min-train-rows", type=int, default=500)
    parser.add_argument("--min-valid-rows", type=int, default=100)
    parser.add_argument("--min-scope-rows", type=int, default=80)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = run_diagnostic(
        quality_labels_path=args.quality_labels_path,
        labels_path=args.labels_path,
        output_dir=args.output_dir,
        feature_dir=args.feature_dir,
        feature_list_csv=args.feature_list_csv,
        max_feature_store_features=args.max_feature_store_features,
        months=_parse_csv(args.months, DEFAULT_MONTHS),
        labels_requested=_parse_csv(args.labels, DEFAULT_LABELS),
        risk_target_names=_parse_csv(args.risk_targets, DEFAULT_RISK_TARGETS),
        feature_sets=_parse_csv(args.feature_sets, DEFAULT_FEATURE_SETS),
        source_buckets=_parse_csv(args.source_buckets, DEFAULT_SOURCE_BUCKETS),
        causal_gate_names=_parse_csv(args.causal_gates, DEFAULT_CAUSAL_GATES),
        selection_names=_parse_csv(args.selections, DEFAULT_SELECTIONS),
        top_fracs=_parse_float_csv(args.top_fracs, DEFAULT_TOP_FRACS),
        seeds=_parse_int_csv(args.seeds, DEFAULT_SEEDS),
        train_lookback_months=args.train_lookback_months,
        min_train_rows=int(args.min_train_rows),
        min_valid_rows=int(args.min_valid_rows),
        min_scope_rows=int(args.min_scope_rows),
    )
    print(json.dumps(_json_safe(manifest), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
