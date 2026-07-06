#!/usr/bin/env python3
"""Gated-oracle recoverability proxy before base/meta training.

This diagnostic asks a narrow question:

Can prior-month causal features recover the sparse, high-confidence oracle rows
defined by margin-clean first-touch utility labels?

It does not fit LightGBM, run Optuna, optimize policy geometry, or train the
production base/meta models. It uses the same simple rank-proxy machinery as
the other label QA scripts, with the validation month kept strictly out of its
own target construction.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from itertools import product
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
    _parse_csv,
    _parse_float_csv,
)
from scripts.run_label_dual_proxy_path_risk_ablation import _proxy_score  # noqa: E402
from scripts.run_label_first_touch_execution_proxy_ablation import (  # noqa: E402
    DEFAULT_FEATURE_DIR,
    DEFAULT_FEATURE_LIST_CSV,
    DEFAULT_FIT_MONTHS,
    DEFAULT_HOLDOUT_MONTH,
    _first_touch_metrics,
    _target_components as _first_touch_target_components,
    _json_safe,
    _safe_max,
    _safe_mean,
    _safe_min,
    _safe_numeric,
    _safe_quantile,
    _spearman,
)
from scripts.run_label_first_touch_soft_recipe_proxy_ablation import (  # noqa: E402
    _effective_n,
    _global_bad_soft,
    _timestamp_top_k_indices,
)
from scripts.run_label_quality_proxy_diagnostics import _feature_columns  # noqa: E402
from scripts.run_label_two_head_abstention_utility_proxy import _utility_targets  # noqa: E402
from scripts.run_soft_label_economic_proxy_ablation import (  # noqa: E402
    DEFAULT_EVENT_FEATURE_STORE_FEATURES,
    DEFAULT_PRIOR_WINDOWS_DAYS,
    DEFAULT_STATE_PATH_PRIOR_FEATURES,
)


DEFAULT_OUTPUT_DIR = Path("data_perp/reports/label_gated_oracle_recoverability_proxy_v1")
DEFAULT_UTILITY_TARGETS = (
    "fast_margin_clean_utility",
    "net_margin_clean_utility",
    "strict_margin_clean_utility",
)
DEFAULT_ORACLE_GATES = (0.50,)
DEFAULT_TOP_KS = (10, 20)
DEFAULT_PROXY_TOP_KS = (8, 12)
DEFAULT_PROXY_METHODS = ("ic",)
DEFAULT_BAD_THRESHOLDS = (0.10, 0.15)
DEFAULT_SCORE_FLOORS = (0.80, 0.90)
DEFAULT_RUN_ENTRY_GAP_HOURS = (0.0,)
DEFAULT_SCORE_FLOOR_SELECTORS = (
    "soft_proxy_bad_gate",
    "hard_oracle_proxy",
    "hard_oracle_proxy_bad_gate",
    "contrast_proxy_bad_gate",
    "soft_hard_blend_bad_gate",
    "soft_contrast_blend_bad_gate",
    "soft_cleanft_blend_bad_gate",
    "soft_pathsafe_blend_bad_gate",
    "soft_low_adverse_blend_bad_gate",
    "soft_low_dirty_blend_bad_gate",
    "cleanft_low_dirty_blend_bad_gate",
)


@dataclass(frozen=True)
class RecoverabilitySpec:
    selector: str
    utility_target: str
    oracle_gate: float
    proxy_method: str
    proxy_top_k: int
    bad_threshold: float | None
    score_floor: float | None
    run_entry_gap_hours: float
    top_k: int

    @property
    def arm(self) -> str:
        bad = "nogate" if self.bad_threshold is None else f"bad{int(round(self.bad_threshold * 100)):02d}"
        floor = "nofloor" if self.score_floor is None else f"floor{int(round(self.score_floor * 100)):02d}"
        gate = f"og{int(round(self.oracle_gate * 100)):02d}"
        run_entry = f"rgap{int(round(self.run_entry_gap_hours)):02d}h"
        return (
            f"{self.utility_target}_{gate}_{self.selector}_{self.proxy_method}"
            f"_pk{int(self.proxy_top_k)}_{bad}_{floor}_{run_entry}_top{int(self.top_k)}"
        )


def _finite_count(values: pd.Series) -> int:
    return int(np.isfinite(_safe_numeric(values).to_numpy(dtype=float)).sum())


def _candidate_stats(frame: pd.DataFrame, score: pd.Series) -> dict[str, Any]:
    score_series = _safe_numeric(score).reset_index(drop=True)
    timestamps = pd.to_datetime(frame["__ts__"], errors="coerce").reset_index(drop=True)
    finite = np.isfinite(score_series.to_numpy(dtype=float))
    total_ts = int(timestamps.nunique(dropna=True))
    active_counts = timestamps[finite].value_counts(dropna=True)
    return {
        "candidate_rows": int(finite.sum()),
        "candidate_rate": float(finite.sum() / len(score_series)) if len(score_series) else float("nan"),
        "candidate_timestamp_coverage": float(len(active_counts) / total_ts) if total_ts else float("nan"),
        "mean_candidates_per_active_ts": _safe_mean(active_counts),
    }


def _run_entry_score(frame: pd.DataFrame, score: pd.Series, *, gap_hours: float) -> pd.Series:
    if float(gap_hours) <= 0.0:
        return _safe_numeric(score).reset_index(drop=True)
    frame_reset = frame.reset_index(drop=True)
    score_reset = _safe_numeric(score).reset_index(drop=True)
    out = pd.Series(np.nan, index=score_reset.index, dtype=float)
    work = pd.DataFrame(
        {
            "__ts__": pd.to_datetime(frame_reset["__ts__"], errors="coerce"),
            "__symbol__": frame_reset["__symbol__"].astype(str),
            "__score__": score_reset.to_numpy(dtype=float, copy=False),
            "__pos__": np.arange(len(score_reset), dtype=np.int64),
        }
    ).sort_values(["__symbol__", "__ts__"], kind="mergesort")
    gap = pd.Timedelta(hours=float(gap_hours))
    for _, group in work.groupby("__symbol__", sort=False):
        prev_ts: pd.Timestamp | None = None
        prev_active = False
        for _, row in group.iterrows():
            score_value = float(row["__score__"])
            active = math.isfinite(score_value)
            ts = row["__ts__"]
            if active and (not prev_active or prev_ts is None or pd.isna(ts) or ts - prev_ts > gap):
                out.iloc[int(row["__pos__"])] = score_value
            prev_active = active
            if active and pd.notna(ts):
                prev_ts = ts
            elif not active:
                prev_ts = None
    return out


def _oracle_selected_mask(
    frame: pd.DataFrame,
    target_score: pd.Series,
    *,
    gate: float,
    top_k: int,
    run_entry_gap_hours: float,
) -> pd.Series:
    target = _safe_numeric(target_score).reset_index(drop=True)
    score = target.where(target >= float(gate))
    score = _run_entry_score(frame, score, gap_hours=float(run_entry_gap_hours))
    idx = _timestamp_top_k_indices(frame.reset_index(drop=True), score, int(top_k))
    out = pd.Series(False, index=frame.reset_index(drop=True).index)
    if len(idx):
        out.iloc[idx] = True
    return out


def _dirty_target(ft: pd.DataFrame) -> pd.Series:
    return (
        (_safe_numeric(ft["u_policy_net"]).fillna(-1.0) <= 0.0)
        | (_safe_numeric(ft["first_touch_stop"]).fillna(0.0) > 0.5)
        | (_safe_numeric(ft["first_touch_timeout"]).fillna(0.0) > 0.5)
        | (_safe_numeric(ft["first_touch_same_bar"]).fillna(0.0) > 0.5)
        | (_safe_numeric(ft["first_touch_mae_to_sl"]).fillna(10.0) >= 1.0)
        | (_safe_numeric(ft["first_touch_bar"]).fillna(36.0) > 16.0)
        | (_safe_numeric(ft["barrier"]).fillna(1.0) > 0.025)
    ).astype(float)


def _target_frame(target_soft: pd.Series, oracle_mask: pd.Series, bad_soft: pd.Series) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "target_soft": _safe_numeric(target_soft).reset_index(drop=True).clip(0.0, 1.0),
            "target_hard": oracle_mask.reset_index(drop=True).astype(float),
            "bad_soft": _safe_numeric(bad_soft).reset_index(drop=True).clip(0.0, 1.0),
        }
    )


def _apply_bad_and_floor(
    score: pd.Series,
    *,
    bad_proxy: pd.Series | None,
    bad_threshold: float | None,
    score_floor: float | None,
) -> pd.Series:
    out = _safe_numeric(score).reset_index(drop=True)
    mask = pd.Series(True, index=out.index)
    if bad_proxy is not None and bad_threshold is not None:
        mask &= _safe_numeric(bad_proxy).reset_index(drop=True) <= float(bad_threshold)
    if score_floor is not None:
        mask &= out >= float(score_floor)
    return out.where(mask)


def _selection_row(
    *,
    frame: pd.DataFrame,
    metrics: pd.DataFrame,
    target: pd.DataFrame,
    oracle_mask: pd.Series,
    score: pd.Series,
    spec: RecoverabilitySpec,
    period: str,
    period_type: str,
    week: str | None,
    diag: dict[str, Any],
) -> dict[str, Any]:
    frame = frame.reset_index(drop=True)
    metrics = metrics.reset_index(drop=True)
    target = target.reset_index(drop=True)
    oracle_mask = oracle_mask.reset_index(drop=True).astype(bool)
    score = _safe_numeric(score).reset_index(drop=True)
    idx = _timestamp_top_k_indices(frame, score, int(spec.top_k))
    selected = metrics.iloc[idx] if len(idx) else metrics.iloc[:0]
    selected_frame = frame.iloc[idx] if len(idx) else frame.iloc[:0]
    selected_mask = pd.Series(False, index=frame.index)
    if len(idx):
        selected_mask.iloc[idx] = True
    recovered = selected_mask & oracle_mask
    symbols = selected_frame.get("__symbol__", pd.Series(dtype=object))
    row = {
        "arm": spec.arm,
        "selector": spec.selector,
        "utility_target": spec.utility_target,
        "oracle_gate": float(spec.oracle_gate),
        "proxy_method": spec.proxy_method,
        "proxy_top_k": int(spec.proxy_top_k),
        "bad_threshold": float(spec.bad_threshold) if spec.bad_threshold is not None else float("nan"),
        "score_floor": float(spec.score_floor) if spec.score_floor is not None else float("nan"),
        "run_entry_gap_hours": float(spec.run_entry_gap_hours),
        "top_k": int(spec.top_k),
        "period_type": period_type,
        "period": str(period),
        "week": str(week) if week is not None else "",
        "rows": int(len(frame)),
        "selected_rows": int(len(idx)),
        "oracle_top_rows": int(oracle_mask.sum()),
        "oracle_recovered_rows": int(recovered.sum()),
        "oracle_recovery_rate": float(recovered.sum() / oracle_mask.sum()) if int(oracle_mask.sum()) else 0.0,
        "selected_oracle_overlap_rate": float(recovered.sum() / selected_mask.sum()) if int(selected_mask.sum()) else 0.0,
        "mean_u": _safe_mean(selected["u_policy_net"]),
        "median_u": _safe_quantile(selected["u_policy_net"], 0.50),
        "q05_u": _safe_quantile(selected["u_policy_net"], 0.05),
        "q10_u": _safe_quantile(selected["u_policy_net"], 0.10),
        "hit_u": _safe_mean(selected["u_policy_net"] > 0.0),
        "mean_return_net": _safe_mean(selected["ret_net"]),
        "hit_return_net": _safe_mean(selected["ret_net"] > 0.0),
        "clean_exec_actual_rate": _safe_mean(selected["clean_exec_actual"]),
        "first_touch_hit_rate": _safe_mean(selected["first_touch_hit"]),
        "first_touch_stop_rate": _safe_mean(selected["first_touch_stop"]),
        "first_touch_timeout_rate": _safe_mean(selected["first_touch_timeout"]),
        "first_touch_same_bar_rate": _safe_mean(selected["first_touch_same_bar"]),
        "first_touch_bad_mae_to_sl_rate": _safe_mean(selected["first_touch_mae_to_sl"] >= 1.0),
        "first_touch_wide_barrier_25bps_rate": _safe_mean(selected["barrier"] > 0.025),
        "mean_first_touch_mae_to_sl": _safe_mean(selected["first_touch_mae_to_sl"]),
        "p90_first_touch_mae_to_sl": _safe_quantile(selected["first_touch_mae_to_sl"], 0.90),
        "p90_first_touch_bar": _safe_quantile(selected["first_touch_bar"], 0.90),
        "score_ic_target": _spearman(score, target["target_soft"]),
        "score_ic_oracle": _spearman(score, oracle_mask.astype(float)),
        "score_ic_u": _spearman(score, metrics["u_policy_net"]),
        "score_ic_bad": _spearman(score, target["bad_soft"]),
        "score_ic_clean_exec": _spearman(score, metrics["clean_exec_actual"]),
        "symbol_effective_n": _effective_n(symbols),
        "top_symbol_share": float(symbols.value_counts(normalize=True, dropna=False).iloc[0]) if len(symbols) else 0.0,
    }
    row.update(_candidate_stats(frame, score))
    row.update(diag)
    return row


def _period_rows(
    *,
    valid_frame: pd.DataFrame,
    valid_metrics: pd.DataFrame,
    valid_target: pd.DataFrame,
    oracle_mask: pd.Series,
    score: pd.Series,
    spec: RecoverabilitySpec,
    month: str,
    diag: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    valid_frame = valid_frame.reset_index(drop=True)
    valid_metrics = valid_metrics.reset_index(drop=True)
    valid_target = valid_target.reset_index(drop=True)
    oracle_mask = oracle_mask.reset_index(drop=True)
    score = _safe_numeric(score).reset_index(drop=True)
    monthly = [
        _selection_row(
            frame=valid_frame,
            metrics=valid_metrics,
            target=valid_target,
            oracle_mask=oracle_mask,
            score=score,
            spec=spec,
            period=month,
            period_type="month",
            week=None,
            diag=diag,
        )
    ]
    weekly: list[dict[str, Any]] = []
    weeks = valid_frame["__ts__"].dt.to_period("W-SUN").astype(str)
    for week, ids in pd.Series(np.arange(len(valid_frame)), index=valid_frame.index).groupby(weeks, dropna=False):
        pos = ids.to_numpy(dtype=np.int64)
        if len(pos) < 20:
            continue
        row = _selection_row(
            frame=valid_frame.iloc[pos].reset_index(drop=True),
            metrics=valid_metrics.iloc[pos].reset_index(drop=True),
            target=valid_target.iloc[pos].reset_index(drop=True),
            oracle_mask=oracle_mask.iloc[pos].reset_index(drop=True),
            score=score.iloc[pos].reset_index(drop=True),
            spec=spec,
            period=month,
            period_type="week",
            week=str(week),
            diag=diag,
        )
        row["week_selected_rows"] = int(row["selected_rows"])
        weekly.append(row)
    return monthly, weekly


def _weighted_mean(frame: pd.DataFrame, value_col: str, weight_col: str = "selected_rows") -> float:
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
            f"{prefix}_mean_month_u": float("nan"),
            f"{prefix}_worst_month_u": float("nan"),
            f"{prefix}_selected_rows": 0,
        }
    mean_u = _safe_numeric(frame["mean_u"])
    mean_return_net = _safe_numeric(frame.get("mean_return_net", pd.Series(np.nan, index=frame.index)))
    selected_rows = _safe_numeric(frame["selected_rows"]).fillna(0.0)
    sum_return_net = float((mean_return_net.fillna(0.0) * selected_rows).sum())
    total_rows = float(selected_rows.sum())
    return {
        f"{prefix}_months": int(frame["period"].astype(str).nunique()),
        f"{prefix}_positive_months": int(mean_u.gt(0.0).sum()),
        f"{prefix}_mean_month_u": _safe_mean(mean_u),
        f"{prefix}_worst_month_u": _safe_min(mean_u),
        f"{prefix}_positive_return_months": int(mean_return_net.gt(0.0).sum()),
        f"{prefix}_mean_month_return_net": _safe_mean(mean_return_net),
        f"{prefix}_worst_month_return_net": _safe_min(mean_return_net),
        f"{prefix}_sum_return_net": sum_return_net,
        f"{prefix}_weighted_mean_return_net": float(sum_return_net / total_rows) if total_rows > 0.0 else float("nan"),
        f"{prefix}_selected_rows": int(total_rows),
        f"{prefix}_clean_exec_actual_rate": _weighted_mean(frame, "clean_exec_actual_rate"),
        f"{prefix}_first_touch_timeout_rate": _weighted_mean(frame, "first_touch_timeout_rate"),
        f"{prefix}_first_touch_bad_mae_to_sl_rate": _weighted_mean(frame, "first_touch_bad_mae_to_sl_rate"),
        f"{prefix}_first_touch_wide_barrier_25bps_rate": _weighted_mean(
            frame, "first_touch_wide_barrier_25bps_rate"
        ),
        f"{prefix}_p90_first_touch_mae_to_sl": _weighted_mean(frame, "p90_first_touch_mae_to_sl"),
        f"{prefix}_oracle_recovery_rate": _weighted_mean(frame, "oracle_recovery_rate", "oracle_top_rows"),
        f"{prefix}_selected_oracle_overlap_rate": _weighted_mean(frame, "selected_oracle_overlap_rate"),
        f"{prefix}_candidate_rate": _safe_mean(frame["candidate_rate"]),
        f"{prefix}_candidate_timestamp_coverage": _safe_mean(frame["candidate_timestamp_coverage"]),
        f"{prefix}_max_top_symbol_share": _safe_max(frame["top_symbol_share"]),
    }


def _summarize_week(prefix: str, frame: pd.DataFrame, *, min_week_rows: int) -> dict[str, Any]:
    if frame.empty:
        return {
            f"{prefix}_weeks": 0,
            f"{prefix}_material_weeks": 0,
            f"{prefix}_material_positive_week_rate": float("nan"),
            f"{prefix}_q25_week_u": float("nan"),
            f"{prefix}_worst_week_u": float("nan"),
            f"{prefix}_q25_week_return_net": float("nan"),
            f"{prefix}_worst_week_return_net": float("nan"),
        }
    week_rows = _safe_numeric(frame["week_selected_rows"]).fillna(0.0)
    material = week_rows >= int(min_week_rows)
    mean_u = _safe_numeric(frame["mean_u"])
    mean_return_net = _safe_numeric(frame.get("mean_return_net", pd.Series(np.nan, index=frame.index)))
    return {
        f"{prefix}_weeks": int(len(frame)),
        f"{prefix}_material_weeks": int(material.sum()),
        f"{prefix}_material_positive_week_rate": float((mean_u.gt(0.0) & material).sum() / material.sum())
        if int(material.sum())
        else float("nan"),
        f"{prefix}_q25_week_u": _safe_quantile(mean_u[material], 0.25) if int(material.sum()) else float("nan"),
        f"{prefix}_worst_week_u": _safe_min(mean_u[material]) if int(material.sum()) else float("nan"),
        f"{prefix}_material_positive_return_week_rate": float(
            (mean_return_net.gt(0.0) & material).sum() / material.sum()
        )
        if int(material.sum())
        else float("nan"),
        f"{prefix}_q25_week_return_net": _safe_quantile(mean_return_net[material], 0.25)
        if int(material.sum())
        else float("nan"),
        f"{prefix}_worst_week_return_net": _safe_min(mean_return_net[material])
        if int(material.sum())
        else float("nan"),
    }


def _fit_holdout_summary(
    *,
    monthly: pd.DataFrame,
    weekly: pd.DataFrame,
    fit_months: list[str],
    holdout_month: str,
    min_week_rows: int,
) -> pd.DataFrame:
    if monthly.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    group_cols = [
        "selector",
        "utility_target",
        "oracle_gate",
        "proxy_method",
        "proxy_top_k",
        "bad_threshold",
        "score_floor",
        "run_entry_gap_hours",
        "top_k",
    ]
    for key, group in monthly.groupby(group_cols, observed=True, dropna=False):
        (
            selector,
            utility_target,
            oracle_gate,
            proxy_method,
            proxy_top_k,
            bad_threshold,
            score_floor,
            run_entry_gap_hours,
            top_k,
        ) = key
        week_group = weekly[
            weekly["selector"].astype(str).eq(str(selector))
            & weekly["utility_target"].astype(str).eq(str(utility_target))
            & _safe_numeric(weekly["oracle_gate"]).eq(float(oracle_gate))
            & weekly["proxy_method"].astype(str).eq(str(proxy_method))
            & _safe_numeric(weekly["proxy_top_k"]).eq(int(proxy_top_k))
            & _safe_numeric(weekly["top_k"]).eq(int(top_k))
            & _safe_numeric(weekly["run_entry_gap_hours"]).eq(float(run_entry_gap_hours))
            & (
                _safe_numeric(weekly["bad_threshold"]).isna()
                if pd.isna(bad_threshold)
                else _safe_numeric(weekly["bad_threshold"]).eq(float(bad_threshold))
            )
            & (
                _safe_numeric(weekly["score_floor"]).isna()
                if pd.isna(score_floor)
                else _safe_numeric(weekly["score_floor"]).eq(float(score_floor))
            )
        ].copy()
        fit_month = group[group["period"].astype(str).isin(fit_months)].copy()
        holdout_monthly = group[group["period"].astype(str).eq(str(holdout_month))].copy()
        fit_week = week_group[week_group["period"].astype(str).isin(fit_months)].copy()
        holdout_week = week_group[week_group["period"].astype(str).eq(str(holdout_month))].copy()
        if fit_month.empty or holdout_monthly.empty:
            continue
        row: dict[str, Any] = {
            "selector": str(selector),
            "utility_target": str(utility_target),
            "oracle_gate": float(oracle_gate),
            "proxy_method": str(proxy_method),
            "proxy_top_k": int(proxy_top_k),
            "bad_threshold": float(bad_threshold) if pd.notna(bad_threshold) else float("nan"),
            "score_floor": float(score_floor) if pd.notna(score_floor) else float("nan"),
            "run_entry_gap_hours": float(run_entry_gap_hours),
            "top_k": int(top_k),
        }
        row.update(_summarize_month("fit", fit_month))
        row.update(_summarize_month("holdout", holdout_monthly))
        row.update(_summarize_week("fit", fit_week, min_week_rows=min_week_rows))
        row.update(_summarize_week("holdout", holdout_week, min_week_rows=min_week_rows))
        fit_sign = (
            row["fit_months"] == len(fit_months)
            and row["fit_positive_months"] == len(fit_months)
            and row["fit_worst_month_u"] > 0.0
            and row["fit_material_weeks"] >= 4
            and row["fit_material_positive_week_rate"] >= 0.55
        )
        holdout_sign = (
            row["holdout_mean_month_u"] > 0.0
            and row["holdout_material_weeks"] >= 2
            and row["holdout_material_positive_week_rate"] >= 0.50
        )
        fit_bounded = (
            fit_sign
            and row["fit_clean_exec_actual_rate"] >= 0.20
            and row["fit_first_touch_timeout_rate"] <= 0.55
            and row["fit_first_touch_bad_mae_to_sl_rate"] <= 0.60
            and row["fit_p90_first_touch_mae_to_sl"] <= 2.0
        )
        holdout_bounded_standalone = (
            holdout_sign
            and row["holdout_clean_exec_actual_rate"] >= 0.20
            and row["holdout_first_touch_timeout_rate"] <= 0.55
            and row["holdout_first_touch_bad_mae_to_sl_rate"] <= 0.60
            and row["holdout_p90_first_touch_mae_to_sl"] <= 2.0
        )
        fit_objective = (
            (row["fit_mean_month_u"] if pd.notna(row["fit_mean_month_u"]) else -1.0)
            + 0.50 * (row["fit_q25_week_u"] if pd.notna(row["fit_q25_week_u"]) else -1.0)
            + 0.20 * (row["fit_oracle_recovery_rate"] if pd.notna(row["fit_oracle_recovery_rate"]) else 0.0)
            - 0.010
            * (row["fit_p90_first_touch_mae_to_sl"] if pd.notna(row["fit_p90_first_touch_mae_to_sl"]) else 10.0)
            - 0.020
            * (
                row["fit_first_touch_bad_mae_to_sl_rate"]
                if pd.notna(row["fit_first_touch_bad_mae_to_sl_rate"])
                else 1.0
            )
        )
        holdout_objective = (
            (row["holdout_mean_month_u"] if pd.notna(row["holdout_mean_month_u"]) else -1.0)
            + 0.50 * (row["holdout_q25_week_u"] if pd.notna(row["holdout_q25_week_u"]) else -1.0)
            + 0.20 * (row["holdout_oracle_recovery_rate"] if pd.notna(row["holdout_oracle_recovery_rate"]) else 0.0)
            - 0.010
            * (
                row["holdout_p90_first_touch_mae_to_sl"]
                if pd.notna(row["holdout_p90_first_touch_mae_to_sl"])
                else 10.0
            )
            - 0.020
            * (
                row["holdout_first_touch_bad_mae_to_sl_rate"]
                if pd.notna(row["holdout_first_touch_bad_mae_to_sl_rate"])
                else 1.0
            )
        )
        row["fit_sign_pass"] = bool(fit_sign)
        row["holdout_sign_pass"] = bool(holdout_sign)
        row["fit_bounded_pass"] = bool(fit_bounded)
        row["holdout_bounded_standalone_pass"] = bool(holdout_bounded_standalone)
        row["holdout_bounded_pass"] = bool(fit_bounded and holdout_bounded_standalone)
        row["positive_dirty_holdout"] = bool(holdout_sign and not holdout_bounded_standalone)
        row["fit_selection_objective"] = float(fit_objective)
        row["holdout_objective"] = float(holdout_objective)
        rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(
        ["holdout_bounded_pass", "positive_dirty_holdout", "holdout_objective", "holdout_mean_month_u"],
        ascending=[False, False, False, False],
    )


def _select_by_fit(fit_holdout: pd.DataFrame) -> pd.DataFrame:
    if fit_holdout.empty:
        return fit_holdout
    rows: list[pd.Series] = []
    causal = fit_holdout[~fit_holdout["selector"].astype(str).eq("oracle_ceiling")].copy()
    group_cols = ["utility_target", "oracle_gate", "top_k"]
    for _, group in causal.groupby(group_cols, observed=True, dropna=False):
        candidates = group.copy()
        if bool(candidates["fit_bounded_pass"].any()):
            candidates = candidates[candidates["fit_bounded_pass"]].copy()
        elif bool(candidates["fit_sign_pass"].any()):
            candidates = candidates[candidates["fit_sign_pass"]].copy()
        chosen = candidates.sort_values(
            ["fit_bounded_pass", "fit_sign_pass", "fit_selection_objective", "fit_mean_month_u"],
            ascending=[False, False, False, False],
        ).iloc[0]
        rows.append(chosen)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(
        ["holdout_bounded_pass", "holdout_sign_pass", "holdout_objective", "fit_selection_objective"],
        ascending=[False, False, False, False],
    )


def _format_table(frame: pd.DataFrame, cols: list[str], limit: int = 60) -> str:
    if frame.empty:
        return "No rows."
    view = frame[[col for col in cols if col in frame.columns]].head(limit).copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda value: f"{float(value):.4f}" if pd.notna(value) else "")
    return view.to_markdown(index=False)


def _write_markdown(
    *,
    output_dir: Path,
    fit_holdout: pd.DataFrame,
    selected_by_fit: pd.DataFrame,
    monthly: pd.DataFrame,
    proxy_diagnostics: pd.DataFrame,
    manifest: dict[str, Any],
) -> Path:
    path = output_dir / "gated_oracle_recoverability_proxy.md"
    counts = (
        pd.DataFrame(
            [
                {
                    "rows": int(len(fit_holdout)),
                    "fit_sign": int(fit_holdout["fit_sign_pass"].sum()),
                    "holdout_sign": int(fit_holdout["holdout_sign_pass"].sum()),
                    "fit_bounded": int(fit_holdout["fit_bounded_pass"].sum()),
                    "holdout_bounded": int(fit_holdout["holdout_bounded_pass"].sum()),
                    "positive_dirty": int(fit_holdout["positive_dirty_holdout"].sum()),
                }
            ]
        )
        if not fit_holdout.empty
        else pd.DataFrame()
    )
    cols = [
        "selector",
        "utility_target",
        "oracle_gate",
        "proxy_method",
        "proxy_top_k",
        "bad_threshold",
        "score_floor",
        "run_entry_gap_hours",
        "top_k",
        "fit_selection_objective",
        "holdout_objective",
        "fit_sign_pass",
        "fit_bounded_pass",
        "holdout_sign_pass",
        "holdout_bounded_pass",
        "positive_dirty_holdout",
        "fit_mean_month_u",
        "fit_material_positive_week_rate",
        "fit_oracle_recovery_rate",
        "fit_clean_exec_actual_rate",
        "fit_first_touch_bad_mae_to_sl_rate",
        "fit_p90_first_touch_mae_to_sl",
        "holdout_mean_month_u",
        "holdout_material_positive_week_rate",
        "holdout_oracle_recovery_rate",
        "holdout_selected_oracle_overlap_rate",
        "holdout_clean_exec_actual_rate",
        "holdout_first_touch_timeout_rate",
        "holdout_first_touch_bad_mae_to_sl_rate",
        "holdout_p90_first_touch_mae_to_sl",
    ]
    month_cols = [
        "period",
        "selector",
        "utility_target",
        "oracle_gate",
        "proxy_method",
        "proxy_top_k",
        "bad_threshold",
        "score_floor",
        "run_entry_gap_hours",
        "top_k",
        "selected_rows",
        "oracle_top_rows",
        "oracle_recovery_rate",
        "selected_oracle_overlap_rate",
        "mean_u",
        "mean_return_net",
        "clean_exec_actual_rate",
        "first_touch_bad_mae_to_sl_rate",
        "p90_first_touch_mae_to_sl",
        "first_touch_timeout_rate",
        "proxy_features",
    ]
    diag_cols = [
        "month",
        "utility_target",
        "oracle_gate",
        "proxy_method",
        "proxy_top_k",
        "oracle_target_pos",
        "contrast_target_pos",
        "contrast_target_neg",
        "soft_proxy_features",
        "hard_proxy_features",
        "contrast_proxy_features",
        "bad_proxy_features",
        "cleanft_proxy_features",
        "early_adverse_proxy_features",
        "slow_timeout_proxy_features",
        "path_dirty_proxy_features",
    ]
    causal_month = monthly[~monthly["selector"].astype(str).eq("oracle_ceiling")].copy()
    lines = [
        "# Gated-Oracle Recoverability Proxy",
        "",
        "Scope: proxy-only development diagnostic. No LightGBM, Optuna, policy geometry optimization, or base/meta training is run.",
        "",
        "For each validation month, the target and proxy features are fit only on prior months. The oracle rows are used as a ceiling and as prior-month labels, never from the validation month itself.",
        "",
        f"Labels: `{manifest['labels_path']}`",
        f"Utility targets: `{', '.join(manifest['utility_targets'])}`",
        f"Oracle gates: `{manifest['oracle_gates']}`",
        f"Top K: `{manifest['top_ks']}`",
        f"Proxy methods: `{manifest['proxy_methods']}`",
        f"Proxy top K features: `{manifest['proxy_top_ks']}`",
        f"Run-entry gap hours: `{manifest['run_entry_gap_hours']}`",
        f"Fit months: `{','.join(manifest['fit_months'])}`",
        f"Holdout month: `{manifest['holdout_month']}`",
        f"Causal outcome priors: `{manifest['include_causal_outcome_priors']}`",
        f"Causal state-path priors: `{manifest['include_causal_state_path_priors']}`",
        f"Event confirmation features: `{manifest['include_event_confirmation_features']}`",
        "",
        "## Counts",
        "",
        _format_table(counts, ["rows", "fit_sign", "holdout_sign", "fit_bounded", "holdout_bounded", "positive_dirty"], limit=10),
        "",
        "## Selected By Apr-May Fit Only",
        "",
        _format_table(selected_by_fit, cols, limit=60),
        "",
        "## Best Causal Grid Rows",
        "",
        _format_table(
            fit_holdout[~fit_holdout["selector"].astype(str).eq("oracle_ceiling")].copy(),
            cols,
            limit=80,
        ),
        "",
        "## Oracle Ceiling Rows",
        "",
        _format_table(
            fit_holdout[fit_holdout["selector"].astype(str).eq("oracle_ceiling")].copy(),
            cols,
            limit=30,
        ),
        "",
        "## June Causal Month Detail",
        "",
        _format_table(
            causal_month[causal_month["period"].astype(str).eq(str(manifest["holdout_month"]))].sort_values(
                ["mean_u", "oracle_recovery_rate"], ascending=[False, False]
            ),
            month_cols,
            limit=80,
        ),
        "",
        "## Proxy Diagnostics",
        "",
        _format_table(proxy_diagnostics, diag_cols, limit=80),
        "",
        "## Outputs",
        "",
        f"- Monthly: `{manifest['outputs']['monthly']}`",
        f"- Weekly: `{manifest['outputs']['weekly']}`",
        f"- Fit/Holdout: `{manifest['outputs']['fit_holdout']}`",
        f"- Selected by fit: `{manifest['outputs']['selected_by_fit']}`",
        f"- Proxy diagnostics: `{manifest['outputs']['proxy_diagnostics']}`",
        f"- Manifest: `{manifest['outputs']['manifest']}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _build_specs(
    *,
    selectors: list[str],
    utility_target: str,
    oracle_gate: float,
    proxy_method: str,
    proxy_top_k: int,
    bad_thresholds: list[float],
    score_floors: list[float],
    run_entry_gap_hours: list[float],
    top_ks: list[int],
) -> list[RecoverabilitySpec]:
    specs: list[RecoverabilitySpec] = []
    for top_k, run_gap in product(top_ks, run_entry_gap_hours):
        specs.append(
            RecoverabilitySpec(
                selector="oracle_ceiling",
                utility_target=utility_target,
                oracle_gate=float(oracle_gate),
                proxy_method=proxy_method,
                proxy_top_k=int(proxy_top_k),
                bad_threshold=None,
                score_floor=None,
                run_entry_gap_hours=float(run_gap),
                top_k=int(top_k),
            )
        )
    for selector, bad_threshold, score_floor, run_gap, top_k in product(
        selectors,
        bad_thresholds,
        score_floors,
        run_entry_gap_hours,
        top_ks,
    ):
        if selector == "hard_oracle_proxy":
            bad_value: float | None = None
        else:
            bad_value = float(bad_threshold)
        specs.append(
            RecoverabilitySpec(
                selector=selector,
                utility_target=utility_target,
                oracle_gate=float(oracle_gate),
                proxy_method=proxy_method,
                proxy_top_k=int(proxy_top_k),
                bad_threshold=bad_value,
                score_floor=float(score_floor),
                run_entry_gap_hours=float(run_gap),
                top_k=int(top_k),
            )
        )
    return specs


def run_ablation(
    *,
    labels_path: Path,
    output_dir: Path,
    feature_dir: Path,
    feature_list_csv: Path,
    max_feature_store_features: int | None,
    utility_targets: list[str],
    oracle_gates: list[float],
    proxy_methods: list[str],
    proxy_top_ks: list[int],
    selectors: list[str],
    top_ks: list[int],
    bad_thresholds: list[float],
    score_floors: list[float],
    run_entry_gap_hours: list[float],
    fit_months: list[str],
    holdout_month: str,
    min_week_rows: int,
    include_causal_outcome_priors: bool,
    include_causal_state_path_priors: bool,
    include_event_confirmation_features: bool,
    include_adverse_path_composites: bool,
    prior_windows_days: list[float],
    prior_embargo_hours: float,
    state_path_prior_features: list[str],
    event_feature_store_features: list[str],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    unknown_selectors = sorted(set(selectors) - set(DEFAULT_SCORE_FLOOR_SELECTORS))
    if unknown_selectors:
        raise ValueError(f"Unknown selector(s): {unknown_selectors}")
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
    features = _feature_columns(frame)
    ft = _first_touch_metrics(frame, metrics)
    bad_soft = _global_bad_soft(ft)
    dirty = _dirty_target(ft)
    first_touch_components = _first_touch_target_components(ft)
    utility_map = _utility_targets(frame, ft)
    missing = sorted(set(utility_targets) - set(utility_map))
    if missing:
        raise ValueError(f"Unknown utility target(s): {missing}")

    month_series = frame["__ts__"].dt.to_period("M").astype(str)
    months = sorted(m for m in month_series.dropna().unique().tolist() if m >= "2026-04")
    monthly_rows: list[dict[str, Any]] = []
    weekly_rows: list[dict[str, Any]] = []
    proxy_diag_rows: list[dict[str, Any]] = []

    for month in months:
        train_mask = month_series < str(month)
        valid_mask = month_series == str(month)
        if int(train_mask.sum()) < 500 or int(valid_mask.sum()) < 100:
            continue
        train = frame.loc[train_mask].copy()
        valid = frame.loc[valid_mask].copy()
        valid_frame = valid.reset_index(drop=True)
        valid_metrics = ft.loc[valid_mask].copy().reset_index(drop=True)
        valid_bad_soft = bad_soft.loc[valid_mask].reset_index(drop=True)
        train_bad_soft = bad_soft.loc[train_mask]

        for proxy_method in proxy_methods:
            for proxy_top_k in proxy_top_ks:
                bad_proxy, bad_diag = _proxy_score(
                    train=train,
                    valid=valid,
                    features=features,
                    target_train=train_bad_soft,
                    top_k=int(proxy_top_k),
                    method=str(proxy_method),
                    tail_frac=0.05,
                )
                bad_proxy = _safe_numeric(bad_proxy).reset_index(drop=True)
                cleanft_proxy, cleanft_diag = _proxy_score(
                    train=train,
                    valid=valid,
                    features=features,
                    target_train=first_touch_components["clean_first_touch"].loc[train_mask],
                    top_k=int(proxy_top_k),
                    method=str(proxy_method),
                    tail_frac=0.05,
                )
                early_adverse_proxy, early_adverse_diag = _proxy_score(
                    train=train,
                    valid=valid,
                    features=features,
                    target_train=first_touch_components["early_adverse"].loc[train_mask],
                    top_k=int(proxy_top_k),
                    method=str(proxy_method),
                    tail_frac=0.05,
                )
                slow_timeout_proxy, slow_timeout_diag = _proxy_score(
                    train=train,
                    valid=valid,
                    features=features,
                    target_train=first_touch_components["slow_timeout"].loc[train_mask],
                    top_k=int(proxy_top_k),
                    method=str(proxy_method),
                    tail_frac=0.05,
                )
                path_dirty_proxy, path_dirty_diag = _proxy_score(
                    train=train,
                    valid=valid,
                    features=features,
                    target_train=first_touch_components["dirty"].loc[train_mask],
                    top_k=int(proxy_top_k),
                    method=str(proxy_method),
                    tail_frac=0.05,
                )
                cleanft_proxy = _safe_numeric(cleanft_proxy).reset_index(drop=True)
                early_adverse_proxy = _safe_numeric(early_adverse_proxy).reset_index(drop=True)
                slow_timeout_proxy = _safe_numeric(slow_timeout_proxy).reset_index(drop=True)
                path_dirty_proxy = _safe_numeric(path_dirty_proxy).reset_index(drop=True)
                low_early_adverse_proxy = (1.0 - early_adverse_proxy.fillna(0.5)).clip(0.0, 1.0)
                low_slow_timeout_proxy = (1.0 - slow_timeout_proxy.fillna(0.5)).clip(0.0, 1.0)
                low_path_dirty_proxy = (1.0 - path_dirty_proxy.fillna(0.5)).clip(0.0, 1.0)
                path_safe_proxy = (
                    (
                        cleanft_proxy.fillna(0.5)
                        + low_early_adverse_proxy
                        + low_slow_timeout_proxy
                        + low_path_dirty_proxy
                    )
                    / 4.0
                ).clip(0.0, 1.0)

                for utility_target in utility_targets:
                    utility_soft = utility_map[utility_target]
                    soft_proxy, soft_diag = _proxy_score(
                        train=train,
                        valid=valid,
                        features=features,
                        target_train=utility_soft.loc[train_mask],
                        top_k=int(proxy_top_k),
                        method=str(proxy_method),
                        tail_frac=0.05,
                    )
                    soft_proxy = _safe_numeric(soft_proxy).reset_index(drop=True)

                    for oracle_gate in oracle_gates:
                        for run_gap in run_entry_gap_hours:
                            train_oracle_mask = _oracle_selected_mask(
                                frame.loc[train_mask],
                                utility_soft.loc[train_mask],
                                gate=float(oracle_gate),
                                top_k=max(top_ks),
                                run_entry_gap_hours=float(run_gap),
                            )
                            valid_oracle_masks = {
                                int(top_k): _oracle_selected_mask(
                                    valid_frame,
                                    utility_soft.loc[valid_mask].reset_index(drop=True),
                                    gate=float(oracle_gate),
                                    top_k=int(top_k),
                                    run_entry_gap_hours=float(run_gap),
                                )
                                for top_k in top_ks
                            }
                            train_index = frame.loc[train_mask].index
                            train_hard_target = pd.Series(
                                train_oracle_mask.to_numpy(dtype=float, copy=False),
                                index=train_index,
                                dtype=float,
                            )
                            train_contrast_target = pd.Series(np.nan, index=train_index, dtype=float)
                            train_oracle_on_index = train_hard_target.astype(bool)
                            pos_ids = train_index[train_oracle_on_index.to_numpy(dtype=bool, copy=False)]
                            neg_mask = dirty.loc[train_mask].astype(bool) & ~train_oracle_on_index
                            neg_ids = train_index[neg_mask.to_numpy(dtype=bool, copy=False)]
                            train_contrast_target.loc[pos_ids] = 1.0
                            train_contrast_target.loc[neg_ids] = 0.0

                            hard_proxy, hard_diag = _proxy_score(
                                train=train,
                                valid=valid,
                                features=features,
                                target_train=train_hard_target,
                                top_k=int(proxy_top_k),
                                method=str(proxy_method),
                                tail_frac=0.05,
                            )
                            contrast_proxy, contrast_diag = _proxy_score(
                                train=train,
                                valid=valid,
                                features=features,
                                target_train=train_contrast_target,
                                top_k=int(proxy_top_k),
                                method=str(proxy_method),
                                tail_frac=0.05,
                            )
                            hard_proxy = _safe_numeric(hard_proxy).reset_index(drop=True)
                            contrast_proxy = _safe_numeric(contrast_proxy).reset_index(drop=True)
                            proxy_diag_rows.append(
                                {
                                    "month": str(month),
                                    "utility_target": str(utility_target),
                                    "oracle_gate": float(oracle_gate),
                                    "run_entry_gap_hours": float(run_gap),
                                    "proxy_method": str(proxy_method),
                                    "proxy_top_k": int(proxy_top_k),
                                    "train_rows": int(train_mask.sum()),
                                    "valid_rows": int(valid_mask.sum()),
                                    "oracle_target_pos": int(train_oracle_mask.sum()),
                                    "oracle_target_rate": float(train_oracle_mask.mean()) if len(train_oracle_mask) else 0.0,
                                    "contrast_target_pos": int((train_contrast_target == 1.0).sum()),
                                    "contrast_target_neg": int((train_contrast_target == 0.0).sum()),
                                    "soft_proxy_features": ",".join(soft_diag.get("features", [])),
                                    "hard_proxy_features": ",".join(hard_diag.get("features", [])),
                                    "contrast_proxy_features": ",".join(contrast_diag.get("features", [])),
                                    "bad_proxy_features": ",".join(bad_diag.get("features", [])),
                                    "cleanft_proxy_features": ",".join(cleanft_diag.get("features", [])),
                                    "early_adverse_proxy_features": ",".join(
                                        early_adverse_diag.get("features", [])
                                    ),
                                    "slow_timeout_proxy_features": ",".join(slow_timeout_diag.get("features", [])),
                                    "path_dirty_proxy_features": ",".join(path_dirty_diag.get("features", [])),
                                }
                            )

                            base_scores = {
                                "soft_proxy_bad_gate": soft_proxy,
                                "hard_oracle_proxy": hard_proxy,
                                "hard_oracle_proxy_bad_gate": hard_proxy,
                                "contrast_proxy_bad_gate": contrast_proxy,
                                "soft_hard_blend_bad_gate": 0.50 * soft_proxy + 0.50 * hard_proxy,
                                "soft_contrast_blend_bad_gate": 0.50 * soft_proxy + 0.50 * contrast_proxy,
                                "soft_cleanft_blend_bad_gate": 0.50 * soft_proxy + 0.50 * cleanft_proxy,
                                "soft_pathsafe_blend_bad_gate": 0.55 * soft_proxy + 0.45 * path_safe_proxy,
                                "soft_low_adverse_blend_bad_gate": (
                                    0.65 * soft_proxy + 0.35 * low_early_adverse_proxy
                                ),
                                "soft_low_dirty_blend_bad_gate": 0.65 * soft_proxy + 0.35 * low_path_dirty_proxy,
                                "cleanft_low_dirty_blend_bad_gate": (
                                    0.50 * cleanft_proxy + 0.50 * low_path_dirty_proxy
                                ),
                            }
                            selector_feature_diags = {
                                "soft_proxy_bad_gate": soft_diag,
                                "hard_oracle_proxy": hard_diag,
                                "hard_oracle_proxy_bad_gate": hard_diag,
                                "contrast_proxy_bad_gate": contrast_diag,
                                "soft_hard_blend_bad_gate": {
                                    "features": list(
                                        dict.fromkeys(
                                            list(soft_diag.get("features", []))
                                            + list(hard_diag.get("features", []))
                                        )
                                    )
                                },
                                "soft_contrast_blend_bad_gate": {
                                    "features": list(
                                        dict.fromkeys(
                                            list(soft_diag.get("features", []))
                                            + list(contrast_diag.get("features", []))
                                        )
                                    )
                                },
                                "soft_cleanft_blend_bad_gate": {
                                    "features": list(
                                        dict.fromkeys(
                                            list(soft_diag.get("features", []))
                                            + list(cleanft_diag.get("features", []))
                                        )
                                    )
                                },
                                "soft_pathsafe_blend_bad_gate": {
                                    "features": list(
                                        dict.fromkeys(
                                            list(soft_diag.get("features", []))
                                            + list(cleanft_diag.get("features", []))
                                            + list(early_adverse_diag.get("features", []))
                                            + list(slow_timeout_diag.get("features", []))
                                            + list(path_dirty_diag.get("features", []))
                                        )
                                    )
                                },
                                "soft_low_adverse_blend_bad_gate": {
                                    "features": list(
                                        dict.fromkeys(
                                            list(soft_diag.get("features", []))
                                            + list(early_adverse_diag.get("features", []))
                                        )
                                    )
                                },
                                "soft_low_dirty_blend_bad_gate": {
                                    "features": list(
                                        dict.fromkeys(
                                            list(soft_diag.get("features", []))
                                            + list(path_dirty_diag.get("features", []))
                                        )
                                    )
                                },
                                "cleanft_low_dirty_blend_bad_gate": {
                                    "features": list(
                                        dict.fromkeys(
                                            list(cleanft_diag.get("features", []))
                                            + list(path_dirty_diag.get("features", []))
                                        )
                                    )
                                },
                            }
                            specs = _build_specs(
                                selectors=list(selectors),
                                utility_target=str(utility_target),
                                oracle_gate=float(oracle_gate),
                                proxy_method=str(proxy_method),
                                proxy_top_k=int(proxy_top_k),
                                bad_thresholds=bad_thresholds,
                                score_floors=score_floors,
                                run_entry_gap_hours=[float(run_gap)],
                                top_ks=top_ks,
                            )
                            valid_target_by_top_k = {
                                int(top_k): _target_frame(
                                    utility_soft.loc[valid_mask].reset_index(drop=True),
                                    valid_oracle_masks[int(top_k)],
                                    valid_bad_soft,
                                )
                                for top_k in top_ks
                            }
                            for spec in specs:
                                if spec.selector == "oracle_ceiling":
                                    raw_score = _safe_numeric(
                                        utility_soft.loc[valid_mask].reset_index(drop=True)
                                    ).where(
                                        _safe_numeric(utility_soft.loc[valid_mask].reset_index(drop=True))
                                        >= float(spec.oracle_gate)
                                    )
                                else:
                                    raw_score = base_scores[spec.selector]
                                score = _apply_bad_and_floor(
                                    raw_score,
                                    bad_proxy=bad_proxy,
                                    bad_threshold=spec.bad_threshold,
                                    score_floor=spec.score_floor,
                                )
                                score = _run_entry_score(
                                    valid_frame,
                                    score,
                                    gap_hours=float(spec.run_entry_gap_hours),
                                )
                                diag = {
                                    "proxy_features": (
                                        "oracle"
                                        if spec.selector == "oracle_ceiling"
                                        else ",".join(selector_feature_diags[spec.selector].get("features", []))
                                    ),
                                    "soft_proxy_features": ",".join(soft_diag.get("features", [])),
                                    "hard_proxy_features": ",".join(hard_diag.get("features", [])),
                                    "contrast_proxy_features": ",".join(contrast_diag.get("features", [])),
                                    "bad_proxy_features": ",".join(bad_diag.get("features", [])),
                                    "cleanft_proxy_features": ",".join(cleanft_diag.get("features", [])),
                                    "early_adverse_proxy_features": ",".join(
                                        early_adverse_diag.get("features", [])
                                    ),
                                    "slow_timeout_proxy_features": ",".join(slow_timeout_diag.get("features", [])),
                                    "path_dirty_proxy_features": ",".join(path_dirty_diag.get("features", [])),
                                }
                                m_rows, w_rows = _period_rows(
                                    valid_frame=valid_frame,
                                    valid_metrics=valid_metrics,
                                    valid_target=valid_target_by_top_k[int(spec.top_k)],
                                    oracle_mask=valid_oracle_masks[int(spec.top_k)],
                                    score=score,
                                    spec=spec,
                                    month=str(month),
                                    diag=diag,
                                )
                                monthly_rows.extend(m_rows)
                                weekly_rows.extend(w_rows)

    monthly = pd.DataFrame(monthly_rows)
    weekly = pd.DataFrame(weekly_rows)
    proxy_diagnostics = pd.DataFrame(proxy_diag_rows)
    fit_holdout = _fit_holdout_summary(
        monthly=monthly,
        weekly=weekly,
        fit_months=fit_months,
        holdout_month=holdout_month,
        min_week_rows=min_week_rows,
    )
    selected_by_fit = _select_by_fit(fit_holdout)

    paths = {
        "monthly": output_dir / "gated_oracle_recoverability_monthly.csv",
        "weekly": output_dir / "gated_oracle_recoverability_weekly.csv",
        "fit_holdout": output_dir / "gated_oracle_recoverability_fit_holdout.csv",
        "selected_by_fit": output_dir / "gated_oracle_recoverability_selected_by_fit.csv",
        "proxy_diagnostics": output_dir / "gated_oracle_recoverability_proxy_diagnostics.csv",
        "manifest": output_dir / "manifest.json",
    }
    monthly.to_csv(paths["monthly"], index=False)
    weekly.to_csv(paths["weekly"], index=False)
    fit_holdout.to_csv(paths["fit_holdout"], index=False)
    selected_by_fit.to_csv(paths["selected_by_fit"], index=False)
    proxy_diagnostics.to_csv(paths["proxy_diagnostics"], index=False)

    manifest = {
        "scope": "proxy_only_gated_oracle_recoverability",
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
        "utility_targets": list(utility_targets),
        "oracle_gates": [float(v) for v in oracle_gates],
        "proxy_methods": list(proxy_methods),
        "proxy_top_ks": [int(v) for v in proxy_top_ks],
        "selectors": list(selectors),
        "top_ks": [int(v) for v in top_ks],
        "bad_thresholds": [float(v) for v in bad_thresholds],
        "score_floors": [float(v) for v in score_floors],
        "run_entry_gap_hours": [float(v) for v in run_entry_gap_hours],
        "fit_months": [str(v) for v in fit_months],
        "holdout_month": str(holdout_month),
        "min_week_rows": int(min_week_rows),
        "include_causal_outcome_priors": bool(include_causal_outcome_priors),
        "include_causal_state_path_priors": bool(include_causal_state_path_priors),
        "include_event_confirmation_features": bool(include_event_confirmation_features),
        "include_adverse_path_composites": bool(include_adverse_path_composites),
        "prior_windows_days": [float(v) for v in prior_windows_days],
        "prior_embargo_hours": float(prior_embargo_hours),
        "rows_monthly": int(len(monthly)),
        "rows_weekly": int(len(weekly)),
        "fit_sign_pass_rows": int(fit_holdout["fit_sign_pass"].sum()) if not fit_holdout.empty else 0,
        "holdout_sign_pass_rows": int(fit_holdout["holdout_sign_pass"].sum()) if not fit_holdout.empty else 0,
        "fit_bounded_pass_rows": int(fit_holdout["fit_bounded_pass"].sum()) if not fit_holdout.empty else 0,
        "holdout_bounded_pass_rows": int(fit_holdout["holdout_bounded_pass"].sum()) if not fit_holdout.empty else 0,
        "positive_dirty_holdout_rows": int(fit_holdout["positive_dirty_holdout"].sum()) if not fit_holdout.empty else 0,
        "selected_by_fit_rows": int(len(selected_by_fit)),
        "selected_by_fit_holdout_bounded_rows": int(selected_by_fit["holdout_bounded_pass"].sum())
        if not selected_by_fit.empty
        else 0,
        "outputs": {key: str(value) for key, value in paths.items()},
        **reports,
    }
    markdown = _write_markdown(
        output_dir=output_dir,
        fit_holdout=fit_holdout,
        selected_by_fit=selected_by_fit,
        monthly=monthly,
        proxy_diagnostics=proxy_diagnostics,
        manifest=manifest,
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
    parser.add_argument("--utility-targets", default=",".join(DEFAULT_UTILITY_TARGETS))
    parser.add_argument("--oracle-gates", type=_parse_float_csv, default=",".join(str(v) for v in DEFAULT_ORACLE_GATES))
    parser.add_argument("--proxy-methods", type=lambda value: _parse_csv(value, DEFAULT_PROXY_METHODS), default=",".join(DEFAULT_PROXY_METHODS))
    parser.add_argument("--proxy-top-ks", type=lambda value: [int(v) for v in _parse_csv(value)], default=",".join(str(v) for v in DEFAULT_PROXY_TOP_KS))
    parser.add_argument(
        "--selectors",
        type=lambda value: _parse_csv(value, DEFAULT_SCORE_FLOOR_SELECTORS),
        default=",".join(DEFAULT_SCORE_FLOOR_SELECTORS),
    )
    parser.add_argument("--top-ks", type=lambda value: [int(v) for v in _parse_csv(value)], default=",".join(str(v) for v in DEFAULT_TOP_KS))
    parser.add_argument("--bad-thresholds", type=_parse_float_csv, default=",".join(str(v) for v in DEFAULT_BAD_THRESHOLDS))
    parser.add_argument("--score-floors", type=_parse_float_csv, default=",".join(str(v) for v in DEFAULT_SCORE_FLOORS))
    parser.add_argument(
        "--run-entry-gap-hours",
        type=_parse_float_csv,
        default=",".join(str(v) for v in DEFAULT_RUN_ENTRY_GAP_HOURS),
    )
    parser.add_argument("--fit-months", type=lambda value: _parse_csv(value, DEFAULT_FIT_MONTHS), default=",".join(DEFAULT_FIT_MONTHS))
    parser.add_argument("--holdout-month", default=DEFAULT_HOLDOUT_MONTH)
    parser.add_argument("--min-week-rows", type=int, default=3)
    parser.add_argument("--include-causal-outcome-priors", action="store_true")
    parser.add_argument("--include-causal-state-path-priors", action="store_true")
    parser.add_argument("--include-event-confirmation-features", action="store_true")
    parser.add_argument("--include-adverse-path-composites", action="store_true")
    parser.add_argument(
        "--prior-windows-days",
        type=_parse_float_csv,
        default=",".join(str(v) for v in DEFAULT_PRIOR_WINDOWS_DAYS),
    )
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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = run_ablation(
        labels_path=args.labels_path,
        output_dir=args.output_dir,
        feature_dir=args.feature_dir,
        feature_list_csv=args.feature_list_csv,
        max_feature_store_features=args.max_feature_store_features,
        utility_targets=_parse_csv(args.utility_targets, DEFAULT_UTILITY_TARGETS),
        oracle_gates=[float(v) for v in args.oracle_gates],
        proxy_methods=list(args.proxy_methods),
        proxy_top_ks=[int(v) for v in args.proxy_top_ks],
        selectors=list(args.selectors),
        top_ks=[int(v) for v in args.top_ks],
        bad_thresholds=[float(v) for v in args.bad_thresholds],
        score_floors=[float(v) for v in args.score_floors],
        run_entry_gap_hours=[float(v) for v in args.run_entry_gap_hours],
        fit_months=list(args.fit_months),
        holdout_month=str(args.holdout_month),
        min_week_rows=int(args.min_week_rows),
        include_causal_outcome_priors=bool(args.include_causal_outcome_priors),
        include_causal_state_path_priors=bool(args.include_causal_state_path_priors),
        include_event_confirmation_features=bool(args.include_event_confirmation_features),
        include_adverse_path_composites=bool(args.include_adverse_path_composites),
        prior_windows_days=[float(v) for v in args.prior_windows_days],
        prior_embargo_hours=float(args.prior_embargo_hours),
        state_path_prior_features=list(args.state_path_prior_features),
        event_feature_store_features=list(args.event_feature_store_features),
    )
    summary_keys = [
        "output_dir",
        "rows",
        "feature_count",
        "utility_targets",
        "oracle_gates",
        "proxy_methods",
        "proxy_top_ks",
        "selectors",
        "top_ks",
        "run_entry_gap_hours",
        "rows_monthly",
        "rows_weekly",
        "fit_sign_pass_rows",
        "holdout_sign_pass_rows",
        "fit_bounded_pass_rows",
        "holdout_bounded_pass_rows",
        "positive_dirty_holdout_rows",
        "selected_by_fit_rows",
        "selected_by_fit_holdout_bounded_rows",
        "outputs",
    ]
    print(json.dumps(_json_safe({key: manifest.get(key) for key in summary_keys}), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
