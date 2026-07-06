#!/usr/bin/env python3
"""First-touch execution proxy ablation before base/meta training.

This diagnostic stays upstream of model training. It asks whether causal
feature-rank proxies can recover rows that are profitable after costs and have
clean first-touch execution: fast edge, limited adverse excursion, no same-bar
ambiguity, and no timeout-heavy path.
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

from scripts.run_label_dual_proxy_path_risk_ablation import (  # noqa: E402
    DEFAULT_FEATURE_DIR,
    DEFAULT_FEATURE_LIST_CSV,
    DEFAULT_LABELS_DIR,
    _feature_columns,
    _json_safe,
    _load_feature_store_columns,
    _load_labels,
    _parse_csv,
    _parse_float_csv,
    _path_metrics,
    _proxy_score,
    _safe_max,
    _safe_mean,
    _safe_min,
    _safe_numeric,
    _safe_quantile,
    _selection_metrics_from_indices,
    _sigmoid,
    _spearman,
    _timestamp_top_indices,
    _weighted_mean,
    _read_feature_list,
)
from scripts.run_label_quality_proxy_diagnostics import _rank_top_indices  # noqa: E402


DEFAULT_OUTPUT_DIR = Path("data_perp/reports/label_first_touch_execution_proxy_ablation_v1")
DEFAULT_TOP_FRACS = (0.01, 0.03, 0.05)
DEFAULT_FIT_MONTHS = ("2026-04", "2026-05")
DEFAULT_HOLDOUT_MONTH = "2026-06"


@dataclass(frozen=True)
class ExecScoreSpec:
    name: str
    utility_weight: float = 0.0
    clean_weight: float = 0.0
    fast_edge_weight: float = 0.0
    early_adverse_weight: float = 0.0
    slow_timeout_weight: float = 0.0
    dirty_weight: float = 0.0
    max_dirty_proxy: float | None = None
    max_early_adverse_proxy: float | None = None
    max_slow_timeout_proxy: float | None = None


SCORE_SPECS = (
    ExecScoreSpec("F0_utility_proxy", utility_weight=1.0),
    ExecScoreSpec("F1_clean_first_touch_direct", clean_weight=1.0),
    ExecScoreSpec("F2_fast_edge_direct", fast_edge_weight=1.0),
    ExecScoreSpec("F3_utility_minus_early_adverse025", utility_weight=1.0, early_adverse_weight=0.25),
    ExecScoreSpec("F4_utility_minus_early_adverse050", utility_weight=1.0, early_adverse_weight=0.50),
    ExecScoreSpec("F5_utility_minus_slow_timeout", utility_weight=1.0, slow_timeout_weight=0.35),
    ExecScoreSpec("F6_utility_minus_dirty", utility_weight=1.0, dirty_weight=0.50),
    ExecScoreSpec("F7_utility_clean_minus_dirty", utility_weight=0.50, clean_weight=0.50, dirty_weight=0.50),
    ExecScoreSpec("F8_utility_dirty_gate35", utility_weight=1.0, max_dirty_proxy=0.35),
    ExecScoreSpec("F9_clean_dirty_gate35", clean_weight=1.0, max_dirty_proxy=0.35),
    ExecScoreSpec("F10_fast_edge_dirty_gate35", fast_edge_weight=1.0, max_dirty_proxy=0.35),
    ExecScoreSpec(
        "F11_utility_exec_gates50",
        utility_weight=1.0,
        max_dirty_proxy=0.50,
        max_early_adverse_proxy=0.50,
        max_slow_timeout_proxy=0.50,
    ),
    ExecScoreSpec(
        "F12_clean_utility_exec_gates50",
        utility_weight=0.50,
        clean_weight=0.50,
        max_dirty_proxy=0.50,
        max_early_adverse_proxy=0.50,
        max_slow_timeout_proxy=0.50,
    ),
)


def _first_touch_metrics(frame: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    out = metrics.copy()
    out["first_touch_hit"] = _safe_numeric(frame.get("__first_touch_hit__", out["y_bin"])).fillna(0.0).clip(0.0, 1.0)
    out["first_touch_stop"] = _safe_numeric(frame.get("__first_touch_stop__")).fillna(0.0).clip(0.0, 1.0)
    out["first_touch_timeout"] = _safe_numeric(
        frame.get("__first_touch_timeout__", out["is_timeout"].astype(float))
    ).fillna(0.0).clip(0.0, 1.0)
    out["first_touch_eligible"] = _safe_numeric(frame.get("__first_touch_eligible__")).fillna(1.0).clip(0.0, 1.0)
    out["first_touch_valid_path"] = _safe_numeric(frame.get("__first_touch_valid_path__")).fillna(1.0).clip(0.0, 1.0)
    out["first_touch_net_positive"] = _safe_numeric(
        frame.get("__first_touch_net_positive__", out["u_policy_net"] > 0.0)
    ).fillna(0.0).clip(0.0, 1.0)
    out["first_touch_same_bar"] = _safe_numeric(frame.get("__first_touch_same_bar_both__")).fillna(0.0).clip(0.0, 1.0)
    out["first_touch_bar"] = _safe_numeric(frame.get("__first_touch_bar__")).fillna(out["bars_policy"]).fillna(36.0).clip(
        lower=0.0
    )
    out["first_touch_mae_to_sl"] = _safe_numeric(frame.get("__first_touch_mae_to_sl__")).fillna(out["mae_norm"]).clip(
        lower=0.0
    )
    out["first_touch_mfe_to_tp"] = _safe_numeric(frame.get("__first_touch_mfe_to_tp__")).fillna(out["mfe_norm"]).clip(
        lower=0.0
    )
    out["clean_exec_actual"] = (
        (out["u_policy_net"] > 0.0)
        & (out["first_touch_hit"] > 0.5)
        & (out["first_touch_stop"] <= 0.5)
        & (out["first_touch_timeout"] <= 0.5)
        & (out["first_touch_same_bar"] <= 0.5)
        & (out["first_touch_mae_to_sl"] <= 1.0)
        & (out["first_touch_bar"] <= 12.0)
        & (out["barrier"] <= 0.025)
    ).astype(float)
    return out


def _target_components(metrics: pd.DataFrame) -> dict[str, pd.Series]:
    u = _safe_numeric(metrics["u_policy_net"]).fillna(-0.02)
    barrier = _safe_numeric(metrics["barrier"]).fillna(0.0)
    hit = _safe_numeric(metrics["first_touch_hit"]).fillna(0.0).clip(0.0, 1.0)
    stop = _safe_numeric(metrics["first_touch_stop"]).fillna(0.0).clip(0.0, 1.0)
    timeout = _safe_numeric(metrics["first_touch_timeout"]).fillna(0.0).clip(0.0, 1.0)
    same_bar = _safe_numeric(metrics["first_touch_same_bar"]).fillna(0.0).clip(0.0, 1.0)
    bar = _safe_numeric(metrics["first_touch_bar"]).fillna(36.0).clip(lower=0.0)
    mae_to_sl = _safe_numeric(metrics["first_touch_mae_to_sl"]).fillna(10.0).clip(lower=0.0)
    mfe_to_tp = _safe_numeric(metrics["first_touch_mfe_to_tp"]).fillna(0.0).clip(lower=0.0)

    utility = pd.Series(_sigmoid((u - 0.0015) / 0.008), index=metrics.index).clip(0.0, 1.0)
    fast_edge = (
        hit
        * pd.Series(_sigmoid((10.0 - bar) / 3.5), index=metrics.index)
        * pd.Series(_sigmoid((mfe_to_tp - 1.0) / 0.25), index=metrics.index)
    ).clip(0.0, 1.0)
    early_adverse = pd.concat(
        [
            stop,
            same_bar,
            pd.Series(_sigmoid((mae_to_sl - 0.85) / 0.18), index=metrics.index),
        ],
        axis=1,
    ).max(axis=1).clip(0.0, 1.0)
    slow_timeout = pd.concat(
        [
            timeout,
            pd.Series(_sigmoid((bar - 14.0) / 4.0), index=metrics.index),
        ],
        axis=1,
    ).max(axis=1).clip(0.0, 1.0)
    wide_barrier = pd.Series(_sigmoid((barrier - 0.025) / 0.006), index=metrics.index).clip(0.0, 1.0)
    dirty = pd.concat(
        [
            early_adverse,
            slow_timeout,
            wide_barrier,
        ],
        axis=1,
    ).max(axis=1).clip(0.0, 1.0)
    clean_envelope = (
        hit
        * (1.0 - stop)
        * (1.0 - timeout)
        * (1.0 - same_bar)
        * pd.Series(_sigmoid((0.85 - mae_to_sl) / 0.18), index=metrics.index)
        * pd.Series(_sigmoid((0.025 - barrier) / 0.006), index=metrics.index)
        * (0.40 + 0.60 * pd.Series(_sigmoid((12.0 - bar) / 4.0), index=metrics.index))
    ).clip(0.0, 1.0)
    clean_first_touch = (utility * clean_envelope).clip(0.0, 1.0)
    return {
        "utility": utility,
        "clean_first_touch": clean_first_touch,
        "fast_edge": fast_edge,
        "early_adverse": early_adverse,
        "slow_timeout": slow_timeout,
        "dirty": dirty,
    }


def _score_from_components(spec: ExecScoreSpec, proxies: dict[str, pd.Series]) -> pd.Series:
    score = (
        float(spec.utility_weight) * _safe_numeric(proxies["utility"])
        + float(spec.clean_weight) * _safe_numeric(proxies["clean_first_touch"])
        + float(spec.fast_edge_weight) * _safe_numeric(proxies["fast_edge"])
        - float(spec.early_adverse_weight) * _safe_numeric(proxies["early_adverse"])
        - float(spec.slow_timeout_weight) * _safe_numeric(proxies["slow_timeout"])
        - float(spec.dirty_weight) * _safe_numeric(proxies["dirty"])
    )
    mask = pd.Series(True, index=score.index)
    if spec.max_dirty_proxy is not None:
        mask &= _safe_numeric(proxies["dirty"]) <= float(spec.max_dirty_proxy)
    if spec.max_early_adverse_proxy is not None:
        mask &= _safe_numeric(proxies["early_adverse"]) <= float(spec.max_early_adverse_proxy)
    if spec.max_slow_timeout_proxy is not None:
        mask &= _safe_numeric(proxies["slow_timeout"]) <= float(spec.max_slow_timeout_proxy)
    return score.where(mask)


def _target_for_selection(components: dict[str, pd.Series], index: pd.Index) -> pd.DataFrame:
    soft = components["clean_first_touch"].reindex(index).clip(0.0, 1.0)
    hard = (soft >= 0.50).astype(float)
    return pd.DataFrame({"target_soft": soft, "target_hard": hard}, index=index)


def _indices_by_mode(frame: pd.DataFrame, score: pd.Series, top_frac: float, selection_mode: str) -> np.ndarray:
    if str(selection_mode) == "timestamp":
        return _timestamp_top_indices(frame, score, top_frac)
    return _rank_top_indices(score, top_frac)


def _selection_metrics_ext(
    *,
    frame: pd.DataFrame,
    metrics: pd.DataFrame,
    target: pd.DataFrame,
    score: pd.Series,
    arm: str,
    selector: str,
    period: str,
    top_frac: float,
    selection_mode: str,
) -> dict[str, Any]:
    frame = frame.reset_index(drop=True)
    metrics = metrics.reset_index(drop=True)
    target = target.reset_index(drop=True)
    score = _safe_numeric(score).reset_index(drop=True)
    idx = _indices_by_mode(frame, score, top_frac, selection_mode)
    row = _selection_metrics_from_indices(
        frame=frame,
        metrics=metrics,
        target=target,
        idx=idx,
        arm=arm,
        selector=selector,
        period=period,
        top_frac=top_frac,
    )
    selected = metrics.iloc[idx] if len(idx) else metrics.iloc[:0]
    row.update(
        {
            "first_touch_hit_rate": _safe_mean(selected["first_touch_hit"]),
            "first_touch_stop_rate": _safe_mean(selected["first_touch_stop"]),
            "first_touch_timeout_rate": _safe_mean(selected["first_touch_timeout"]),
            "first_touch_same_bar_rate": _safe_mean(selected["first_touch_same_bar"]),
            "first_touch_valid_path_rate": _safe_mean(selected["first_touch_valid_path"]),
            "first_touch_net_positive_rate": _safe_mean(selected["first_touch_net_positive"]),
            "clean_exec_actual_rate": _safe_mean(selected["clean_exec_actual"]),
            "mean_first_touch_bar": _safe_mean(selected["first_touch_bar"]),
            "p90_first_touch_bar": _safe_quantile(selected["first_touch_bar"], 0.90),
            "mean_first_touch_mae_to_sl": _safe_mean(selected["first_touch_mae_to_sl"]),
            "p90_first_touch_mae_to_sl": _safe_quantile(selected["first_touch_mae_to_sl"], 0.90),
            "first_touch_bad_mae_to_sl_rate": _safe_mean(selected["first_touch_mae_to_sl"] >= 1.0),
            "mean_first_touch_mfe_to_tp": _safe_mean(selected["first_touch_mfe_to_tp"]),
        }
    )
    return row


def _diag_columns(component_diag: dict[str, dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, diag in component_diag.items():
        out[f"{name}_proxy_top_abs_ic"] = diag.get("top_abs_ic")
        out[f"{name}_proxy_mean_top_abs_ic"] = diag.get("mean_top_abs_ic")
        out[f"{name}_proxy_features"] = ",".join(str(v) for v in diag.get("features", []))
    return out


def _monthly_weekly_rows(
    *,
    valid_frame: pd.DataFrame,
    valid_metrics: pd.DataFrame,
    valid_target: pd.DataFrame,
    score: pd.Series,
    score_arm: str,
    month: str,
    top_fracs: list[float],
    component_diag: dict[str, dict[str, Any]],
    selection_mode: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    monthly_rows: list[dict[str, Any]] = []
    weekly_rows: list[dict[str, Any]] = []
    score_reset = score.reset_index(drop=True)
    frame_reset = valid_frame.reset_index(drop=True)
    metrics_reset = valid_metrics.reset_index(drop=True)
    target_reset = valid_target.reset_index(drop=True)
    selector = f"first_touch_exec_proxy_oos_{selection_mode}"
    for frac in top_fracs:
        row = _selection_metrics_ext(
            frame=frame_reset,
            metrics=metrics_reset,
            target=target_reset,
            score=score_reset,
            arm=score_arm,
            selector=selector,
            period=str(month),
            top_frac=float(frac),
            selection_mode=selection_mode,
        )
        row.update(_diag_columns(component_diag))
        monthly_rows.append(row)

        weeks = frame_reset["__ts__"].dt.to_period("W-SUN").astype(str)
        for week, ids in pd.Series(np.arange(len(frame_reset)), index=frame_reset.index).groupby(weeks, dropna=False):
            pos = ids.to_numpy(dtype=np.int64)
            if len(pos) < 20:
                continue
            week_row = _selection_metrics_ext(
                frame=frame_reset.iloc[pos].reset_index(drop=True),
                metrics=metrics_reset.iloc[pos].reset_index(drop=True),
                target=target_reset.iloc[pos].reset_index(drop=True),
                score=score_reset.iloc[pos].reset_index(drop=True),
                arm=score_arm,
                selector=selector,
                period=str(month),
                top_frac=float(frac),
                selection_mode=selection_mode,
            )
            week_row["week"] = str(week)
            week_row["week_selected_rows"] = int(week_row["selected_rows"])
            week_row["week_selected_share"] = float(week_row["selected_rows"] / len(pos)) if len(pos) else float("nan")
            week_row.update(_diag_columns(component_diag))
            weekly_rows.append(week_row)
    return monthly_rows, weekly_rows


def _weighted_metric(frame: pd.DataFrame, value_col: str, weight_col: str = "selected_rows") -> float:
    if value_col not in frame.columns or weight_col not in frame.columns:
        return float("nan")
    return _weighted_mean(frame, value_col, weight_col)


def _summarize_month(prefix: str, frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {
            f"{prefix}_months": 0,
            f"{prefix}_positive_months": 0,
            f"{prefix}_mean_month_u": float("nan"),
            f"{prefix}_worst_month_u": float("nan"),
            f"{prefix}_selected_rows": 0,
            f"{prefix}_bad_mae_1r_rate": float("nan"),
            f"{prefix}_p90_mae_norm": float("nan"),
            f"{prefix}_clean_exec_actual_rate": float("nan"),
            f"{prefix}_first_touch_stop_rate": float("nan"),
            f"{prefix}_first_touch_timeout_rate": float("nan"),
            f"{prefix}_first_touch_same_bar_rate": float("nan"),
            f"{prefix}_first_touch_bad_mae_to_sl_rate": float("nan"),
            f"{prefix}_p90_first_touch_mae_to_sl": float("nan"),
            f"{prefix}_p90_first_touch_bar": float("nan"),
        }
    mean_u = _safe_numeric(frame["mean_u"])
    return {
        f"{prefix}_months": int(frame["period"].astype(str).nunique()),
        f"{prefix}_positive_months": int(mean_u.gt(0.0).sum()),
        f"{prefix}_mean_month_u": _safe_mean(mean_u),
        f"{prefix}_worst_month_u": _safe_min(mean_u),
        f"{prefix}_selected_rows": int(_safe_numeric(frame["selected_rows"]).sum()),
        f"{prefix}_bad_mae_1r_rate": _weighted_metric(frame, "bad_mae_1r_rate"),
        f"{prefix}_p90_mae_norm": _weighted_metric(frame, "p90_mae_norm"),
        f"{prefix}_wide_25bps_rate": _weighted_metric(frame, "wide_barrier_25bps_rate"),
        f"{prefix}_timeout_rate": _weighted_metric(frame, "timeout_rate"),
        f"{prefix}_clean_exec_actual_rate": _weighted_metric(frame, "clean_exec_actual_rate"),
        f"{prefix}_first_touch_hit_rate": _weighted_metric(frame, "first_touch_hit_rate"),
        f"{prefix}_first_touch_stop_rate": _weighted_metric(frame, "first_touch_stop_rate"),
        f"{prefix}_first_touch_timeout_rate": _weighted_metric(frame, "first_touch_timeout_rate"),
        f"{prefix}_first_touch_same_bar_rate": _weighted_metric(frame, "first_touch_same_bar_rate"),
        f"{prefix}_first_touch_valid_path_rate": _weighted_metric(frame, "first_touch_valid_path_rate"),
        f"{prefix}_first_touch_bad_mae_to_sl_rate": _weighted_metric(frame, "first_touch_bad_mae_to_sl_rate"),
        f"{prefix}_p90_first_touch_mae_to_sl": _weighted_metric(frame, "p90_first_touch_mae_to_sl"),
        f"{prefix}_p90_first_touch_bar": _weighted_metric(frame, "p90_first_touch_bar"),
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
            f"{prefix}_weekly_clean_exec_actual_rate": float("nan"),
        }
    mean_u = _safe_numeric(frame["mean_u"])
    week_rows = _safe_numeric(frame["week_selected_rows"]).fillna(0.0)
    material = week_rows >= int(min_week_rows)
    positive = mean_u > 0.0
    return {
        f"{prefix}_weeks": int(len(frame)),
        f"{prefix}_material_weeks": int(material.sum()),
        f"{prefix}_material_positive_week_rate": float((positive & material).sum() / material.sum())
        if int(material.sum())
        else float("nan"),
        f"{prefix}_q25_week_u": _safe_quantile(mean_u, 0.25),
        f"{prefix}_worst_week_u": _safe_min(mean_u),
        f"{prefix}_weekly_bad_mae_1r_rate": _weighted_metric(frame, "bad_mae_1r_rate", "week_selected_rows"),
        f"{prefix}_weekly_clean_exec_actual_rate": _weighted_metric(
            frame,
            "clean_exec_actual_rate",
            "week_selected_rows",
        ),
        f"{prefix}_weekly_first_touch_stop_rate": _weighted_metric(
            frame,
            "first_touch_stop_rate",
            "week_selected_rows",
        ),
        f"{prefix}_weekly_first_touch_timeout_rate": _weighted_metric(
            frame,
            "first_touch_timeout_rate",
            "week_selected_rows",
        ),
    }


def _fit_holdout_summary(
    *,
    monthly: pd.DataFrame,
    weekly: pd.DataFrame,
    fit_months: list[str],
    holdout_month: str,
    min_week_rows: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (arm, frac), group in monthly.groupby(["arm", "top_frac"], observed=True, dropna=False):
        week_group = weekly[
            weekly["arm"].astype(str).eq(str(arm))
            & _safe_numeric(weekly["top_frac"]).eq(float(frac))
        ].copy()
        fit_month = group[group["period"].astype(str).isin(fit_months)].copy()
        holdout_monthly = group[group["period"].astype(str).eq(str(holdout_month))].copy()
        fit_week = week_group[week_group["period"].astype(str).isin(fit_months)].copy()
        holdout_week = week_group[week_group["period"].astype(str).eq(str(holdout_month))].copy()
        if fit_month.empty or holdout_monthly.empty:
            continue
        row: dict[str, Any] = {"score_arm": str(arm), "top_frac": float(frac)}
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
        fit_clean = (
            fit_sign
            and row["fit_clean_exec_actual_rate"] >= 0.30
            and row["fit_first_touch_timeout_rate"] <= 0.40
            and row["fit_first_touch_same_bar_rate"] <= 0.05
            and row["fit_first_touch_bad_mae_to_sl_rate"] <= 0.40
            and row["fit_p90_first_touch_mae_to_sl"] <= 1.25
        )
        holdout_clean_standalone = (
            holdout_sign
            and row["holdout_clean_exec_actual_rate"] >= 0.30
            and row["holdout_first_touch_timeout_rate"] <= 0.40
            and row["holdout_first_touch_same_bar_rate"] <= 0.05
            and row["holdout_first_touch_bad_mae_to_sl_rate"] <= 0.40
            and row["holdout_p90_first_touch_mae_to_sl"] <= 1.25
        )
        fit_bounded = (
            fit_sign
            and row["fit_clean_exec_actual_rate"] >= 0.20
            and row["fit_first_touch_timeout_rate"] <= 0.55
            and row["fit_first_touch_same_bar_rate"] <= 0.10
            and row["fit_first_touch_bad_mae_to_sl_rate"] <= 0.60
            and row["fit_p90_first_touch_mae_to_sl"] <= 2.0
        )
        holdout_bounded_standalone = (
            holdout_sign
            and row["holdout_clean_exec_actual_rate"] >= 0.20
            and row["holdout_first_touch_timeout_rate"] <= 0.55
            and row["holdout_first_touch_same_bar_rate"] <= 0.10
            and row["holdout_first_touch_bad_mae_to_sl_rate"] <= 0.60
            and row["holdout_p90_first_touch_mae_to_sl"] <= 2.0
        )
        row["fit_sign_pass"] = bool(fit_sign)
        row["holdout_sign_pass"] = bool(holdout_sign)
        row["fit_clean_pass"] = bool(fit_clean)
        row["holdout_clean_standalone_pass"] = bool(holdout_clean_standalone)
        row["holdout_clean_pass"] = bool(fit_clean and holdout_clean_standalone)
        row["fit_bounded_pass"] = bool(fit_bounded)
        row["holdout_bounded_standalone_pass"] = bool(holdout_bounded_standalone)
        row["holdout_bounded_pass"] = bool(fit_bounded and holdout_bounded_standalone)
        row["positive_dirty_holdout"] = bool(holdout_sign and not holdout_bounded_standalone)
        row["exec_risk_score"] = float(
            (row["holdout_mean_month_u"] if pd.notna(row["holdout_mean_month_u"]) else 0.0)
            + 0.50 * (row["holdout_q25_week_u"] if pd.notna(row["holdout_q25_week_u"]) else 0.0)
            + 0.010
            * (row["holdout_clean_exec_actual_rate"] if pd.notna(row["holdout_clean_exec_actual_rate"]) else 0.0)
            - 0.020
            * (
                row["holdout_first_touch_bad_mae_to_sl_rate"]
                if pd.notna(row["holdout_first_touch_bad_mae_to_sl_rate"])
                else 0.0
            )
            - 0.002
            * (
                row["holdout_p90_first_touch_mae_to_sl"]
                if pd.notna(row["holdout_p90_first_touch_mae_to_sl"])
                else 0.0
            )
            - 0.010
            * (row["holdout_first_touch_timeout_rate"] if pd.notna(row["holdout_first_touch_timeout_rate"]) else 0.0)
        )
        rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(
        ["holdout_clean_pass", "holdout_bounded_pass", "positive_dirty_holdout", "exec_risk_score"],
        ascending=[False, False, False, False],
    )


def _format_table(frame: pd.DataFrame, cols: list[str], limit: int = 30) -> str:
    if frame.empty:
        return "No rows."
    view = frame[[col for col in cols if col in frame.columns]].head(limit).copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda value: f"{float(value):.4f}" if pd.notna(value) else "")
    return view.to_markdown(index=False)


def _write_markdown(
    output_dir: Path,
    fit_holdout: pd.DataFrame,
    components_summary: pd.DataFrame,
    manifest: dict[str, Any],
) -> Path:
    path = output_dir / "first_touch_execution_proxy_ablation.md"
    cols = [
        "score_arm",
        "top_frac",
        "exec_risk_score",
        "fit_sign_pass",
        "fit_clean_pass",
        "fit_bounded_pass",
        "fit_mean_month_u",
        "fit_worst_month_u",
        "fit_material_positive_week_rate",
        "fit_bad_mae_1r_rate",
        "fit_p90_mae_norm",
        "fit_clean_exec_actual_rate",
        "fit_first_touch_timeout_rate",
        "fit_first_touch_bad_mae_to_sl_rate",
        "fit_p90_first_touch_mae_to_sl",
        "holdout_clean_pass",
        "holdout_bounded_pass",
        "holdout_mean_month_u",
        "holdout_material_positive_week_rate",
        "holdout_bad_mae_1r_rate",
        "holdout_p90_mae_norm",
        "holdout_clean_exec_actual_rate",
        "holdout_first_touch_timeout_rate",
        "holdout_first_touch_bad_mae_to_sl_rate",
        "holdout_p90_first_touch_mae_to_sl",
    ]
    component_cols = [
        "period",
        "component",
        "proxy_top_abs_ic",
        "proxy_mean_top_abs_ic",
        "proxy_ic_actual_u",
        "proxy_ic_actual_clean_exec",
        "proxy_ic_actual_dirty",
        "proxy_ic_actual_fast_edge",
        "proxy_features",
    ]
    clean_pass = fit_holdout[fit_holdout["holdout_clean_pass"].eq(True)] if not fit_holdout.empty else fit_holdout
    bounded_pass = fit_holdout[fit_holdout["holdout_bounded_pass"].eq(True)] if not fit_holdout.empty else fit_holdout
    positive_dirty = (
        fit_holdout[fit_holdout["positive_dirty_holdout"].eq(True)].sort_values(
            "holdout_mean_month_u",
            ascending=False,
        )
        if not fit_holdout.empty
        else fit_holdout
    )
    best = fit_holdout.sort_values("exec_risk_score", ascending=False) if not fit_holdout.empty else fit_holdout
    comp = (
        components_summary[components_summary["period"].astype(str).isin(manifest["fit_months"] + [manifest["holdout_month"]])]
        .sort_values(["period", "component"])
        if not components_summary.empty
        else components_summary
    )
    lines = [
        "# First-Touch Execution Proxy Label Ablation",
        "",
        "Scope: proxy tests only. No LightGBM, Optuna, policy geometry, or base/meta training is run.",
        "",
        f"Labels: `{manifest['labels_path']}`",
        f"Feature dir: `{manifest['feature_dir']}`",
        f"Feature count: `{manifest['feature_count']}`",
        f"Fit months: `{','.join(manifest['fit_months'])}`",
        f"Holdout month: `{manifest['holdout_month']}`",
        f"Selection mode: `{manifest['selection_mode']}`",
        f"Top fractions: `{','.join(str(v) for v in manifest['top_fracs'])}`",
        "",
        "Each arm is built from prior-month feature-rank proxies for utility, clean first-touch execution, fast edge, early adverse risk, slow/timeout risk, and combined dirty execution.",
        "Timestamp selection mode chooses the top fraction inside each timestamp bucket before aggregating monthly/weekly economics.",
        "",
        "Clean pass requires positive Apr-May fit and June holdout, clean-exec rate at least `30%`, first-touch timeout rate at most `40%`, same-bar rate at most `5%`, first-touch bad-MAE-to-SL rate at most `40%`, and p90 first-touch MAE-to-SL at most `1.25R`.",
        "Bounded pass relaxes those first-touch execution limits but still requires positive fit and holdout. Full-horizon MAE remains in the tables as an audit field, not as the first-touch execution gate.",
        "",
        "## Counts",
        "",
        f"- Monthly rows: `{manifest['rows_monthly']}`",
        f"- Weekly rows: `{manifest['rows_weekly']}`",
        f"- Fit clean pass: `{manifest['fit_clean_pass_rows']}`",
        f"- Holdout clean pass after fit selection: `{manifest['holdout_clean_pass_rows']}`",
        f"- Fit bounded pass: `{manifest['fit_bounded_pass_rows']}`",
        f"- Holdout bounded pass after fit selection: `{manifest['holdout_bounded_pass_rows']}`",
        f"- Positive but economically dirty holdout: `{manifest['positive_dirty_holdout_rows']}`",
        "",
        "## Clean Holdout Passes",
        "",
        _format_table(clean_pass, cols, limit=40),
        "",
        "## Bounded Holdout Passes",
        "",
        _format_table(bounded_pass, cols, limit=40),
        "",
        "## Positive But Economically Dirty Holdout",
        "",
        _format_table(positive_dirty, cols, limit=40),
        "",
        "## Best Rejected Or Failed Rows",
        "",
        _format_table(best, cols, limit=40),
        "",
        "## Component Proxy ICs",
        "",
        _format_table(comp, component_cols, limit=80),
        "",
        "## Outputs",
        "",
        f"- Monthly: `{manifest['outputs']['monthly']}`",
        f"- Weekly: `{manifest['outputs']['weekly']}`",
        f"- Component IC: `{manifest['outputs']['component_proxy']}`",
        f"- Fit/Holdout: `{manifest['outputs']['fit_holdout']}`",
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
    proxy_method: str,
    proxy_tail_frac: float,
    top_fracs: list[float],
    fit_months: list[str],
    holdout_month: str,
    min_week_rows: int,
    selection_mode: str,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = _load_labels(labels_path)
    selected_features = _read_feature_list(feature_list_csv, max_features=max_feature_store_features)
    feature_matrix, feature_store_report = _load_feature_store_columns(
        frame,
        feature_dir=feature_dir,
        selected_features=selected_features,
    )
    if not feature_matrix.empty:
        new_cols = [col for col in feature_matrix.columns if col not in frame.columns]
        frame = pd.concat(
            [
                frame.reset_index(drop=True),
                feature_matrix.loc[:, new_cols].reset_index(drop=True),
            ],
            axis=1,
        )
    metrics = _first_touch_metrics(frame, _path_metrics(frame))
    components = _target_components(metrics)
    features = _feature_columns(frame)

    month_series = frame["__ts__"].dt.to_period("M").astype(str)
    months = sorted(month_series.dropna().unique())
    monthly_rows: list[dict[str, Any]] = []
    weekly_rows: list[dict[str, Any]] = []
    component_rows: list[dict[str, Any]] = []

    for month in months[1:]:
        train_mask = month_series < str(month)
        valid_mask = month_series == str(month)
        if int(train_mask.sum()) < 500 or int(valid_mask.sum()) < 100:
            continue
        train = frame.loc[train_mask].copy()
        valid = frame.loc[valid_mask].copy()
        valid_metrics = metrics.loc[valid_mask].copy()
        valid_components = {name: series.loc[valid_mask] for name, series in components.items()}
        proxies: dict[str, pd.Series] = {}
        diag: dict[str, dict[str, Any]] = {}
        for name in ("utility", "clean_first_touch", "fast_edge", "early_adverse", "slow_timeout", "dirty"):
            score, score_diag = _proxy_score(
                train=train,
                valid=valid,
                features=features,
                target_train=components[name].loc[train_mask],
                top_k=proxy_top_k,
                method=str(proxy_method),
                tail_frac=float(proxy_tail_frac),
            )
            proxies[name] = score
            diag[name] = score_diag
            component_rows.append(
                {
                    "period": str(month),
                    "component": name,
                    "proxy_top_abs_ic": score_diag.get("top_abs_ic"),
                    "proxy_mean_top_abs_ic": score_diag.get("mean_top_abs_ic"),
                    "proxy_ic_actual_u": _spearman(score, valid_metrics["u_policy_net"]),
                    "proxy_ic_actual_clean_exec": _spearman(score, valid_metrics["clean_exec_actual"]),
                    "proxy_ic_actual_dirty": _spearman(score, valid_components["dirty"]),
                    "proxy_ic_actual_fast_edge": _spearman(score, valid_components["fast_edge"]),
                    "proxy_features": ",".join(score_diag.get("features", [])),
                }
            )

        valid_target = _target_for_selection(valid_components, valid.index)
        for spec in SCORE_SPECS:
            score = _score_from_components(spec, proxies)
            m_rows, w_rows = _monthly_weekly_rows(
                valid_frame=valid,
                valid_metrics=valid_metrics,
                valid_target=valid_target,
                score=score,
                score_arm=spec.name,
                month=str(month),
                top_fracs=top_fracs,
                component_diag=diag,
                selection_mode=str(selection_mode),
            )
            monthly_rows.extend(m_rows)
            weekly_rows.extend(w_rows)

    monthly = pd.DataFrame(monthly_rows)
    weekly = pd.DataFrame(weekly_rows)
    components_summary = pd.DataFrame(component_rows)
    fit_holdout = _fit_holdout_summary(
        monthly=monthly,
        weekly=weekly,
        fit_months=fit_months,
        holdout_month=holdout_month,
        min_week_rows=min_week_rows,
    )

    paths = {
        "monthly": output_dir / "first_touch_execution_proxy_monthly.csv",
        "weekly": output_dir / "first_touch_execution_proxy_weekly.csv",
        "component_proxy": output_dir / "first_touch_execution_component_ic.csv",
        "fit_holdout": output_dir / "first_touch_execution_proxy_fit_holdout.csv",
        "manifest": output_dir / "manifest.json",
    }
    monthly.to_csv(paths["monthly"], index=False)
    weekly.to_csv(paths["weekly"], index=False)
    components_summary.to_csv(paths["component_proxy"], index=False)
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
        "max_feature_store_features": max_feature_store_features,
        "feature_store": feature_store_report,
        "features": features,
        "feature_count": int(len(features)),
        "proxy_top_k": int(proxy_top_k),
        "proxy_method": str(proxy_method),
        "proxy_tail_frac": float(proxy_tail_frac),
        "selection_mode": str(selection_mode),
        "score_specs": [spec.__dict__ for spec in SCORE_SPECS],
        "top_fracs": [float(v) for v in top_fracs],
        "fit_months": [str(v) for v in fit_months],
        "holdout_month": str(holdout_month),
        "rows_monthly": int(len(monthly)),
        "rows_weekly": int(len(weekly)),
        "fit_clean_pass_rows": int(fit_holdout["fit_clean_pass"].sum()) if not fit_holdout.empty else 0,
        "holdout_clean_pass_rows": int(fit_holdout["holdout_clean_pass"].sum()) if not fit_holdout.empty else 0,
        "fit_bounded_pass_rows": int(fit_holdout["fit_bounded_pass"].sum()) if not fit_holdout.empty else 0,
        "holdout_bounded_pass_rows": int(fit_holdout["holdout_bounded_pass"].sum()) if not fit_holdout.empty else 0,
        "positive_dirty_holdout_rows": int(fit_holdout["positive_dirty_holdout"].sum()) if not fit_holdout.empty else 0,
        "outputs": {key: str(value) for key, value in paths.items()},
    }
    markdown = _write_markdown(output_dir, fit_holdout, components_summary, manifest)
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
    parser.add_argument("--proxy-top-k", type=int, default=12)
    parser.add_argument("--proxy-method", choices=["ic", "tail_lift"], default="ic")
    parser.add_argument("--proxy-tail-frac", type=float, default=0.05)
    parser.add_argument("--top-fracs", default=",".join(str(v) for v in DEFAULT_TOP_FRACS))
    parser.add_argument("--fit-months", default=",".join(DEFAULT_FIT_MONTHS))
    parser.add_argument("--holdout-month", default=DEFAULT_HOLDOUT_MONTH)
    parser.add_argument("--min-week-rows", type=int, default=3)
    parser.add_argument("--selection-mode", choices=["global", "timestamp"], default="global")
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
        proxy_method=str(args.proxy_method),
        proxy_tail_frac=float(args.proxy_tail_frac),
        top_fracs=_parse_float_csv(args.top_fracs),
        fit_months=_parse_csv(args.fit_months),
        holdout_month=str(args.holdout_month),
        min_week_rows=int(args.min_week_rows),
        selection_mode=str(args.selection_mode),
    )
    print(json.dumps(_json_safe(manifest), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
