#!/usr/bin/env python3
"""Timestamp-local soft/tri-state label proxy ablation before training.

This script follows the soft-label ablation plan without fitting a model. It
fits causal prior-month univariate feature proxies for each candidate label,
optionally subtracts/gates a separately learned first-touch bad-execution
proxy, then evaluates timestamp-local top-10/20/30 rankings.
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

from scripts.run_label_first_touch_execution_proxy_ablation import (  # noqa: E402
    DEFAULT_FEATURE_DIR,
    DEFAULT_FEATURE_LIST_CSV,
    DEFAULT_FIT_MONTHS,
    DEFAULT_HOLDOUT_MONTH,
    DEFAULT_LABELS_DIR,
    _feature_columns,
    _first_touch_metrics,
    _json_safe,
    _load_feature_store_columns,
    _load_labels,
    _parse_csv,
    _path_metrics,
    _proxy_score,
    _read_feature_list,
    _safe_max,
    _safe_mean,
    _safe_min,
    _safe_numeric,
    _safe_quantile,
    _sigmoid,
    _spearman,
    _weighted_metric,
)
from scripts.run_label_quality_proxy_diagnostics import _make_targets  # noqa: E402


DEFAULT_OUTPUT_DIR = Path("data_perp/reports/label_first_touch_soft_recipe_proxy_ablation_v1")
DEFAULT_TOP_KS = (10, 20, 30)
DEFAULT_PLAN_ARMS = (
    "S0_current_y_bin",
    "S2_cost_aware_return",
    "S3_path_quality",
    "S6_asymmetric_downside",
    "S7_horizon_blended",
    "S8_timestamp_rank_path",
)
DEFAULT_FT_ARMS = (
    "FT1_clean_positive_strict",
    "FT2_clean_positive_margin",
    "FT3_tri_clean_minus_bad",
    "FT4_fast_clean_positive",
    "FT5_timestamp_rank_clean",
    "FT6_decisive_tri_state",
)
DEFAULT_SELECTORS = (
    "oracle_label_sort",
    "feature_proxy",
    "feature_proxy_bad_gate50",
    "feature_proxy_minus_bad50_gate50",
)


@dataclass(frozen=True)
class RecipeTarget:
    name: str
    description: str
    target_soft: pd.Series
    target_hard: pd.Series
    bad_soft: pd.Series


def _rank_pct_by_timestamp(values: pd.Series, timestamps: pd.Series) -> pd.Series:
    ranks = _safe_numeric(values).groupby(timestamps, dropna=False).rank(method="average", pct=True)
    fallback = _safe_numeric(values).rank(method="average", pct=True)
    return ranks.fillna(fallback).clip(0.0, 1.0)


def _target_frame(target: RecipeTarget) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "target_soft": _safe_numeric(target.target_soft).clip(0.0, 1.0),
            "target_hard": _safe_numeric(target.target_hard).fillna(0.0).clip(0.0, 1.0),
            "bad_soft": _safe_numeric(target.bad_soft).clip(0.0, 1.0),
        },
        index=target.target_soft.index,
    )


def _global_bad_soft(ft: pd.DataFrame) -> pd.Series:
    stop = _safe_numeric(ft["first_touch_stop"]).fillna(0.0).clip(0.0, 1.0)
    timeout = _safe_numeric(ft["first_touch_timeout"]).fillna(0.0).clip(0.0, 1.0)
    same_bar = _safe_numeric(ft["first_touch_same_bar"]).fillna(0.0).clip(0.0, 1.0)
    bar = _safe_numeric(ft["first_touch_bar"]).fillna(36.0).clip(lower=0.0)
    mae_to_sl = _safe_numeric(ft["first_touch_mae_to_sl"]).fillna(10.0).clip(lower=0.0)
    barrier = _safe_numeric(ft["barrier"]).fillna(0.0).clip(lower=0.0)
    parts = [
        stop,
        same_bar,
        timeout,
        pd.Series(_sigmoid((mae_to_sl - 0.75) / 0.16), index=ft.index),
        pd.Series(_sigmoid((bar - 12.0) / 3.5), index=ft.index),
        pd.Series(_sigmoid((barrier - 0.025) / 0.006), index=ft.index),
    ]
    return pd.concat(parts, axis=1).max(axis=1).clip(0.0, 1.0)


def _build_recipe_targets(
    *,
    frame: pd.DataFrame,
    metrics: pd.DataFrame,
    ft: pd.DataFrame,
) -> dict[str, RecipeTarget]:
    bad_soft = _global_bad_soft(ft)
    plan_raw = _make_targets(frame, metrics)
    descriptions = {
        "S0_current_y_bin": "current hard TP/SL/timeout label",
        "S2_cost_aware_return": "future return after explicit round-trip cost",
        "S3_path_quality": "MFE/MAE/timing path-quality soft label",
        "S6_asymmetric_downside": "path quality with hard downside caps",
        "S7_horizon_blended": "blend hard label, TP2/SL1 path, and fast MFE",
        "S8_timestamp_rank_path": "timestamp-local rank of path quality",
    }
    out: dict[str, RecipeTarget] = {}
    for name in DEFAULT_PLAN_ARMS:
        target = plan_raw[name]
        out[name] = RecipeTarget(
            name=name,
            description=descriptions.get(name, ""),
            target_soft=target["target_soft"],
            target_hard=target["target_hard"],
            bad_soft=bad_soft,
        )

    u = _safe_numeric(ft["u_policy_net"]).fillna(-0.02)
    hit = _safe_numeric(ft["first_touch_hit"]).fillna(0.0).clip(0.0, 1.0)
    stop = _safe_numeric(ft["first_touch_stop"]).fillna(0.0).clip(0.0, 1.0)
    timeout = _safe_numeric(ft["first_touch_timeout"]).fillna(0.0).clip(0.0, 1.0)
    same_bar = _safe_numeric(ft["first_touch_same_bar"]).fillna(0.0).clip(0.0, 1.0)
    bar = _safe_numeric(ft["first_touch_bar"]).fillna(36.0).clip(lower=0.0)
    mae_to_sl = _safe_numeric(ft["first_touch_mae_to_sl"]).fillna(10.0).clip(lower=0.0)
    mfe_to_tp = _safe_numeric(ft["first_touch_mfe_to_tp"]).fillna(0.0).clip(lower=0.0)
    barrier = _safe_numeric(ft["barrier"]).fillna(0.0).clip(lower=0.0)

    utility = pd.Series(_sigmoid((u - 0.0015) / 0.006), index=ft.index).clip(0.0, 1.0)
    margin_utility = pd.Series(_sigmoid((u - 0.0030) / 0.006), index=ft.index).clip(0.0, 1.0)
    strict_clean = (
        hit
        * (1.0 - stop)
        * (1.0 - timeout)
        * (1.0 - same_bar)
        * pd.Series(_sigmoid((0.75 - mae_to_sl) / 0.16), index=ft.index)
        * pd.Series(_sigmoid((12.0 - bar) / 3.5), index=ft.index)
        * pd.Series(_sigmoid((0.025 - barrier) / 0.006), index=ft.index)
    ).clip(0.0, 1.0)
    bounded_clean = (
        hit
        * (1.0 - stop)
        * (1.0 - same_bar)
        * (1.0 - 0.75 * timeout)
        * pd.Series(_sigmoid((1.00 - mae_to_sl) / 0.22), index=ft.index)
        * pd.Series(_sigmoid((16.0 - bar) / 4.0), index=ft.index)
        * pd.Series(_sigmoid((0.030 - barrier) / 0.007), index=ft.index)
    ).clip(0.0, 1.0)
    fast_edge = (
        hit
        * (1.0 - stop)
        * (1.0 - same_bar)
        * pd.Series(_sigmoid((mfe_to_tp - 1.0) / 0.25), index=ft.index)
        * pd.Series(_sigmoid((10.0 - bar) / 3.0), index=ft.index)
    ).clip(0.0, 1.0)

    hard_strict = (
        (u > 0.0)
        & (hit > 0.5)
        & (stop <= 0.5)
        & (timeout <= 0.5)
        & (same_bar <= 0.5)
        & (mae_to_sl <= 0.75)
        & (bar <= 12.0)
        & (barrier <= 0.025)
    ).astype(float)
    hard_margin = (
        (u > 0.0025)
        & (hit > 0.5)
        & (stop <= 0.5)
        & (timeout <= 0.5)
        & (same_bar <= 0.5)
        & (mae_to_sl <= 0.75)
        & (bar <= 12.0)
        & (barrier <= 0.025)
    ).astype(float)
    hard_bounded = (
        (u > 0.0)
        & (hit > 0.5)
        & (stop <= 0.5)
        & (same_bar <= 0.5)
        & (mae_to_sl <= 1.00)
        & (bar <= 16.0)
        & (barrier <= 0.030)
    ).astype(float)

    ft1 = (utility * strict_clean).clip(0.0, 1.0)
    ft2 = (margin_utility * strict_clean).clip(0.0, 1.0)
    ft3 = (utility * bounded_clean * (1.0 - bad_soft)).clip(0.0, 1.0)
    ft4 = (utility * fast_edge * (1.0 - bad_soft)).clip(0.0, 1.0)
    ft5_rank = _rank_pct_by_timestamp(ft2, frame["__ts__"])
    ft5 = (0.50 * ft2 + 0.50 * ft5_rank).clip(0.0, 1.0)
    ft6 = (0.50 + 0.55 * ft1 - 0.55 * bad_soft).clip(0.0, 1.0)

    out.update(
        {
            "FT1_clean_positive_strict": RecipeTarget(
                "FT1_clean_positive_strict",
                "strict first-touch clean positive utility",
                ft1,
                hard_strict,
                bad_soft,
            ),
            "FT2_clean_positive_margin": RecipeTarget(
                "FT2_clean_positive_margin",
                "strict first-touch clean positive utility with explicit net margin",
                ft2,
                hard_margin,
                bad_soft,
            ),
            "FT3_tri_clean_minus_bad": RecipeTarget(
                "FT3_tri_clean_minus_bad",
                "bounded first-touch clean utility with bad-execution suppression",
                ft3,
                hard_bounded,
                bad_soft,
            ),
            "FT4_fast_clean_positive": RecipeTarget(
                "FT4_fast_clean_positive",
                "fast first-touch TP edge with bad-execution suppression",
                ft4,
                hard_strict,
                bad_soft,
            ),
            "FT5_timestamp_rank_clean": RecipeTarget(
                "FT5_timestamp_rank_clean",
                "timestamp-ranked strict clean margin utility",
                ft5,
                hard_margin,
                bad_soft,
            ),
            "FT6_decisive_tri_state": RecipeTarget(
                "FT6_decisive_tri_state",
                "tri-state decisive label: clean positives high, bad execution low, neutral middle",
                ft6,
                hard_strict,
                bad_soft,
            ),
        }
    )
    return out


def _timestamp_top_k_indices(frame: pd.DataFrame, score: pd.Series, top_k: int) -> np.ndarray:
    score_series = _safe_numeric(score).reset_index(drop=True)
    timestamps = pd.to_datetime(frame["__ts__"], errors="coerce").reset_index(drop=True)
    chosen: list[np.ndarray] = []
    for _, ids in pd.Series(np.arange(len(score_series)), index=score_series.index).groupby(timestamps, dropna=False):
        pos = ids.to_numpy(dtype=np.int64)
        valid_pos = pos[np.isfinite(score_series.iloc[pos].to_numpy(dtype=np.float64))]
        if len(valid_pos) == 0:
            continue
        k = min(int(top_k), len(valid_pos))
        values = score_series.iloc[valid_pos].to_numpy(dtype=np.float64)
        order = np.argsort(-values, kind="mergesort")[:k]
        chosen.append(valid_pos[order].astype(np.int64, copy=False))
    if not chosen:
        return np.array([], dtype=np.int64)
    return np.concatenate(chosen).astype(np.int64, copy=False)


def _ndcg_at_k(gain: np.ndarray, score: np.ndarray, top_k: int) -> float:
    valid = np.isfinite(gain) & np.isfinite(score)
    if int(valid.sum()) == 0:
        return float("nan")
    gain = np.clip(gain[valid].astype(float), 0.0, None)
    score = score[valid].astype(float)
    k = min(int(top_k), len(gain))
    if k <= 0:
        return float("nan")
    order = np.argsort(-score, kind="mergesort")[:k]
    ideal = np.argsort(-gain, kind="mergesort")[:k]
    discount = 1.0 / np.log2(np.arange(2, k + 2, dtype=float))
    dcg = float(np.sum(gain[order] * discount))
    idcg = float(np.sum(gain[ideal] * discount))
    return dcg / idcg if idcg > 0.0 else float("nan")


def _timestamp_balanced_metrics(
    *,
    frame: pd.DataFrame,
    metrics: pd.DataFrame,
    target: pd.DataFrame,
    score: pd.Series,
    top_k: int,
) -> dict[str, Any]:
    score_series = _safe_numeric(score).reset_index(drop=True)
    frame_reset = frame.reset_index(drop=True)
    metrics_reset = metrics.reset_index(drop=True)
    target_reset = target.reset_index(drop=True)
    timestamps = pd.to_datetime(frame_reset["__ts__"], errors="coerce")
    rows: list[dict[str, float]] = []
    for _, ids in pd.Series(np.arange(len(frame_reset)), index=frame_reset.index).groupby(timestamps, dropna=False):
        pos = ids.to_numpy(dtype=np.int64)
        valid_pos = pos[np.isfinite(score_series.iloc[pos].to_numpy(dtype=float))]
        if len(valid_pos) == 0:
            continue
        k = min(int(top_k), len(valid_pos))
        local_score = score_series.iloc[valid_pos].to_numpy(dtype=float)
        order = np.argsort(-local_score, kind="mergesort")[:k]
        sel = valid_pos[order]
        local_target = _safe_numeric(target_reset.loc[valid_pos, "target_soft"]).to_numpy(dtype=float)
        local_u = _safe_numeric(metrics_reset.loc[valid_pos, "u_policy_net"]).to_numpy(dtype=float)
        rows.append(
            {
                "hr_label": _safe_mean(target_reset.loc[sel, "target_hard"]),
                "hr_u": _safe_mean(metrics_reset.loc[sel, "u_policy_net"] > 0.0),
                "mean_u": _safe_mean(metrics_reset.loc[sel, "u_policy_net"]),
                "ndcg_label": _ndcg_at_k(local_target, local_score, k),
                "ndcg_u": _ndcg_at_k(np.clip(local_u, 0.0, None), local_score, k),
            }
        )
    if not rows:
        return {
            f"tb_hr_label_at_{top_k}": float("nan"),
            f"tb_hr_u_at_{top_k}": float("nan"),
            f"tb_mean_u_at_{top_k}": float("nan"),
            f"tb_ndcg_label_at_{top_k}": float("nan"),
            f"tb_ndcg_u_at_{top_k}": float("nan"),
            f"timestamp_coverage_at_{top_k}": float("nan"),
        }
    out = pd.DataFrame(rows)
    return {
        f"tb_hr_label_at_{top_k}": _safe_mean(out["hr_label"]),
        f"tb_hr_u_at_{top_k}": _safe_mean(out["hr_u"]),
        f"tb_mean_u_at_{top_k}": _safe_mean(out["mean_u"]),
        f"tb_ndcg_label_at_{top_k}": _safe_mean(out["ndcg_label"]),
        f"tb_ndcg_u_at_{top_k}": _safe_mean(out["ndcg_u"]),
        f"timestamp_coverage_at_{top_k}": float(len(out) / timestamps.nunique(dropna=True))
        if int(timestamps.nunique(dropna=True))
        else float("nan"),
    }


def _selection_row(
    *,
    frame: pd.DataFrame,
    metrics: pd.DataFrame,
    target: pd.DataFrame,
    score: pd.Series,
    arm: str,
    selector: str,
    period: str,
    top_k: int,
    diag: dict[str, Any],
) -> dict[str, Any]:
    frame_reset = frame.reset_index(drop=True)
    metrics_reset = metrics.reset_index(drop=True)
    target_reset = target.reset_index(drop=True)
    score_reset = _safe_numeric(score).reset_index(drop=True)
    idx = _timestamp_top_k_indices(frame_reset, score_reset, top_k)
    selected = metrics_reset.iloc[idx] if len(idx) else metrics_reset.iloc[:0]
    selected_frame = frame_reset.iloc[idx] if len(idx) else frame_reset.iloc[:0]
    selected_target = target_reset.iloc[idx] if len(idx) else target_reset.iloc[:0]
    symbols = selected_frame.get("__symbol__", pd.Series(dtype=object))
    timestamps = selected_frame.get("__ts__", pd.Series(dtype="datetime64[ns]"))
    row = {
        "arm": str(arm),
        "selector": str(selector),
        "period": str(period),
        "top_k": int(top_k),
        "rows": int(len(frame_reset)),
        "selected_rows": int(len(idx)),
        "target_top_soft_mean": _safe_mean(selected_target.get("target_soft")),
        "target_top_hard_rate": _safe_mean(selected_target.get("target_hard")),
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
        "mean_first_touch_mae_to_sl": _safe_mean(selected["first_touch_mae_to_sl"]),
        "p90_first_touch_mae_to_sl": _safe_quantile(selected["first_touch_mae_to_sl"], 0.90),
        "p90_first_touch_bar": _safe_quantile(selected["first_touch_bar"], 0.90),
        "bad_soft_top_mean": _safe_mean(selected_target.get("bad_soft")),
        "symbol_effective_n": _effective_n(symbols),
        "top_symbol_share": float(symbols.value_counts(normalize=True, dropna=False).iloc[0]) if len(symbols) else 0.0,
        "timestamp_effective_n": _effective_n(timestamps.astype(str)),
        "top_timestamp_share": float(timestamps.astype(str).value_counts(normalize=True, dropna=False).iloc[0])
        if len(timestamps)
        else 0.0,
    }
    row.update(_timestamp_balanced_metrics(frame=frame_reset, metrics=metrics_reset, target=target_reset, score=score_reset, top_k=top_k))
    row.update(diag)
    return row


def _effective_n(values: pd.Series) -> float:
    counts = pd.Series(values, dtype=object).value_counts(dropna=False)
    if counts.empty:
        return 0.0
    shares = counts.to_numpy(dtype=float) / float(counts.sum())
    denom = float(np.sum(shares * shares))
    return 1.0 / denom if denom > 0.0 else 0.0


def _monthly_weekly_rows(
    *,
    valid_frame: pd.DataFrame,
    valid_metrics: pd.DataFrame,
    valid_target: pd.DataFrame,
    score: pd.Series,
    arm: str,
    selector: str,
    month: str,
    top_ks: list[int],
    diag: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    monthly_rows: list[dict[str, Any]] = []
    weekly_rows: list[dict[str, Any]] = []
    frame_reset = valid_frame.reset_index(drop=True)
    metrics_reset = valid_metrics.reset_index(drop=True)
    target_reset = valid_target.reset_index(drop=True)
    score_reset = _safe_numeric(score).reset_index(drop=True)
    for top_k in top_ks:
        monthly_rows.append(
            _selection_row(
                frame=frame_reset,
                metrics=metrics_reset,
                target=target_reset,
                score=score_reset,
                arm=arm,
                selector=selector,
                period=str(month),
                top_k=int(top_k),
                diag=diag,
            )
        )
        weeks = frame_reset["__ts__"].dt.to_period("W-SUN").astype(str)
        for week, ids in pd.Series(np.arange(len(frame_reset)), index=frame_reset.index).groupby(weeks, dropna=False):
            pos = ids.to_numpy(dtype=np.int64)
            if len(pos) < 20:
                continue
            week_row = _selection_row(
                frame=frame_reset.iloc[pos].reset_index(drop=True),
                metrics=metrics_reset.iloc[pos].reset_index(drop=True),
                target=target_reset.iloc[pos].reset_index(drop=True),
                score=score_reset.iloc[pos].reset_index(drop=True),
                arm=arm,
                selector=selector,
                period=str(month),
                top_k=int(top_k),
                diag=diag,
            )
            week_row["week"] = str(week)
            week_row["week_selected_rows"] = int(week_row["selected_rows"])
            weekly_rows.append(week_row)
    return monthly_rows, weekly_rows


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
        empty = {
            f"{prefix}_months": 0,
            f"{prefix}_positive_months": 0,
            f"{prefix}_mean_month_u": float("nan"),
            f"{prefix}_worst_month_u": float("nan"),
            f"{prefix}_selected_rows": 0,
        }
        for k in DEFAULT_TOP_KS:
            empty[f"{prefix}_tb_hr{k}_label"] = float("nan")
            empty[f"{prefix}_tb_hr{k}_u"] = float("nan")
            empty[f"{prefix}_tb_ndcg{k}_label"] = float("nan")
            empty[f"{prefix}_tb_ndcg{k}_u"] = float("nan")
        return empty
    mean_u = _safe_numeric(frame["mean_u"])
    out = {
        f"{prefix}_months": int(frame["period"].astype(str).nunique()),
        f"{prefix}_positive_months": int(mean_u.gt(0.0).sum()),
        f"{prefix}_mean_month_u": _safe_mean(mean_u),
        f"{prefix}_worst_month_u": _safe_min(mean_u),
        f"{prefix}_selected_rows": int(_safe_numeric(frame["selected_rows"]).sum()),
        f"{prefix}_clean_exec_actual_rate": _weighted_mean(frame, "clean_exec_actual_rate"),
        f"{prefix}_first_touch_timeout_rate": _weighted_mean(frame, "first_touch_timeout_rate"),
        f"{prefix}_first_touch_bad_mae_to_sl_rate": _weighted_mean(frame, "first_touch_bad_mae_to_sl_rate"),
        f"{prefix}_p90_first_touch_mae_to_sl": _weighted_mean(frame, "p90_first_touch_mae_to_sl"),
        f"{prefix}_q05_u": _weighted_mean(frame, "q05_u"),
        f"{prefix}_bad_soft_top_mean": _weighted_mean(frame, "bad_soft_top_mean"),
        f"{prefix}_max_top_symbol_share": _safe_max(frame["top_symbol_share"]),
    }
    for k in DEFAULT_TOP_KS:
        out[f"{prefix}_tb_hr{k}_label"] = (
            _safe_mean(frame[f"tb_hr_label_at_{k}"]) if f"tb_hr_label_at_{k}" in frame else float("nan")
        )
        out[f"{prefix}_tb_hr{k}_u"] = _safe_mean(frame[f"tb_hr_u_at_{k}"]) if f"tb_hr_u_at_{k}" in frame else float("nan")
        out[f"{prefix}_tb_ndcg{k}_label"] = (
            _safe_mean(frame[f"tb_ndcg_label_at_{k}"]) if f"tb_ndcg_label_at_{k}" in frame else float("nan")
        )
        out[f"{prefix}_tb_ndcg{k}_u"] = (
            _safe_mean(frame[f"tb_ndcg_u_at_{k}"]) if f"tb_ndcg_u_at_{k}" in frame else float("nan")
        )
    return out


def _summarize_week(prefix: str, frame: pd.DataFrame, *, min_week_rows: int) -> dict[str, Any]:
    if frame.empty:
        empty = {
            f"{prefix}_weeks": 0,
            f"{prefix}_material_weeks": 0,
            f"{prefix}_material_positive_week_rate": float("nan"),
            f"{prefix}_q25_week_u": float("nan"),
            f"{prefix}_worst_week_u": float("nan"),
        }
        for k in DEFAULT_TOP_KS:
            empty[f"{prefix}_q10_week_hr{k}_u"] = float("nan")
            empty[f"{prefix}_q25_week_hr{k}_u"] = float("nan")
        return empty
    week_rows = _safe_numeric(frame["week_selected_rows"]).fillna(0.0)
    material = week_rows >= int(min_week_rows)
    mean_u = _safe_numeric(frame["mean_u"])
    positive = mean_u > 0.0
    out = {
        f"{prefix}_weeks": int(len(frame)),
        f"{prefix}_material_weeks": int(material.sum()),
        f"{prefix}_material_positive_week_rate": float((positive & material).sum() / material.sum())
        if int(material.sum())
        else float("nan"),
        f"{prefix}_q25_week_u": _safe_quantile(mean_u[material], 0.25) if int(material.sum()) else float("nan"),
        f"{prefix}_worst_week_u": _safe_min(mean_u[material]) if int(material.sum()) else float("nan"),
    }
    for k in DEFAULT_TOP_KS:
        hr_u = _safe_numeric(frame.get(f"tb_hr_u_at_{k}", pd.Series(np.nan, index=frame.index)))
        out[f"{prefix}_q10_week_hr{k}_u"] = _safe_quantile(hr_u[material], 0.10) if int(material.sum()) else float("nan")
        out[f"{prefix}_q25_week_hr{k}_u"] = _safe_quantile(hr_u[material], 0.25) if int(material.sum()) else float("nan")
    return out


def _fit_holdout_summary(
    *,
    monthly: pd.DataFrame,
    weekly: pd.DataFrame,
    fit_months: list[str],
    holdout_month: str,
    min_week_rows: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if monthly.empty:
        return pd.DataFrame()
    for (arm, selector, top_k), group in monthly.groupby(["arm", "selector", "top_k"], observed=True, dropna=False):
        week_group = weekly[
            weekly["arm"].astype(str).eq(str(arm))
            & weekly["selector"].astype(str).eq(str(selector))
            & _safe_numeric(weekly["top_k"]).eq(int(top_k))
        ].copy()
        fit_month = group[group["period"].astype(str).isin(fit_months)].copy()
        holdout_monthly = group[group["period"].astype(str).eq(str(holdout_month))].copy()
        fit_week = week_group[week_group["period"].astype(str).isin(fit_months)].copy()
        holdout_week = week_group[week_group["period"].astype(str).eq(str(holdout_month))].copy()
        if fit_month.empty or holdout_monthly.empty:
            continue
        row: dict[str, Any] = {"arm": str(arm), "selector": str(selector), "top_k": int(top_k)}
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
        metric_k = int(top_k)
        rank_objective = (
            (row[f"holdout_tb_hr{metric_k}_u"] if pd.notna(row[f"holdout_tb_hr{metric_k}_u"]) else 0.0)
            + 0.50 * (row[f"holdout_tb_ndcg{metric_k}_u"] if pd.notna(row[f"holdout_tb_ndcg{metric_k}_u"]) else 0.0)
            + 0.25
            * (row[f"holdout_q25_week_hr{metric_k}_u"] if pd.notna(row[f"holdout_q25_week_hr{metric_k}_u"]) else 0.0)
            + 0.15
            * (row[f"holdout_q10_week_hr{metric_k}_u"] if pd.notna(row[f"holdout_q10_week_hr{metric_k}_u"]) else 0.0)
            - 0.25
            * (
                row["holdout_first_touch_bad_mae_to_sl_rate"]
                if pd.notna(row["holdout_first_touch_bad_mae_to_sl_rate"])
                else 0.0
            )
            - 0.05
            * (
                row["holdout_p90_first_touch_mae_to_sl"]
                if pd.notna(row["holdout_p90_first_touch_mae_to_sl"])
                else 0.0
            )
        )
        row["fit_sign_pass"] = bool(fit_sign)
        row["holdout_sign_pass"] = bool(holdout_sign)
        row["fit_bounded_pass"] = bool(fit_bounded)
        row["holdout_bounded_standalone_pass"] = bool(holdout_bounded_standalone)
        row["holdout_bounded_pass"] = bool(fit_bounded and holdout_bounded_standalone)
        row["positive_dirty_holdout"] = bool(holdout_sign and not holdout_bounded_standalone)
        row["rank_objective"] = float(rank_objective)
        rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(
        ["holdout_bounded_pass", "positive_dirty_holdout", "rank_objective", "holdout_mean_month_u"],
        ascending=[False, False, False, False],
    )


def _label_summary(
    *,
    frame: pd.DataFrame,
    metrics: pd.DataFrame,
    ft: pd.DataFrame,
    targets: dict[str, RecipeTarget],
    features: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for name, target in targets.items():
        soft = _safe_numeric(target.target_soft)
        hard = _safe_numeric(target.target_hard)
        feature_ics: list[float] = []
        for feature in features:
            ic = _spearman(frame[feature], soft)
            if math.isfinite(ic):
                feature_ics.append(abs(float(ic)))
        feature_ics = sorted(feature_ics, reverse=True)
        rows.append(
            {
                "arm": name,
                "description": target.description,
                "soft_mean": _safe_mean(soft),
                "soft_std": float(soft.std(ddof=0)),
                "soft_p90": _safe_quantile(soft, 0.90),
                "hard_rate": _safe_mean(hard),
                "bad_soft_mean": _safe_mean(target.bad_soft),
                "ic_soft_vs_u": _spearman(soft, metrics["u_policy_net"]),
                "ic_soft_vs_clean_exec": _spearman(soft, ft["clean_exec_actual"]),
                "ic_soft_vs_bad_soft": _spearman(soft, target.bad_soft),
                "ic_soft_vs_first_touch_mae": _spearman(soft, ft["first_touch_mae_to_sl"]),
                "feature_top_abs_ic": feature_ics[0] if feature_ics else float("nan"),
                "feature_mean_top_abs_ic": float(np.mean(feature_ics[:12])) if feature_ics else float("nan"),
                "feature_n_abs_ic_ge_002": int(np.sum(np.asarray(feature_ics) >= 0.02)) if feature_ics else 0,
            }
        )
    return pd.DataFrame(rows)


def _format_table(frame: pd.DataFrame, cols: list[str], limit: int = 40) -> str:
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
    label_summary: pd.DataFrame,
    fit_holdout: pd.DataFrame,
    proxy_ic: pd.DataFrame,
    manifest: dict[str, Any],
) -> Path:
    path = output_dir / "first_touch_soft_recipe_proxy_ablation.md"
    counts = (
        fit_holdout.groupby("selector", observed=True)
        .agg(
            rows=("arm", "size"),
            fit_sign=("fit_sign_pass", "sum"),
            holdout_sign=("holdout_sign_pass", "sum"),
            fit_bounded=("fit_bounded_pass", "sum"),
            holdout_bounded=("holdout_bounded_pass", "sum"),
            positive_dirty=("positive_dirty_holdout", "sum"),
        )
        .reset_index()
        if not fit_holdout.empty
        else pd.DataFrame()
    )
    cols = [
        "arm",
        "selector",
        "top_k",
        "rank_objective",
        "fit_sign_pass",
        "fit_bounded_pass",
        "holdout_sign_pass",
        "holdout_bounded_pass",
        "positive_dirty_holdout",
        "fit_mean_month_u",
        "fit_material_positive_week_rate",
        "holdout_mean_month_u",
        "holdout_material_positive_week_rate",
        "holdout_tb_hr10_u",
        "holdout_tb_ndcg10_u",
        "holdout_tb_hr20_u",
        "holdout_tb_ndcg20_u",
        "holdout_tb_hr30_u",
        "holdout_tb_ndcg30_u",
        "holdout_q25_week_hr10_u",
        "holdout_q25_week_hr20_u",
        "holdout_q25_week_hr30_u",
        "holdout_clean_exec_actual_rate",
        "holdout_first_touch_timeout_rate",
        "holdout_first_touch_bad_mae_to_sl_rate",
        "holdout_p90_first_touch_mae_to_sl",
    ]
    label_cols = [
        "arm",
        "soft_mean",
        "soft_std",
        "hard_rate",
        "ic_soft_vs_u",
        "ic_soft_vs_clean_exec",
        "ic_soft_vs_bad_soft",
        "ic_soft_vs_first_touch_mae",
        "feature_top_abs_ic",
        "feature_mean_top_abs_ic",
        "feature_n_abs_ic_ge_002",
    ]
    proxy_cols = [
        "period",
        "arm",
        "proxy_method",
        "oos_ic_target",
        "oos_ic_u",
        "oos_ic_clean_exec",
        "oos_ic_bad_soft",
        "proxy_top_abs_ic",
        "proxy_mean_top_abs_ic",
        "proxy_features",
    ]
    lines = [
        "# First-Touch Soft Recipe Proxy Ablation",
        "",
        "Scope: proxy tests only. No LightGBM, Optuna, policy geometry, or base/meta training is run.",
        "",
        f"Labels: `{manifest['labels_path']}`",
        f"Proxy method: `{manifest['proxy_method']}`",
        f"Fit months: `{','.join(manifest['fit_months'])}`",
        f"Holdout month: `{manifest['holdout_month']}`",
        f"Top K: `{','.join(str(v) for v in manifest['top_ks'])}`",
        "",
        "The report compares the plan's S0/S2/S3/S6/S7/S8 soft labels against first-touch tri-state labels. Feature proxies are trained on prior months only. All ranking metrics are timestamp-local.",
        "",
        "## Gate Counts",
        "",
        _format_table(counts, ["selector", "rows", "fit_sign", "holdout_sign", "fit_bounded", "holdout_bounded", "positive_dirty"], limit=20),
        "",
        "## Best Rows",
        "",
        _format_table(fit_holdout.sort_values(["holdout_bounded_pass", "rank_objective"], ascending=[False, False]), cols, limit=50),
        "",
        "## Label Shape",
        "",
        _format_table(label_summary.sort_values("feature_mean_top_abs_ic", ascending=False), label_cols, limit=40),
        "",
        "## Monthly Proxy ICs",
        "",
        _format_table(proxy_ic.sort_values(["period", "oos_ic_u"], ascending=[True, False]), proxy_cols, limit=80),
        "",
        "## Outputs",
        "",
        f"- Label summary: `{manifest['outputs']['label_summary']}`",
        f"- Monthly: `{manifest['outputs']['monthly']}`",
        f"- Weekly: `{manifest['outputs']['weekly']}`",
        f"- Proxy IC: `{manifest['outputs']['proxy_ic']}`",
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
    arms: list[str],
    selectors: list[str],
    top_ks: list[int],
    fit_months: list[str],
    holdout_month: str,
    min_week_rows: int,
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
        frame = pd.concat([frame.reset_index(drop=True), feature_matrix.loc[:, new_cols].reset_index(drop=True)], axis=1)
    metrics = _path_metrics(frame)
    ft = _first_touch_metrics(frame, metrics)
    all_targets = _build_recipe_targets(frame=frame, metrics=metrics, ft=ft)
    missing = sorted(set(arms) - set(all_targets))
    if missing:
        raise ValueError(f"Unknown arm(s): {missing}")
    targets = {arm: all_targets[arm] for arm in arms}
    features = _feature_columns(frame)
    label_summary = _label_summary(frame=frame, metrics=metrics, ft=ft, targets=targets, features=features)

    month_series = frame["__ts__"].dt.to_period("M").astype(str)
    months = sorted(month_series.dropna().unique())
    monthly_rows: list[dict[str, Any]] = []
    weekly_rows: list[dict[str, Any]] = []
    proxy_ic_rows: list[dict[str, Any]] = []

    for month in months[1:]:
        train_mask = month_series < str(month)
        valid_mask = month_series == str(month)
        if int(train_mask.sum()) < 500 or int(valid_mask.sum()) < 100:
            continue
        train = frame.loc[train_mask].copy()
        valid = frame.loc[valid_mask].copy()
        valid_metrics = ft.loc[valid_mask].copy()
        bad_target_train = _global_bad_soft(ft.loc[train_mask])
        bad_proxy, bad_diag = _proxy_score(
            train=train,
            valid=valid,
            features=features,
            target_train=bad_target_train,
            top_k=proxy_top_k,
            method=str(proxy_method),
            tail_frac=float(proxy_tail_frac),
        )
        for arm, recipe in targets.items():
            target_frame = _target_frame(recipe)
            train_target = target_frame.loc[train_mask, "target_soft"]
            valid_target = target_frame.loc[valid_mask].reset_index(drop=True)
            proxy, diag = _proxy_score(
                train=train,
                valid=valid,
                features=features,
                target_train=train_target,
                top_k=proxy_top_k,
                method=str(proxy_method),
                tail_frac=float(proxy_tail_frac),
            )
            proxy_reset = _safe_numeric(proxy).reset_index(drop=True)
            bad_proxy_reset = _safe_numeric(bad_proxy).reset_index(drop=True)
            proxy_ic_rows.append(
                {
                    "period": str(month),
                    "arm": arm,
                    "description": recipe.description,
                    "proxy_method": str(proxy_method),
                    "oos_ic_target": _spearman(proxy_reset, valid_target["target_soft"]),
                    "oos_ic_u": _spearman(proxy_reset, valid_metrics.reset_index(drop=True)["u_policy_net"]),
                    "oos_ic_clean_exec": _spearman(
                        proxy_reset,
                        valid_metrics.reset_index(drop=True)["clean_exec_actual"],
                    ),
                    "oos_ic_bad_soft": _spearman(proxy_reset, valid_target["bad_soft"]),
                    "proxy_top_abs_ic": diag.get("top_abs_ic"),
                    "proxy_mean_top_abs_ic": diag.get("mean_top_abs_ic"),
                    "proxy_features": ",".join(diag.get("features", [])),
                    "bad_proxy_top_abs_ic": bad_diag.get("top_abs_ic"),
                    "bad_proxy_mean_top_abs_ic": bad_diag.get("mean_top_abs_ic"),
                    "bad_proxy_features": ",".join(bad_diag.get("features", [])),
                }
            )
            selector_scores: dict[str, tuple[pd.Series, dict[str, Any]]] = {}
            if "oracle_label_sort" in selectors:
                selector_scores["oracle_label_sort"] = (
                    valid_target["target_soft"],
                    {
                        "oos_ic_target": 1.0,
                        "oos_ic_u": _spearman(
                            valid_target["target_soft"],
                            valid_metrics.reset_index(drop=True)["u_policy_net"],
                        ),
                        "oos_ic_clean_exec": _spearman(
                            valid_target["target_soft"],
                            valid_metrics.reset_index(drop=True)["clean_exec_actual"],
                        ),
                        "oos_ic_bad_soft": _spearman(valid_target["target_soft"], valid_target["bad_soft"]),
                        "proxy_top_abs_ic": float("nan"),
                        "proxy_mean_top_abs_ic": float("nan"),
                        "proxy_features": "",
                    },
                )
            if "feature_proxy" in selectors:
                selector_scores["feature_proxy"] = (
                    proxy,
                    {
                        "oos_ic_target": _spearman(proxy_reset, valid_target["target_soft"]),
                        "oos_ic_u": _spearman(proxy_reset, valid_metrics.reset_index(drop=True)["u_policy_net"]),
                        "oos_ic_clean_exec": _spearman(
                            proxy_reset,
                            valid_metrics.reset_index(drop=True)["clean_exec_actual"],
                        ),
                        "oos_ic_bad_soft": _spearman(proxy_reset, valid_target["bad_soft"]),
                        "proxy_top_abs_ic": diag.get("top_abs_ic"),
                        "proxy_mean_top_abs_ic": diag.get("mean_top_abs_ic"),
                        "proxy_features": ",".join(diag.get("features", [])),
                    },
                )
            if "feature_proxy_bad_gate50" in selectors:
                selector_scores["feature_proxy_bad_gate50"] = (
                    proxy.where(_safe_numeric(bad_proxy) <= 0.50),
                    {
                        "oos_ic_target": _spearman(proxy_reset, valid_target["target_soft"]),
                        "oos_ic_u": _spearman(proxy_reset, valid_metrics.reset_index(drop=True)["u_policy_net"]),
                        "oos_ic_clean_exec": _spearman(
                            proxy_reset,
                            valid_metrics.reset_index(drop=True)["clean_exec_actual"],
                        ),
                        "oos_ic_bad_soft": _spearman(proxy_reset, valid_target["bad_soft"]),
                        "proxy_top_abs_ic": diag.get("top_abs_ic"),
                        "proxy_mean_top_abs_ic": diag.get("mean_top_abs_ic"),
                        "proxy_features": ",".join(diag.get("features", [])),
                        "bad_proxy_gate": 0.50,
                    },
                )
            if "feature_proxy_minus_bad50_gate50" in selectors:
                selector_scores["feature_proxy_minus_bad50_gate50"] = (
                    (proxy - 0.50 * _safe_numeric(bad_proxy)).where(_safe_numeric(bad_proxy) <= 0.50),
                    {
                        "oos_ic_target": _spearman(proxy_reset, valid_target["target_soft"]),
                        "oos_ic_u": _spearman(proxy_reset, valid_metrics.reset_index(drop=True)["u_policy_net"]),
                        "oos_ic_clean_exec": _spearman(
                            proxy_reset,
                            valid_metrics.reset_index(drop=True)["clean_exec_actual"],
                        ),
                        "oos_ic_bad_soft": _spearman(proxy_reset, valid_target["bad_soft"]),
                        "proxy_top_abs_ic": diag.get("top_abs_ic"),
                        "proxy_mean_top_abs_ic": diag.get("mean_top_abs_ic"),
                        "proxy_features": ",".join(diag.get("features", [])),
                        "bad_proxy_gate": 0.50,
                        "bad_proxy_penalty": 0.50,
                    },
                )
            for selector, (score, selector_diag) in selector_scores.items():
                m_rows, w_rows = _monthly_weekly_rows(
                    valid_frame=valid,
                    valid_metrics=valid_metrics,
                    valid_target=valid_target,
                    score=score,
                    arm=arm,
                    selector=selector,
                    month=str(month),
                    top_ks=top_ks,
                    diag=selector_diag,
                )
                monthly_rows.extend(m_rows)
                weekly_rows.extend(w_rows)

    monthly = pd.DataFrame(monthly_rows)
    weekly = pd.DataFrame(weekly_rows)
    proxy_ic = pd.DataFrame(proxy_ic_rows)
    fit_holdout = _fit_holdout_summary(
        monthly=monthly,
        weekly=weekly,
        fit_months=fit_months,
        holdout_month=holdout_month,
        min_week_rows=min_week_rows,
    )

    paths = {
        "label_summary": output_dir / "first_touch_soft_recipe_label_summary.csv",
        "monthly": output_dir / "first_touch_soft_recipe_monthly.csv",
        "weekly": output_dir / "first_touch_soft_recipe_weekly.csv",
        "proxy_ic": output_dir / "first_touch_soft_recipe_proxy_ic.csv",
        "fit_holdout": output_dir / "first_touch_soft_recipe_fit_holdout.csv",
        "manifest": output_dir / "manifest.json",
    }
    label_summary.to_csv(paths["label_summary"], index=False)
    monthly.to_csv(paths["monthly"], index=False)
    weekly.to_csv(paths["weekly"], index=False)
    proxy_ic.to_csv(paths["proxy_ic"], index=False)
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
        "feature_count": int(len(features)),
        "proxy_top_k": int(proxy_top_k),
        "proxy_method": str(proxy_method),
        "proxy_tail_frac": float(proxy_tail_frac),
        "arms": list(arms),
        "selectors": list(selectors),
        "top_ks": [int(v) for v in top_ks],
        "fit_months": [str(v) for v in fit_months],
        "holdout_month": str(holdout_month),
        "rows_monthly": int(len(monthly)),
        "rows_weekly": int(len(weekly)),
        "fit_sign_pass_rows": int(fit_holdout["fit_sign_pass"].sum()) if not fit_holdout.empty else 0,
        "holdout_sign_pass_rows": int(fit_holdout["holdout_sign_pass"].sum()) if not fit_holdout.empty else 0,
        "fit_bounded_pass_rows": int(fit_holdout["fit_bounded_pass"].sum()) if not fit_holdout.empty else 0,
        "holdout_bounded_pass_rows": int(fit_holdout["holdout_bounded_pass"].sum()) if not fit_holdout.empty else 0,
        "positive_dirty_holdout_rows": int(fit_holdout["positive_dirty_holdout"].sum()) if not fit_holdout.empty else 0,
        "outputs": {key: str(value) for key, value in paths.items()},
    }
    markdown = _write_markdown(
        output_dir=output_dir,
        label_summary=label_summary,
        fit_holdout=fit_holdout,
        proxy_ic=proxy_ic,
        manifest=manifest,
    )
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
    parser.add_argument("--arms", default=",".join(DEFAULT_PLAN_ARMS + DEFAULT_FT_ARMS))
    parser.add_argument("--selectors", default=",".join(DEFAULT_SELECTORS))
    parser.add_argument("--top-ks", default=",".join(str(v) for v in DEFAULT_TOP_KS))
    parser.add_argument("--fit-months", default=",".join(DEFAULT_FIT_MONTHS))
    parser.add_argument("--holdout-month", default=DEFAULT_HOLDOUT_MONTH)
    parser.add_argument("--min-week-rows", type=int, default=3)
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
        arms=_parse_csv(args.arms),
        selectors=_parse_csv(args.selectors),
        top_ks=[int(v) for v in _parse_csv(args.top_ks)],
        fit_months=_parse_csv(args.fit_months),
        holdout_month=str(args.holdout_month),
        min_week_rows=int(args.min_week_rows),
    )
    summary_keys = [
        "output_dir",
        "rows",
        "feature_count",
        "proxy_method",
        "arms",
        "selectors",
        "rows_monthly",
        "rows_weekly",
        "fit_sign_pass_rows",
        "holdout_sign_pass_rows",
        "fit_bounded_pass_rows",
        "holdout_bounded_pass_rows",
        "positive_dirty_holdout_rows",
        "outputs",
    ]
    print(json.dumps(_json_safe({key: manifest.get(key) for key in summary_keys}), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
