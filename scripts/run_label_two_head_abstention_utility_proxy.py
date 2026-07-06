#!/usr/bin/env python3
"""Two-head abstention-then-utility proxy test before model training.

This diagnostic fits two separate prior-month feature proxies:

1. a bad first-touch execution proxy;
2. a utility proxy.

It then gates rows by the bad-execution proxy and ranks accepted rows by the
utility proxy inside each timestamp. Thresholds are selected on Apr-May fit
evidence only, then evaluated on the June holdout.
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
    _parse_float_csv,
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
)
from scripts.run_label_first_touch_soft_recipe_proxy_ablation import (  # noqa: E402
    DEFAULT_TOP_KS,
    _effective_n,
    _global_bad_soft,
    _ndcg_at_k,
    _timestamp_top_k_indices,
)


DEFAULT_OUTPUT_DIR = Path("data_perp/reports/label_two_head_abstention_utility_proxy_v1")
DEFAULT_BAD_THRESHOLDS = (0.30, 0.40, 0.50, 0.60, 0.70)
DEFAULT_UTILITY_THRESHOLDS: tuple[float, ...] = ()
DEFAULT_UTILITY_TARGETS = ("utility", "margin_utility", "oracle_s2")
DEFAULT_SCORE_RULES = ("utility", "utility_minus_bad025", "utility_minus_bad050")


@dataclass(frozen=True)
class TwoHeadSpec:
    name: str
    utility_target: str
    score_rule: str
    bad_threshold: float
    utility_threshold: float | None = None


def _utility_targets(frame: pd.DataFrame, ft: pd.DataFrame) -> dict[str, pd.Series]:
    u = _safe_numeric(ft["u_policy_net"]).fillna(-0.02)
    ret_net = _safe_numeric(ft["ret_net"]).fillna(-0.02)
    hit = _safe_numeric(ft["first_touch_hit"]).fillna(0.0).clip(0.0, 1.0)
    stop = _safe_numeric(ft["first_touch_stop"]).fillna(0.0).clip(0.0, 1.0)
    timeout = _safe_numeric(ft["first_touch_timeout"]).fillna(0.0).clip(0.0, 1.0)
    same_bar = _safe_numeric(ft["first_touch_same_bar"]).fillna(0.0).clip(0.0, 1.0)
    mae_to_sl = _safe_numeric(ft["first_touch_mae_to_sl"]).fillna(10.0).clip(lower=0.0)
    mfe_to_tp = _safe_numeric(ft["first_touch_mfe_to_tp"]).fillna(0.0).clip(lower=0.0)
    bar = _safe_numeric(ft["first_touch_bar"]).fillna(36.0).clip(lower=0.0)
    barrier = _safe_numeric(ft["barrier"]).fillna(0.0).clip(lower=0.0)
    clean = (
        hit
        * (1.0 - stop)
        * (1.0 - same_bar)
        * (1.0 - 0.5 * timeout)
        * pd.Series(_sigmoid((1.00 - mae_to_sl) / 0.22), index=ft.index)
        * pd.Series(_sigmoid((16.0 - bar) / 4.0), index=ft.index)
        * pd.Series(_sigmoid((0.030 - barrier) / 0.007), index=ft.index)
    ).clip(0.0, 1.0)
    strict_clean = (
        hit
        * (1.0 - stop)
        * (1.0 - same_bar)
        * (1.0 - timeout)
        * pd.Series(_sigmoid((0.80 - mae_to_sl) / 0.18), index=ft.index)
        * pd.Series(_sigmoid((12.0 - bar) / 3.0), index=ft.index)
        * pd.Series(_sigmoid((0.025 - barrier) / 0.005), index=ft.index)
    ).clip(0.0, 1.0)
    low_mae_clean = (
        hit
        * (1.0 - stop)
        * (1.0 - same_bar)
        * (1.0 - timeout)
        * pd.Series(_sigmoid((0.60 - mae_to_sl) / 0.14), index=ft.index)
        * pd.Series(_sigmoid((10.0 - bar) / 3.0), index=ft.index)
        * pd.Series(_sigmoid((0.022 - barrier) / 0.004), index=ft.index)
    ).clip(0.0, 1.0)
    fast_decisive_clean = (
        hit
        * (1.0 - stop)
        * (1.0 - same_bar)
        * (1.0 - timeout)
        * pd.Series(_sigmoid((8.0 - bar) / 2.5), index=ft.index)
        * pd.Series(_sigmoid((0.75 - mae_to_sl) / 0.18), index=ft.index)
        * pd.Series(_sigmoid((0.025 - barrier) / 0.005), index=ft.index)
        * pd.Series(_sigmoid((mfe_to_tp - 1.0) / 0.25), index=ft.index)
    ).clip(0.0, 1.0)
    utility = pd.Series(_sigmoid((u - 0.0015) / 0.006), index=ft.index).clip(0.0, 1.0)
    margin_utility = pd.Series(_sigmoid((u - 0.0030) / 0.006), index=ft.index).clip(0.0, 1.0)
    strict_margin_utility = pd.Series(_sigmoid((u - 0.0040) / 0.005), index=ft.index).clip(0.0, 1.0)
    net_margin_utility = pd.Series(_sigmoid((ret_net - 0.0030) / 0.006), index=ft.index).clip(0.0, 1.0)
    s2 = pd.Series(_sigmoid(ret_net / 0.006), index=ft.index).clip(0.0, 1.0)
    return {
        "utility": utility,
        "margin_utility": margin_utility,
        "oracle_s2": s2,
        "clean_utility": (utility * clean).clip(0.0, 1.0),
        "margin_clean_utility": (margin_utility * strict_clean).clip(0.0, 1.0),
        "strict_margin_clean_utility": (strict_margin_utility * strict_clean).clip(0.0, 1.0),
        "low_mae_margin_utility": (strict_margin_utility * low_mae_clean).clip(0.0, 1.0),
        "fast_margin_clean_utility": (strict_margin_utility * fast_decisive_clean).clip(0.0, 1.0),
        "net_margin_clean_utility": (net_margin_utility * strict_clean).clip(0.0, 1.0),
    }


def _build_specs(
    *,
    utility_targets: list[str],
    score_rules: list[str],
    bad_thresholds: list[float],
    utility_thresholds: list[float],
) -> list[TwoHeadSpec]:
    specs: list[TwoHeadSpec] = []
    active_utility_thresholds: list[float | None] = list(utility_thresholds) if utility_thresholds else [None]
    for utility_target, score_rule, bad_threshold, utility_threshold in product(
        utility_targets,
        score_rules,
        bad_thresholds,
        active_utility_thresholds,
    ):
        utility_part = (
            f"_umin{int(round(float(utility_threshold) * 100)):02d}"
            if utility_threshold is not None
            else ""
        )
        name = (
            f"{utility_target}_{score_rule}"
            f"_badmax{int(round(float(bad_threshold) * 100)):02d}"
            f"{utility_part}"
        )
        specs.append(
            TwoHeadSpec(
                name=name,
                utility_target=str(utility_target),
                score_rule=str(score_rule),
                bad_threshold=float(bad_threshold),
                utility_threshold=float(utility_threshold) if utility_threshold is not None else None,
            )
        )
    return specs


def _score_from_spec(spec: TwoHeadSpec, utility_proxy: pd.Series, bad_proxy: pd.Series) -> pd.Series:
    utility = _safe_numeric(utility_proxy)
    bad = _safe_numeric(bad_proxy)
    if spec.score_rule == "utility":
        score = utility
    elif spec.score_rule == "utility_minus_bad025":
        score = utility - 0.25 * bad
    elif spec.score_rule == "utility_minus_bad050":
        score = utility - 0.50 * bad
    else:
        raise ValueError(f"unknown score rule: {spec.score_rule}")
    gate = bad <= float(spec.bad_threshold)
    if spec.utility_threshold is not None:
        gate = gate & (utility >= float(spec.utility_threshold))
    return score.where(gate)


def _target_for_selection(ft: pd.DataFrame, utility_soft: pd.Series, bad_soft: pd.Series) -> pd.DataFrame:
    index = ft.index
    soft = pd.Series(_safe_numeric(utility_soft).to_numpy(dtype=float, copy=False), index=index)
    bad = pd.Series(_safe_numeric(bad_soft).to_numpy(dtype=float, copy=False), index=index)
    u = _safe_numeric(ft["u_policy_net"]).fillna(-0.02)
    hit = _safe_numeric(ft["first_touch_hit"]).fillna(0.0)
    stop = _safe_numeric(ft["first_touch_stop"]).fillna(0.0)
    same_bar = _safe_numeric(ft["first_touch_same_bar"]).fillna(0.0)
    timeout = _safe_numeric(ft["first_touch_timeout"]).fillna(0.0)
    mae = _safe_numeric(ft["first_touch_mae_to_sl"]).fillna(10.0)
    bar = _safe_numeric(ft["first_touch_bar"]).fillna(36.0)
    hard = (
        (u > 0.0)
        & (hit > 0.5)
        & (stop <= 0.5)
        & (same_bar <= 0.5)
        & (timeout <= 0.5)
        & (mae <= 1.0)
        & (bar <= 16.0)
    ).astype(float)
    return pd.DataFrame(
        {
            "target_soft": soft.clip(0.0, 1.0),
            "target_hard": hard,
            "bad_soft": bad.clip(0.0, 1.0),
        },
        index=ft.index,
    )


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


def _candidate_stats(frame: pd.DataFrame, score: pd.Series) -> dict[str, Any]:
    score_series = _safe_numeric(score).reset_index(drop=True)
    timestamps = pd.to_datetime(frame["__ts__"], errors="coerce").reset_index(drop=True)
    finite = np.isfinite(score_series.to_numpy(dtype=float))
    rows = int(len(score_series))
    active_counts = timestamps[finite].value_counts(dropna=True)
    total_ts = int(timestamps.nunique(dropna=True))
    return {
        "candidate_rows": int(finite.sum()),
        "candidate_rate": float(finite.sum() / rows) if rows else float("nan"),
        "candidate_timestamp_coverage": float(len(active_counts) / total_ts) if total_ts else float("nan"),
        "mean_candidates_per_active_ts": _safe_mean(active_counts),
    }


def _selection_row(
    *,
    frame: pd.DataFrame,
    metrics: pd.DataFrame,
    target: pd.DataFrame,
    score: pd.Series,
    spec: TwoHeadSpec,
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
    symbols = selected_frame.get("__symbol__", pd.Series(dtype=object))
    row = {
        "arm": spec.name,
        "utility_target": spec.utility_target,
        "score_rule": spec.score_rule,
        "bad_threshold": float(spec.bad_threshold),
        "utility_threshold": float(spec.utility_threshold) if spec.utility_threshold is not None else float("nan"),
        "selector": "two_head_abstain_then_utility",
        "period": str(period),
        "top_k": int(top_k),
        "rows": int(len(frame_reset)),
        "selected_rows": int(len(idx)),
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
        "symbol_effective_n": _effective_n(symbols),
        "top_symbol_share": float(symbols.value_counts(normalize=True, dropna=False).iloc[0]) if len(symbols) else 0.0,
    }
    row.update(_candidate_stats(frame_reset, score_reset))
    row.update(
        _timestamp_balanced_metrics(
            frame=frame_reset,
            metrics=metrics_reset,
            target=target_reset,
            score=score_reset,
            top_k=top_k,
        )
    )
    row.update(diag)
    return row


def _monthly_weekly_rows(
    *,
    valid_frame: pd.DataFrame,
    valid_metrics: pd.DataFrame,
    valid_target: pd.DataFrame,
    score: pd.Series,
    spec: TwoHeadSpec,
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
                spec=spec,
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
                spec=spec,
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
        return {
            f"{prefix}_months": 0,
            f"{prefix}_positive_months": 0,
            f"{prefix}_mean_month_u": float("nan"),
            f"{prefix}_worst_month_u": float("nan"),
            f"{prefix}_positive_return_months": 0,
            f"{prefix}_mean_month_return_net": float("nan"),
            f"{prefix}_worst_month_return_net": float("nan"),
            f"{prefix}_sum_return_net": float("nan"),
            f"{prefix}_weighted_mean_return_net": float("nan"),
            f"{prefix}_selected_rows": 0,
        }
    mean_u = _safe_numeric(frame["mean_u"])
    mean_return_net = _safe_numeric(frame.get("mean_return_net", pd.Series(np.nan, index=frame.index)))
    selected_rows = _safe_numeric(frame["selected_rows"]).fillna(0.0)
    sum_return_net = float((mean_return_net.fillna(0.0) * selected_rows).sum())
    total_rows = float(selected_rows.sum())
    out = {
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
        f"{prefix}_p90_first_touch_mae_to_sl": _weighted_mean(frame, "p90_first_touch_mae_to_sl"),
        f"{prefix}_candidate_rate": _safe_mean(frame["candidate_rate"]),
        f"{prefix}_candidate_timestamp_coverage": _safe_mean(frame["candidate_timestamp_coverage"]),
        f"{prefix}_max_top_symbol_share": _safe_max(frame["top_symbol_share"]),
    }
    for k in DEFAULT_TOP_KS:
        out[f"{prefix}_tb_hr{k}_u"] = _safe_mean(frame[f"tb_hr_u_at_{k}"]) if f"tb_hr_u_at_{k}" in frame else float("nan")
        out[f"{prefix}_tb_ndcg{k}_u"] = (
            _safe_mean(frame[f"tb_ndcg_u_at_{k}"]) if f"tb_ndcg_u_at_{k}" in frame else float("nan")
        )
    return out


def _summarize_week(prefix: str, frame: pd.DataFrame, *, min_week_rows: int) -> dict[str, Any]:
    if frame.empty:
        out = {
            f"{prefix}_weeks": 0,
            f"{prefix}_material_weeks": 0,
            f"{prefix}_material_positive_week_rate": float("nan"),
            f"{prefix}_q25_week_u": float("nan"),
            f"{prefix}_worst_week_u": float("nan"),
            f"{prefix}_material_positive_return_week_rate": float("nan"),
            f"{prefix}_q25_week_return_net": float("nan"),
            f"{prefix}_worst_week_return_net": float("nan"),
        }
        for k in DEFAULT_TOP_KS:
            out[f"{prefix}_q10_week_hr{k}_u"] = float("nan")
            out[f"{prefix}_q25_week_hr{k}_u"] = float("nan")
        return out
    week_rows = _safe_numeric(frame["week_selected_rows"]).fillna(0.0)
    material = week_rows >= int(min_week_rows)
    mean_u = _safe_numeric(frame["mean_u"])
    mean_return_net = _safe_numeric(frame.get("mean_return_net", pd.Series(np.nan, index=frame.index)))
    positive = mean_u > 0.0
    positive_return = mean_return_net > 0.0
    out = {
        f"{prefix}_weeks": int(len(frame)),
        f"{prefix}_material_weeks": int(material.sum()),
        f"{prefix}_material_positive_week_rate": float((positive & material).sum() / material.sum())
        if int(material.sum())
        else float("nan"),
        f"{prefix}_q25_week_u": _safe_quantile(mean_u[material], 0.25) if int(material.sum()) else float("nan"),
        f"{prefix}_worst_week_u": _safe_min(mean_u[material]) if int(material.sum()) else float("nan"),
        f"{prefix}_material_positive_return_week_rate": float((positive_return & material).sum() / material.sum())
        if int(material.sum())
        else float("nan"),
        f"{prefix}_q25_week_return_net": _safe_quantile(mean_return_net[material], 0.25)
        if int(material.sum())
        else float("nan"),
        f"{prefix}_worst_week_return_net": _safe_min(mean_return_net[material])
        if int(material.sum())
        else float("nan"),
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
    group_cols = ["arm", "utility_target", "score_rule", "bad_threshold", "utility_threshold", "top_k"]
    for key, group in monthly.groupby(group_cols, observed=True, dropna=False):
        arm, utility_target, score_rule, bad_threshold, utility_threshold, top_k = key
        week_group = weekly[
            weekly["arm"].astype(str).eq(str(arm))
            & _safe_numeric(weekly["top_k"]).eq(int(top_k))
        ].copy()
        fit_month = group[group["period"].astype(str).isin(fit_months)].copy()
        holdout_monthly = group[group["period"].astype(str).eq(str(holdout_month))].copy()
        fit_week = week_group[week_group["period"].astype(str).isin(fit_months)].copy()
        holdout_week = week_group[week_group["period"].astype(str).eq(str(holdout_month))].copy()
        if fit_month.empty or holdout_monthly.empty:
            continue
        row: dict[str, Any] = {
            "arm": str(arm),
            "utility_target": str(utility_target),
            "score_rule": str(score_rule),
            "bad_threshold": float(bad_threshold),
            "utility_threshold": float(utility_threshold) if pd.notna(utility_threshold) else float("nan"),
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
        fit_return_sign = (
            row["fit_months"] == len(fit_months)
            and row["fit_positive_return_months"] == len(fit_months)
            and row["fit_worst_month_return_net"] > 0.0
            and row["fit_material_weeks"] >= 4
            and row["fit_material_positive_return_week_rate"] >= 0.55
        )
        holdout_sign = (
            row["holdout_mean_month_u"] > 0.0
            and row["holdout_material_weeks"] >= 2
            and row["holdout_material_positive_week_rate"] >= 0.50
        )
        holdout_return_sign = (
            row["holdout_mean_month_return_net"] > 0.0
            and row["holdout_material_weeks"] >= 2
            and row["holdout_material_positive_return_week_rate"] >= 0.50
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
        k = int(top_k)
        fit_objective = (
            (row["fit_mean_month_u"] if pd.notna(row["fit_mean_month_u"]) else -1.0)
            + 0.50 * (row["fit_q25_week_u"] if pd.notna(row["fit_q25_week_u"]) else -1.0)
            + 0.10 * (row[f"fit_tb_hr{k}_u"] if pd.notna(row.get(f"fit_tb_hr{k}_u")) else 0.0)
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
            + 0.10 * (row[f"holdout_tb_hr{k}_u"] if pd.notna(row.get(f"holdout_tb_hr{k}_u")) else 0.0)
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
        row["fit_return_sign_pass"] = bool(fit_return_sign)
        row["holdout_sign_pass"] = bool(holdout_sign)
        row["holdout_return_sign_pass"] = bool(holdout_return_sign)
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
    group_cols = ["utility_target", "score_rule", "top_k"]
    for _, group in fit_holdout.groupby(group_cols, observed=True, dropna=False):
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
    fit_holdout: pd.DataFrame,
    selected_by_fit: pd.DataFrame,
    proxy_ic: pd.DataFrame,
    manifest: dict[str, Any],
) -> Path:
    path = output_dir / "two_head_abstention_utility_proxy.md"
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
        "utility_target",
        "score_rule",
        "bad_threshold",
        "utility_threshold",
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
        "fit_candidate_rate",
        "fit_p90_first_touch_mae_to_sl",
        "holdout_mean_month_u",
        "holdout_material_positive_week_rate",
        "holdout_tb_hr10_u",
        "holdout_tb_hr20_u",
        "holdout_tb_hr30_u",
        "holdout_clean_exec_actual_rate",
        "holdout_first_touch_timeout_rate",
        "holdout_first_touch_bad_mae_to_sl_rate",
        "holdout_p90_first_touch_mae_to_sl",
        "holdout_candidate_rate",
        "holdout_candidate_timestamp_coverage",
    ]
    proxy_cols = [
        "period",
        "utility_target",
        "utility_proxy_ic_target",
        "utility_proxy_ic_u",
        "utility_proxy_ic_clean_exec",
        "utility_proxy_ic_bad",
        "bad_proxy_ic_bad",
        "bad_proxy_ic_u",
        "bad_proxy_ic_clean_exec",
        "utility_proxy_features",
        "bad_proxy_features",
    ]
    lines = [
        "# Two-Head Abstention Then Utility Proxy",
        "",
        "Scope: proxy tests only. No LightGBM, Optuna, policy geometry, or base/meta training is run.",
        "",
        f"Labels: `{manifest['labels_path']}`",
        f"Proxy method: `{manifest['proxy_method']}`",
        f"Fit months: `{','.join(manifest['fit_months'])}`",
        f"Holdout month: `{manifest['holdout_month']}`",
        f"Top K: `{','.join(str(v) for v in manifest['top_ks'])}`",
        "",
        "The threshold grid is reported in full. `selected_by_fit` chooses the bad-execution threshold using Apr-May only, then shows the June result.",
        "",
        "## Counts",
        "",
        _format_table(counts, ["rows", "fit_sign", "holdout_sign", "fit_bounded", "holdout_bounded", "positive_dirty"], limit=10),
        "",
        "## Selected By Apr-May Fit Only",
        "",
        _format_table(selected_by_fit, cols, limit=40),
        "",
        "## Best Grid Rows",
        "",
        _format_table(fit_holdout, cols, limit=60),
        "",
        "## Monthly Proxy IC",
        "",
        _format_table(proxy_ic.sort_values(["period", "utility_proxy_ic_u"], ascending=[True, False]), proxy_cols, limit=80),
        "",
        "## Outputs",
        "",
        f"- Monthly: `{manifest['outputs']['monthly']}`",
        f"- Weekly: `{manifest['outputs']['weekly']}`",
        f"- Proxy IC: `{manifest['outputs']['proxy_ic']}`",
        f"- Fit/Holdout: `{manifest['outputs']['fit_holdout']}`",
        f"- Selected by fit: `{manifest['outputs']['selected_by_fit']}`",
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
    utility_targets: list[str],
    score_rules: list[str],
    bad_thresholds: list[float],
    utility_thresholds: list[float],
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
    bad_soft = _global_bad_soft(ft)
    utility_map = _utility_targets(frame, ft)
    missing = sorted(set(utility_targets) - set(utility_map))
    if missing:
        raise ValueError(f"Unknown utility target(s): {missing}")
    features = _feature_columns(frame)
    specs = _build_specs(
        utility_targets=utility_targets,
        score_rules=score_rules,
        bad_thresholds=bad_thresholds,
        utility_thresholds=utility_thresholds,
    )

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
        bad_proxy, bad_diag = _proxy_score(
            train=train,
            valid=valid,
            features=features,
            target_train=bad_soft.loc[train_mask],
            top_k=proxy_top_k,
            method=str(proxy_method),
            tail_frac=float(proxy_tail_frac),
        )
        bad_proxy_reset = _safe_numeric(bad_proxy).reset_index(drop=True)
        valid_bad = bad_soft.loc[valid_mask].reset_index(drop=True)
        for utility_target in utility_targets:
            utility_soft = utility_map[utility_target]
            utility_proxy, utility_diag = _proxy_score(
                train=train,
                valid=valid,
                features=features,
                target_train=utility_soft.loc[train_mask],
                top_k=proxy_top_k,
                method=str(proxy_method),
                tail_frac=float(proxy_tail_frac),
            )
            utility_proxy_reset = _safe_numeric(utility_proxy).reset_index(drop=True)
            valid_target = _target_for_selection(
                valid_metrics,
                utility_soft.loc[valid_mask].reset_index(drop=True),
                valid_bad,
            )
            valid_target_reset = valid_target.reset_index(drop=True)
            proxy_ic_rows.append(
                {
                    "period": str(month),
                    "utility_target": str(utility_target),
                    "proxy_method": str(proxy_method),
                    "utility_proxy_ic_target": _spearman(utility_proxy_reset, valid_target_reset["target_soft"]),
                    "utility_proxy_ic_u": _spearman(
                        utility_proxy_reset,
                        valid_metrics.reset_index(drop=True)["u_policy_net"],
                    ),
                    "utility_proxy_ic_clean_exec": _spearman(
                        utility_proxy_reset,
                        valid_metrics.reset_index(drop=True)["clean_exec_actual"],
                    ),
                    "utility_proxy_ic_bad": _spearman(utility_proxy_reset, valid_bad),
                    "bad_proxy_ic_bad": _spearman(bad_proxy_reset, valid_bad),
                    "bad_proxy_ic_u": _spearman(
                        bad_proxy_reset,
                        valid_metrics.reset_index(drop=True)["u_policy_net"],
                    ),
                    "bad_proxy_ic_clean_exec": _spearman(
                        bad_proxy_reset,
                        valid_metrics.reset_index(drop=True)["clean_exec_actual"],
                    ),
                    "utility_proxy_top_abs_ic": utility_diag.get("top_abs_ic"),
                    "utility_proxy_mean_top_abs_ic": utility_diag.get("mean_top_abs_ic"),
                    "utility_proxy_features": ",".join(utility_diag.get("features", [])),
                    "bad_proxy_top_abs_ic": bad_diag.get("top_abs_ic"),
                    "bad_proxy_mean_top_abs_ic": bad_diag.get("mean_top_abs_ic"),
                    "bad_proxy_features": ",".join(bad_diag.get("features", [])),
                }
            )
            target_specs = [spec for spec in specs if spec.utility_target == utility_target]
            for spec in target_specs:
                score = _score_from_spec(spec, utility_proxy, bad_proxy)
                diag = {
                    "utility_proxy_ic_target": _spearman(utility_proxy_reset, valid_target_reset["target_soft"]),
                    "utility_proxy_ic_u": _spearman(
                        utility_proxy_reset,
                        valid_metrics.reset_index(drop=True)["u_policy_net"],
                    ),
                    "utility_proxy_ic_clean_exec": _spearman(
                        utility_proxy_reset,
                        valid_metrics.reset_index(drop=True)["clean_exec_actual"],
                    ),
                    "utility_proxy_ic_bad": _spearman(utility_proxy_reset, valid_bad),
                    "bad_proxy_ic_bad": _spearman(bad_proxy_reset, valid_bad),
                    "bad_proxy_ic_u": _spearman(
                        bad_proxy_reset,
                        valid_metrics.reset_index(drop=True)["u_policy_net"],
                    ),
                    "bad_proxy_ic_clean_exec": _spearman(
                        bad_proxy_reset,
                        valid_metrics.reset_index(drop=True)["clean_exec_actual"],
                    ),
                    "utility_proxy_features": ",".join(utility_diag.get("features", [])),
                    "bad_proxy_features": ",".join(bad_diag.get("features", [])),
                }
                m_rows, w_rows = _monthly_weekly_rows(
                    valid_frame=valid,
                    valid_metrics=valid_metrics,
                    valid_target=valid_target,
                    score=score,
                    spec=spec,
                    month=str(month),
                    top_ks=top_ks,
                    diag=diag,
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
    selected_by_fit = _select_by_fit(fit_holdout)

    paths = {
        "monthly": output_dir / "two_head_proxy_monthly.csv",
        "weekly": output_dir / "two_head_proxy_weekly.csv",
        "proxy_ic": output_dir / "two_head_proxy_ic.csv",
        "fit_holdout": output_dir / "two_head_proxy_fit_holdout.csv",
        "selected_by_fit": output_dir / "two_head_proxy_selected_by_fit.csv",
        "manifest": output_dir / "manifest.json",
    }
    monthly.to_csv(paths["monthly"], index=False)
    weekly.to_csv(paths["weekly"], index=False)
    proxy_ic.to_csv(paths["proxy_ic"], index=False)
    fit_holdout.to_csv(paths["fit_holdout"], index=False)
    selected_by_fit.to_csv(paths["selected_by_fit"], index=False)

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
        "utility_targets": list(utility_targets),
        "score_rules": list(score_rules),
        "bad_thresholds": [float(v) for v in bad_thresholds],
        "utility_thresholds": [float(v) for v in utility_thresholds],
        "top_ks": [int(v) for v in top_ks],
        "fit_months": [str(v) for v in fit_months],
        "holdout_month": str(holdout_month),
        "spec_count": int(len(specs)),
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
    }
    markdown = _write_markdown(
        output_dir=output_dir,
        fit_holdout=fit_holdout,
        selected_by_fit=selected_by_fit,
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
    parser.add_argument("--utility-targets", default=",".join(DEFAULT_UTILITY_TARGETS))
    parser.add_argument("--score-rules", default=",".join(DEFAULT_SCORE_RULES))
    parser.add_argument("--bad-thresholds", default=",".join(str(v) for v in DEFAULT_BAD_THRESHOLDS))
    parser.add_argument("--utility-thresholds", default=",".join(str(v) for v in DEFAULT_UTILITY_THRESHOLDS))
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
        utility_targets=_parse_csv(args.utility_targets),
        score_rules=_parse_csv(args.score_rules),
        bad_thresholds=_parse_float_csv(args.bad_thresholds),
        utility_thresholds=_parse_float_csv(args.utility_thresholds),
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
        "spec_count",
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
