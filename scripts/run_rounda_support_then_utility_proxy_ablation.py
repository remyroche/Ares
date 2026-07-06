#!/usr/bin/env python3
"""Round-A support-then-utility proxy ablation.

This diagnostic tests the Stage 126 follow-up hypothesis before any model
training: learn a causal path-safe executable support proxy first, then rank a
separate utility proxy only inside rows accepted by support.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
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
    _table,
)
from scripts.run_label_first_touch_execution_proxy_ablation import _first_touch_metrics  # noqa: E402
from scripts.run_label_quality_proxy_diagnostics import (  # noqa: E402
    DEFAULT_FEATURE_DIR,
    _feature_columns,
    _json_safe,
    _safe_mean,
    _safe_quantile,
    _spearman,
)
from scripts.run_soft_label_economic_proxy_ablation import (  # noqa: E402
    DEFAULT_EVENT_FEATURE_STORE_FEATURES,
    DEFAULT_PRIOR_WINDOWS_DAYS,
    DEFAULT_STATE_PATH_PRIOR_FEATURES,
)
from scripts.run_soft_label_rounda_topk_proxy_diagnostics import (  # noqa: E402
    _mfe_mae,
    _parse_csv,
    _parse_float_csv,
    _rounda_proxy_score,
    _safe_numeric,
    _topk_positions_by_timestamp,
)


DEFAULT_OUTPUT_DIR = Path("data_perp/reports/rounda_support_then_utility_proxy_stage127_v1")
DEFAULT_MONTHS = ("2026-04", "2026-05", "2026-06")
DEFAULT_FIT_MONTHS = ("2026-04", "2026-05")
DEFAULT_HOLDOUT_MONTH = "2026-06"
DEFAULT_SUPPORT_TARGETS = (
    "pathsafe_noadverse",
    "clean_exec_support",
    "fast_tp_support",
    "bounded_exec_support",
)
DEFAULT_UTILITY_TARGETS = ("net_margin_utility", "ret_net_margin_utility", "fast_net_utility")
DEFAULT_SCORE_RULES = ("utility", "support_blend", "support_product")
DEFAULT_SUPPORT_THRESHOLDS = (0.55, 0.65, 0.75, 0.85)
DEFAULT_UTILITY_THRESHOLDS = (0.0, 0.60)
DEFAULT_TOP_KS = (1, 3, 5, 10)


@dataclass(frozen=True)
class SelectorSpec:
    support_target: str
    utility_target: str
    score_rule: str
    support_threshold: float
    utility_threshold: float
    top_k: int

    @property
    def selector(self) -> str:
        support_part = f"supp{int(round(float(self.support_threshold) * 100)):02d}"
        utility_part = f"util{int(round(float(self.utility_threshold) * 100)):02d}"
        return f"{self.score_rule}_{support_part}_{utility_part}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) | {"selector": self.selector}


def _parse_int_csv(value: str | list[int] | tuple[int, ...], default: tuple[int, ...] = ()) -> list[int]:
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    text = str(value or "").strip()
    if not text:
        return list(default)
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def _sigmoid(values: Any) -> pd.Series:
    arr = np.asarray(values, dtype=np.float64)
    return pd.Series(1.0 / (1.0 + np.exp(-np.clip(arr, -60.0, 60.0))))


def _target_frame(soft: pd.Series, hard: pd.Series) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "target_soft": _safe_numeric(soft).fillna(0.0).clip(0.0, 1.0).astype(np.float32),
            "target_hard": pd.Series(hard, index=soft.index).fillna(False).astype(float).astype(np.float32),
        },
        index=soft.index,
    )


def _support_targets(ft: pd.DataFrame) -> dict[str, pd.DataFrame]:
    idx = ft.index
    u = _safe_numeric(ft["u_policy_net"]).fillna(-0.02)
    hit = _safe_numeric(ft["first_touch_hit"]).fillna(0.0).clip(0.0, 1.0)
    stop = _safe_numeric(ft["first_touch_stop"]).fillna(0.0).clip(0.0, 1.0)
    timeout = _safe_numeric(ft["first_touch_timeout"]).fillna(0.0).clip(0.0, 1.0)
    same_bar = _safe_numeric(ft["first_touch_same_bar"]).fillna(0.0).clip(0.0, 1.0)
    bar = _safe_numeric(ft["first_touch_bar"]).fillna(36.0).clip(lower=0.0)
    mae_to_sl = _safe_numeric(ft["first_touch_mae_to_sl"]).fillna(10.0).clip(lower=0.0)
    mfe_to_tp = _safe_numeric(ft["first_touch_mfe_to_tp"]).fillna(0.0).clip(lower=0.0)
    barrier = _safe_numeric(ft["barrier"]).fillna(1.0).clip(lower=0.0)
    mae_norm = _safe_numeric(ft["mae_norm"]).fillna(10.0).clip(lower=0.0)
    mfe_norm = _safe_numeric(ft["mfe_norm"]).fillna(0.0).clip(lower=0.0)
    mfe_mae = _mfe_mae(ft).fillna(0.0)

    no_adverse = (
        (1.0 - stop)
        * (1.0 - timeout)
        * (1.0 - same_bar)
        * _sigmoid((1.00 - mae_to_sl) / 0.20).set_axis(idx)
        * _sigmoid((0.027 - barrier) / 0.006).set_axis(idx)
        * _sigmoid((14.0 - bar) / 4.0).set_axis(idx)
    ).clip(0.0, 1.0)
    clean_exec = (
        hit
        * no_adverse
        * _sigmoid((u - 0.0010) / 0.006).set_axis(idx)
        * _sigmoid((0.85 - mae_to_sl) / 0.18).set_axis(idx)
        * _sigmoid((12.0 - bar) / 3.0).set_axis(idx)
    ).clip(0.0, 1.0)
    fast_tp = (
        hit
        * (1.0 - stop)
        * (1.0 - timeout)
        * (1.0 - same_bar)
        * _sigmoid((0.75 - mae_to_sl) / 0.18).set_axis(idx)
        * _sigmoid((0.025 - barrier) / 0.005).set_axis(idx)
        * _sigmoid((8.0 - bar) / 2.5).set_axis(idx)
        * _sigmoid((mfe_to_tp - 1.0) / 0.25).set_axis(idx)
    ).clip(0.0, 1.0)
    bounded = (
        _sigmoid((u - 0.0010) / 0.006).set_axis(idx)
        * _sigmoid((1.50 - mae_norm) / 0.35).set_axis(idx)
        * _sigmoid((0.030 - barrier) / 0.007).set_axis(idx)
        * _sigmoid((mfe_mae - 1.05) / 0.30).set_axis(idx)
        * _sigmoid((mfe_norm - 1.00) / 0.35).set_axis(idx)
        * _sigmoid((18.0 - bar) / 5.0).set_axis(idx)
        * (1.0 - timeout)
    ).clip(0.0, 1.0)

    return {
        "pathsafe_noadverse": _target_frame(
            no_adverse,
            (stop <= 0.0)
            & (timeout <= 0.0)
            & (same_bar <= 0.0)
            & (mae_to_sl <= 1.0)
            & (barrier <= 0.027)
            & (bar <= 14.0),
        ),
        "clean_exec_support": _target_frame(clean_exec, _safe_numeric(ft["clean_exec_actual"]).gt(0.0)),
        "fast_tp_support": _target_frame(
            fast_tp,
            (hit > 0.5)
            & (stop <= 0.0)
            & (timeout <= 0.0)
            & (same_bar <= 0.0)
            & (mae_to_sl <= 0.75)
            & (barrier <= 0.025)
            & (bar <= 8.0)
            & (mfe_to_tp >= 1.0),
        ),
        "bounded_exec_support": _target_frame(
            bounded,
            (u > 0.0010)
            & (mae_norm <= 1.50)
            & (barrier <= 0.030)
            & (mfe_mae >= 1.05)
            & (mfe_norm >= 1.00)
            & (bar <= 18.0)
            & (timeout <= 0.0),
        ),
    }


def _utility_targets(ft: pd.DataFrame) -> dict[str, pd.Series]:
    idx = ft.index
    u = _safe_numeric(ft["u_policy_net"]).fillna(-0.02)
    ret_net = _safe_numeric(ft["ret_net"]).fillna(-0.02)
    bar = _safe_numeric(ft["first_touch_bar"]).fillna(36.0).clip(lower=0.0)
    mfe_to_tp = _safe_numeric(ft["first_touch_mfe_to_tp"]).fillna(0.0).clip(lower=0.0)
    net_margin = _sigmoid((u - 0.0030) / 0.006).set_axis(idx).clip(0.0, 1.0)
    ret_margin = _sigmoid((ret_net - 0.0030) / 0.006).set_axis(idx).clip(0.0, 1.0)
    fast = (
        net_margin
        * _sigmoid((12.0 - bar) / 4.0).set_axis(idx)
        * _sigmoid((mfe_to_tp - 1.0) / 0.25).set_axis(idx)
    ).clip(0.0, 1.0)
    return {
        "net_margin_utility": net_margin.astype(np.float32),
        "ret_net_margin_utility": ret_margin.astype(np.float32),
        "fast_net_utility": fast.astype(np.float32),
    }


def _selected_positions(frame: pd.DataFrame, score: pd.Series, top_k: int) -> np.ndarray:
    selections = _topk_positions_by_timestamp(frame, score, int(top_k), min_score=-1.0)
    if not selections:
        return np.array([], dtype=np.int64)
    return np.concatenate([idx for _, idx in selections]).astype(np.int64, copy=False)


def _make_score(
    *,
    spec: SelectorSpec,
    support_proxy: pd.Series,
    utility_proxy: pd.Series,
) -> pd.Series:
    support = _safe_numeric(support_proxy).reset_index(drop=True)
    utility = _safe_numeric(utility_proxy).reset_index(drop=True)
    if spec.score_rule == "utility":
        score = utility
    elif spec.score_rule == "support_blend":
        score = (0.65 * utility + 0.35 * support).clip(0.0, 1.0)
    elif spec.score_rule == "support_product":
        score = (utility * support).clip(0.0, 1.0)
    else:
        raise ValueError(f"Unknown score rule: {spec.score_rule}")
    gate = support >= float(spec.support_threshold)
    if float(spec.utility_threshold) > 0.0:
        gate = gate & (utility >= float(spec.utility_threshold))
    return score.where(gate)


def _row_metrics(
    *,
    frame: pd.DataFrame,
    ft: pd.DataFrame,
    support_target: pd.DataFrame,
    utility_target: pd.Series,
    score: pd.Series,
    spec: SelectorSpec,
    month: str,
    period_type: str,
    period: str,
    pos: np.ndarray,
) -> dict[str, Any]:
    local_frame = frame.iloc[pos].reset_index(drop=True)
    local_ft = ft.iloc[pos].reset_index(drop=True)
    local_support = support_target.iloc[pos].reset_index(drop=True)
    local_utility = utility_target.iloc[pos].reset_index(drop=True)
    local_score = _safe_numeric(score).iloc[pos].reset_index(drop=True)
    selected_idx = _selected_positions(local_frame, local_score, int(spec.top_k))
    selected = local_ft.iloc[selected_idx] if len(selected_idx) else local_ft.iloc[:0]
    selected_frame = local_frame.iloc[selected_idx] if len(selected_idx) else local_frame.iloc[:0]
    selected_support = local_support.iloc[selected_idx] if len(selected_idx) else local_support.iloc[:0]
    selected_utility = local_utility.iloc[selected_idx] if len(selected_idx) else local_utility.iloc[:0]
    ret = _safe_numeric(selected.get("ret_net"))
    u = _safe_numeric(selected.get("u_policy_net"))
    mae = _safe_numeric(selected.get("mae_norm"))
    ft_mae = _safe_numeric(selected.get("first_touch_mae_to_sl"))
    timeout = _safe_numeric(selected.get("first_touch_timeout"))
    same_bar = _safe_numeric(selected.get("first_touch_same_bar"))
    stop = _safe_numeric(selected.get("first_touch_stop"))
    barrier = _safe_numeric(selected.get("barrier"))
    mfe_mae = _mfe_mae(selected) if len(selected) else pd.Series(dtype=float)
    symbols = selected_frame["__symbol__"].astype(str) if len(selected_frame) else pd.Series(dtype=object)
    return {
        "month": str(month),
        "period_type": str(period_type),
        "period": str(period),
        "selector": spec.selector,
        "support_target": spec.support_target,
        "utility_target": spec.utility_target,
        "score_rule": spec.score_rule,
        "support_threshold": float(spec.support_threshold),
        "utility_threshold": float(spec.utility_threshold),
        "top_k": int(spec.top_k),
        "rows": int(len(local_frame)),
        "selected_rows": int(len(selected_idx)),
        "timestamp_count": int(pd.to_datetime(local_frame["__ts__"], errors="coerce").nunique(dropna=True)),
        "accepted_timestamp_count": int(
            pd.to_datetime(local_frame.loc[local_score.notna(), "__ts__"], errors="coerce").nunique(dropna=True)
        ),
        "candidate_rate": _safe_mean(local_score.notna()),
        "mean_u": _safe_mean(u),
        "hit_u": _safe_mean(u > 0.0),
        "mean_return_net": _safe_mean(ret),
        "q05_return_net": _safe_quantile(ret, 0.05),
        "net_pnl": float(ret.sum(skipna=True)) if len(ret) else 0.0,
        "target_support_soft_mean": _safe_mean(selected_support.get("target_soft")),
        "target_support_hard_rate": _safe_mean(selected_support.get("target_hard")),
        "target_utility_soft_mean": _safe_mean(selected_utility),
        "clean_exec_actual_rate": _safe_mean(selected.get("clean_exec_actual")),
        "first_touch_hit_rate": _safe_mean(selected.get("first_touch_hit")),
        "first_touch_stop_rate": _safe_mean(stop),
        "first_touch_timeout_rate": _safe_mean(timeout),
        "first_touch_same_bar_rate": _safe_mean(same_bar),
        "first_touch_bad_mae_to_sl_rate": _safe_mean(ft_mae >= 1.0),
        "p90_first_touch_mae_to_sl": _safe_quantile(ft_mae, 0.90),
        "bad_mae_1r_rate": _safe_mean(mae >= 1.0),
        "p90_mae_norm": _safe_quantile(mae, 0.90),
        "wide_barrier_25bps_rate": _safe_mean(barrier > 0.025),
        "timeout_rate": _safe_mean(timeout > 0.0),
        "mean_mfe_mae_ratio": _safe_mean(mfe_mae),
        "symbol_count": int(symbols.nunique(dropna=True)) if len(symbols) else 0,
        "top_symbol_share": float(symbols.value_counts(normalize=True).iloc[0]) if len(symbols) else 0.0,
        "score_ic_u": _spearman(local_score, local_ft["u_policy_net"]),
        "score_ic_support": _spearman(local_score, local_support["target_soft"]),
        "score_ic_bad_mae": _spearman(local_score, (local_ft["mae_norm"] >= 1.0).astype(float)),
        "score_ic_timeout": _spearman(local_score, local_ft["first_touch_timeout"]),
    }


def _period_positions(frame: pd.DataFrame) -> list[tuple[str, str, np.ndarray]]:
    out: list[tuple[str, str, np.ndarray]] = [("month", "month", np.arange(len(frame), dtype=np.int64))]
    weeks = pd.to_datetime(frame["__ts__"], errors="coerce").dt.to_period("W-SUN").astype(str)
    for week, ids in pd.Series(np.arange(len(frame), dtype=np.int64)).groupby(weeks, sort=False):
        out.append(("week", str(week), ids.to_numpy(dtype=np.int64)))
    return out


def _weighted_mean(frame: pd.DataFrame, col: str, weight_col: str = "selected_rows") -> float:
    if frame.empty or col not in frame.columns:
        return float("nan")
    values = _safe_numeric(frame[col])
    weights = _safe_numeric(frame.get(weight_col, pd.Series(1.0, index=frame.index))).fillna(0.0)
    mask = values.notna() & weights.gt(0.0)
    return float(np.average(values[mask], weights=weights[mask])) if bool(mask.any()) else float("nan")


def _fit_holdout_summary(period_rows: pd.DataFrame, *, fit_months: list[str], holdout_month: str, min_week_rows: int) -> pd.DataFrame:
    if period_rows.empty:
        return pd.DataFrame()
    monthly = period_rows[period_rows["period_type"].eq("month")].copy()
    weekly = period_rows[period_rows["period_type"].eq("week")].copy()
    rows: list[dict[str, Any]] = []
    keys = ["selector", "support_target", "utility_target", "score_rule", "support_threshold", "utility_threshold", "top_k"]
    for key, group in monthly.groupby(keys, dropna=False, observed=True):
        selector, support_target, utility_target, score_rule, support_threshold, utility_threshold, top_k = key
        fit = group[group["month"].astype(str).isin(fit_months)].copy()
        holdout = group[group["month"].astype(str).eq(str(holdout_month))].copy()
        if fit.empty or holdout.empty:
            continue
        wgroup = weekly[
            weekly["selector"].astype(str).eq(str(selector))
            & weekly["support_target"].astype(str).eq(str(support_target))
            & weekly["utility_target"].astype(str).eq(str(utility_target))
            & _safe_numeric(weekly["top_k"]).eq(int(top_k))
            & _safe_numeric(weekly["support_threshold"]).eq(float(support_threshold))
            & _safe_numeric(weekly["utility_threshold"]).eq(float(utility_threshold))
        ].copy()
        fit_week = wgroup[wgroup["month"].astype(str).isin(fit_months)]
        holdout_week = wgroup[wgroup["month"].astype(str).eq(str(holdout_month))]

        def week_stats(frame: pd.DataFrame, col: str) -> tuple[int, float, float]:
            selected = _safe_numeric(frame["selected_rows"]).fillna(0.0)
            material = selected.ge(int(min_week_rows))
            values = _safe_numeric(frame[col])
            if not bool(material.any()):
                return 0, float("nan"), float("nan")
            return int(material.sum()), float((values.gt(0.0) & material).sum() / material.sum()), _safe_quantile(values[material], 0.25)

        fit_material_weeks, fit_week_return_rate, fit_q25_week_return = week_stats(fit_week, "mean_return_net")
        holdout_material_weeks, holdout_week_return_rate, holdout_q25_week_return = week_stats(
            holdout_week,
            "mean_return_net",
        )
        fit_returns = _safe_numeric(fit["mean_return_net"])
        holdout_returns = _safe_numeric(holdout["mean_return_net"])
        row: dict[str, Any] = {
            "selector": str(selector),
            "support_target": str(support_target),
            "utility_target": str(utility_target),
            "score_rule": str(score_rule),
            "support_threshold": float(support_threshold),
            "utility_threshold": float(utility_threshold),
            "top_k": int(top_k),
            "fit_mean_return_net": _safe_mean(fit_returns),
            "fit_worst_return_net": _safe_quantile(fit_returns, 0.0),
            "holdout_mean_return_net": _safe_mean(holdout_returns),
            "fit_positive_months": int(fit_returns.gt(0.0).sum()),
            "holdout_positive_months": int(holdout_returns.gt(0.0).sum()),
            "fit_material_weeks": fit_material_weeks,
            "holdout_material_weeks": holdout_material_weeks,
            "fit_material_positive_week_rate": fit_week_return_rate,
            "holdout_material_positive_week_rate": holdout_week_return_rate,
            "fit_q25_week_return_net": fit_q25_week_return,
            "holdout_q25_week_return_net": holdout_q25_week_return,
            "fit_selected_rows": int(_safe_numeric(fit["selected_rows"]).sum(skipna=True)),
            "holdout_selected_rows": int(_safe_numeric(holdout["selected_rows"]).sum(skipna=True)),
            "fit_candidate_rate": _safe_mean(fit["candidate_rate"]),
            "holdout_candidate_rate": _safe_mean(holdout["candidate_rate"]),
            "fit_clean_exec_actual_rate": _weighted_mean(fit, "clean_exec_actual_rate"),
            "holdout_clean_exec_actual_rate": _weighted_mean(holdout, "clean_exec_actual_rate"),
            "fit_first_touch_bad_mae_to_sl_rate": _weighted_mean(fit, "first_touch_bad_mae_to_sl_rate"),
            "holdout_first_touch_bad_mae_to_sl_rate": _weighted_mean(holdout, "first_touch_bad_mae_to_sl_rate"),
            "fit_p90_first_touch_mae_to_sl": _weighted_mean(fit, "p90_first_touch_mae_to_sl"),
            "holdout_p90_first_touch_mae_to_sl": _weighted_mean(holdout, "p90_first_touch_mae_to_sl"),
            "fit_timeout_rate": _weighted_mean(fit, "first_touch_timeout_rate"),
            "holdout_timeout_rate": _weighted_mean(holdout, "first_touch_timeout_rate"),
            "fit_same_bar_rate": _weighted_mean(fit, "first_touch_same_bar_rate"),
            "holdout_same_bar_rate": _weighted_mean(holdout, "first_touch_same_bar_rate"),
            "fit_bad_mae_1r_rate": _weighted_mean(fit, "bad_mae_1r_rate"),
            "holdout_bad_mae_1r_rate": _weighted_mean(holdout, "bad_mae_1r_rate"),
            "fit_p90_mae_norm": _weighted_mean(fit, "p90_mae_norm"),
            "holdout_p90_mae_norm": _weighted_mean(holdout, "p90_mae_norm"),
            "fit_score_ic_u": _safe_mean(fit["score_ic_u"]),
            "holdout_score_ic_u": _safe_mean(holdout["score_ic_u"]),
        }
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
        fit_econ = (
            fit_sign
            and row["fit_score_ic_u"] > 0.0
            and row["fit_clean_exec_actual_rate"] >= 0.35
            and row["fit_first_touch_bad_mae_to_sl_rate"] <= 0.40
            and row["fit_p90_first_touch_mae_to_sl"] <= 2.0
            and row["fit_timeout_rate"] <= 0.50
            and row["fit_same_bar_rate"] <= 0.20
        )
        holdout_econ = (
            holdout_sign
            and row["holdout_score_ic_u"] > 0.0
            and row["holdout_clean_exec_actual_rate"] >= 0.35
            and row["holdout_first_touch_bad_mae_to_sl_rate"] <= 0.40
            and row["holdout_p90_first_touch_mae_to_sl"] <= 2.0
            and row["holdout_timeout_rate"] <= 0.50
            and row["holdout_same_bar_rate"] <= 0.20
        )
        row["fit_sign_pass"] = bool(fit_sign)
        row["holdout_sign_pass"] = bool(holdout_sign)
        row["fit_economic_pass"] = bool(fit_econ)
        row["holdout_economic_pass"] = bool(holdout_econ)
        row["trainworthy_pass"] = bool(fit_econ and holdout_econ)
        row["objective"] = (
            (row["fit_mean_return_net"] if pd.notna(row["fit_mean_return_net"]) else -1.0)
            + 0.50 * (row["fit_q25_week_return_net"] if pd.notna(row["fit_q25_week_return_net"]) else -1.0)
            + 0.25 * (row["fit_clean_exec_actual_rate"] if pd.notna(row["fit_clean_exec_actual_rate"]) else 0.0)
            - 0.15 * (row["fit_first_touch_bad_mae_to_sl_rate"] if pd.notna(row["fit_first_touch_bad_mae_to_sl_rate"]) else 1.0)
            - 0.10 * (row["fit_timeout_rate"] if pd.notna(row["fit_timeout_rate"]) else 1.0)
        )
        rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(
        ["trainworthy_pass", "holdout_economic_pass", "fit_economic_pass", "holdout_mean_return_net", "objective"],
        ascending=[False, False, False, False, False],
    )


def _write_markdown(
    *,
    output_dir: Path,
    summary: pd.DataFrame,
    monthly: pd.DataFrame,
    proxy_ic: pd.DataFrame,
    manifest: dict[str, Any],
) -> Path:
    path = output_dir / "rounda_support_then_utility_proxy_ablation.md"
    summary_cols = [
        "trainworthy_pass",
        "fit_economic_pass",
        "holdout_economic_pass",
        "selector",
        "support_target",
        "utility_target",
        "top_k",
        "support_threshold",
        "utility_threshold",
        "fit_mean_return_net",
        "holdout_mean_return_net",
        "fit_selected_rows",
        "holdout_selected_rows",
        "fit_clean_exec_actual_rate",
        "holdout_clean_exec_actual_rate",
        "fit_first_touch_bad_mae_to_sl_rate",
        "holdout_first_touch_bad_mae_to_sl_rate",
        "fit_p90_first_touch_mae_to_sl",
        "holdout_p90_first_touch_mae_to_sl",
        "fit_timeout_rate",
        "holdout_timeout_rate",
        "fit_score_ic_u",
        "holdout_score_ic_u",
    ]
    monthly_cols = [
        "month",
        "selector",
        "support_target",
        "utility_target",
        "top_k",
        "selected_rows",
        "candidate_rate",
        "mean_return_net",
        "net_pnl",
        "clean_exec_actual_rate",
        "first_touch_bad_mae_to_sl_rate",
        "p90_first_touch_mae_to_sl",
        "first_touch_timeout_rate",
        "score_ic_u",
        "score_ic_support",
    ]
    proxy_cols = [
        "month",
        "target_kind",
        "target_name",
        "proxy_mean_train_target_ic",
        "proxy_mean_train_utility_ic",
        "proxy_mean_train_bad_mae_ic",
        "proxy_mean_train_timeout_ic",
        "proxy_features",
    ]
    month_view = monthly[monthly["period_type"].eq("month")].copy()
    lines = [
        "# Round-A Support-Then-Utility Proxy Ablation",
        "",
        "Scope: proxy-only diagnostic. It learns support and utility proxies separately, then ranks utility inside support.",
        "",
        f"Fit months: `{', '.join(manifest['fit_months'])}`. Holdout: `{manifest['holdout_month']}`.",
        f"Support targets: `{', '.join(manifest['support_targets'])}`.",
        f"Utility targets: `{', '.join(manifest['utility_targets'])}`.",
        f"Features: `{manifest['feature_count']}`. Proxy objective: `{manifest['proxy_objective']}`. Proxy top-k: `{manifest['proxy_top_k']}`.",
        "",
        "## Fit/Holdout Summary",
        "",
        _table(summary, summary_cols, limit=100),
        "",
        "## Month Detail",
        "",
        _table(month_view.sort_values(["month", "selector", "support_target", "utility_target", "top_k"]), monthly_cols, limit=180),
        "",
        "## Proxy Features",
        "",
        _table(proxy_ic, proxy_cols, limit=120),
        "",
        "## Outputs",
        "",
        f"- Fit/holdout: `{manifest['outputs']['fit_holdout']}`",
        f"- Monthly/weekly rows: `{manifest['outputs']['period_rows']}`",
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
    months: list[str],
    fit_months: list[str],
    holdout_month: str,
    support_targets: list[str],
    utility_targets: list[str],
    score_rules: list[str],
    support_thresholds: list[float],
    utility_thresholds: list[float],
    top_ks: list[int],
    proxy_top_k: int,
    proxy_objective: str,
    proxy_min_target_ic: float,
    proxy_min_utility_ic: float,
    proxy_max_bad_mae_ic: float,
    proxy_max_wide_ic: float,
    proxy_max_timeout_ic: float,
    proxy_utility_weight: float,
    proxy_bad_mae_weight: float,
    proxy_wide_weight: float,
    proxy_timeout_weight: float,
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
    ft = _first_touch_metrics(frame, metrics)
    support_map = _support_targets(ft)
    utility_map = _utility_targets(ft)
    missing_support = sorted(set(support_targets) - set(support_map))
    missing_utility = sorted(set(utility_targets) - set(utility_map))
    if missing_support:
        raise ValueError(f"Unknown support targets: {missing_support}")
    if missing_utility:
        raise ValueError(f"Unknown utility targets: {missing_utility}")
    features = _feature_columns(frame)
    specs = [
        SelectorSpec(
            support_target=support_target,
            utility_target=utility_target,
            score_rule=score_rule,
            support_threshold=float(support_threshold),
            utility_threshold=float(utility_threshold),
            top_k=int(top_k),
        )
        for support_target in support_targets
        for utility_target in utility_targets
        for score_rule in score_rules
        for support_threshold in support_thresholds
        for utility_threshold in utility_thresholds
        for top_k in top_ks
    ]

    month_series = frame["__ts__"].dt.to_period("M").astype(str)
    rows: list[dict[str, Any]] = []
    proxy_rows: list[dict[str, Any]] = []

    for month in months:
        train_mask = month_series.lt(str(month))
        valid_mask = month_series.eq(str(month))
        if int(train_mask.sum()) < 500 or int(valid_mask.sum()) < 100:
            continue
        train = frame.loc[train_mask].copy()
        train_metrics = ft.loc[train_mask].copy()
        valid_raw = frame.loc[valid_mask].copy()
        valid = valid_raw.reset_index(drop=True)
        valid_ft = ft.loc[valid_mask].copy().reset_index(drop=True)
        support_proxy: dict[str, tuple[pd.Series, dict[str, Any]]] = {}
        utility_proxy: dict[str, tuple[pd.Series, dict[str, Any]]] = {}

        for name in support_targets:
            proxy, diag = _rounda_proxy_score(
                train=train,
                valid=valid_raw,
                features=features,
                y_train=support_map[name].loc[train_mask, "target_soft"],
                metrics_train=train_metrics,
                proxy_top_k=proxy_top_k,
                proxy_objective=proxy_objective,
                proxy_min_target_ic=proxy_min_target_ic,
                proxy_min_utility_ic=proxy_min_utility_ic,
                proxy_max_bad_mae_ic=proxy_max_bad_mae_ic,
                proxy_max_wide_ic=proxy_max_wide_ic,
                proxy_max_timeout_ic=proxy_max_timeout_ic,
                proxy_utility_weight=proxy_utility_weight,
                proxy_bad_mae_weight=proxy_bad_mae_weight,
                proxy_wide_weight=proxy_wide_weight,
                proxy_timeout_weight=proxy_timeout_weight,
            )
            proxy = proxy.reset_index(drop=True)
            support_proxy[name] = (proxy, diag)
            valid_support = support_map[name].loc[valid_mask].reset_index(drop=True)
            proxy_rows.append(
                {
                    "month": str(month),
                    "target_kind": "support",
                    "target_name": str(name),
                    "proxy_ic_target": _spearman(proxy, valid_support["target_soft"]),
                    "proxy_ic_u": _spearman(proxy, valid_ft["u_policy_net"]),
                    "proxy_ic_bad_mae": _spearman(proxy, (valid_ft["mae_norm"] >= 1.0).astype(float)),
                    "proxy_ic_timeout": _spearman(proxy, valid_ft["first_touch_timeout"]),
                    "proxy_mean_train_target_ic": float(diag.get("proxy_mean_train_target_ic", np.nan)),
                    "proxy_mean_train_utility_ic": float(diag.get("proxy_mean_train_utility_ic", np.nan)),
                    "proxy_mean_train_bad_mae_ic": float(diag.get("proxy_mean_train_bad_mae_ic", np.nan)),
                    "proxy_mean_train_timeout_ic": float(diag.get("proxy_mean_train_timeout_ic", np.nan)),
                    "proxy_features": ",".join(diag.get("proxy_features", [])),
                }
            )
        for name in utility_targets:
            proxy, diag = _rounda_proxy_score(
                train=train,
                valid=valid_raw,
                features=features,
                y_train=utility_map[name].loc[train_mask],
                metrics_train=train_metrics,
                proxy_top_k=proxy_top_k,
                proxy_objective=proxy_objective,
                proxy_min_target_ic=proxy_min_target_ic,
                proxy_min_utility_ic=proxy_min_utility_ic,
                proxy_max_bad_mae_ic=proxy_max_bad_mae_ic,
                proxy_max_wide_ic=proxy_max_wide_ic,
                proxy_max_timeout_ic=proxy_max_timeout_ic,
                proxy_utility_weight=proxy_utility_weight,
                proxy_bad_mae_weight=proxy_bad_mae_weight,
                proxy_wide_weight=proxy_wide_weight,
                proxy_timeout_weight=proxy_timeout_weight,
            )
            proxy = proxy.reset_index(drop=True)
            utility_proxy[name] = (proxy, diag)
            valid_utility = utility_map[name].loc[valid_mask].reset_index(drop=True)
            proxy_rows.append(
                {
                    "month": str(month),
                    "target_kind": "utility",
                    "target_name": str(name),
                    "proxy_ic_target": _spearman(proxy, valid_utility),
                    "proxy_ic_u": _spearman(proxy, valid_ft["u_policy_net"]),
                    "proxy_ic_bad_mae": _spearman(proxy, (valid_ft["mae_norm"] >= 1.0).astype(float)),
                    "proxy_ic_timeout": _spearman(proxy, valid_ft["first_touch_timeout"]),
                    "proxy_mean_train_target_ic": float(diag.get("proxy_mean_train_target_ic", np.nan)),
                    "proxy_mean_train_utility_ic": float(diag.get("proxy_mean_train_utility_ic", np.nan)),
                    "proxy_mean_train_bad_mae_ic": float(diag.get("proxy_mean_train_bad_mae_ic", np.nan)),
                    "proxy_mean_train_timeout_ic": float(diag.get("proxy_mean_train_timeout_ic", np.nan)),
                    "proxy_features": ",".join(diag.get("proxy_features", [])),
                }
            )

        period_slices = _period_positions(valid)
        for spec in specs:
            support_score, _ = support_proxy[spec.support_target]
            utility_score, _ = utility_proxy[spec.utility_target]
            score = _make_score(spec=spec, support_proxy=support_score, utility_proxy=utility_score)
            valid_support = support_map[spec.support_target].loc[valid_mask].reset_index(drop=True)
            valid_utility = utility_map[spec.utility_target].loc[valid_mask].reset_index(drop=True)
            for period_type, period_name, pos in period_slices:
                rows.append(
                    _row_metrics(
                        frame=valid,
                        ft=valid_ft,
                        support_target=valid_support,
                        utility_target=valid_utility,
                        score=score,
                        spec=spec,
                        month=str(month),
                        period_type=period_type,
                        period=str(period_name if period_type == "week" else month),
                        pos=pos,
                    )
                )
        print(json.dumps({"month": str(month), "progress": "complete", "rows": len(rows)}))

    period_rows = pd.DataFrame(rows)
    proxy_ic = pd.DataFrame(proxy_rows)
    fit_holdout = _fit_holdout_summary(
        period_rows,
        fit_months=fit_months,
        holdout_month=holdout_month,
        min_week_rows=int(min_week_rows),
    )
    paths = {
        "period_rows": output_dir / "rounda_support_then_utility_period_rows.csv",
        "fit_holdout": output_dir / "rounda_support_then_utility_fit_holdout.csv",
        "proxy_ic": output_dir / "rounda_support_then_utility_proxy_ic.csv",
        "manifest": output_dir / "manifest.json",
    }
    period_rows.to_csv(paths["period_rows"], index=False)
    fit_holdout.to_csv(paths["fit_holdout"], index=False)
    proxy_ic.to_csv(paths["proxy_ic"], index=False)
    manifest = {
        "scope": "rounda_support_then_utility_proxy_ablation",
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
        "months": list(months),
        "fit_months": list(fit_months),
        "holdout_month": str(holdout_month),
        "support_targets": list(support_targets),
        "utility_targets": list(utility_targets),
        "score_rules": list(score_rules),
        "support_thresholds": [float(v) for v in support_thresholds],
        "utility_thresholds": [float(v) for v in utility_thresholds],
        "top_ks": [int(v) for v in top_ks],
        "spec_count": int(len(specs)),
        "proxy_top_k": int(proxy_top_k),
        "proxy_objective": str(proxy_objective),
        "proxy_min_target_ic": float(proxy_min_target_ic),
        "proxy_min_utility_ic": float(proxy_min_utility_ic),
        "proxy_max_bad_mae_ic": float(proxy_max_bad_mae_ic),
        "proxy_max_wide_ic": float(proxy_max_wide_ic),
        "proxy_max_timeout_ic": float(proxy_max_timeout_ic),
        "proxy_utility_weight": float(proxy_utility_weight),
        "proxy_bad_mae_weight": float(proxy_bad_mae_weight),
        "proxy_wide_weight": float(proxy_wide_weight),
        "proxy_timeout_weight": float(proxy_timeout_weight),
        "min_week_rows": int(min_week_rows),
        "reports": reports,
        "outputs": {key: str(value) for key, value in paths.items()},
    }
    markdown = _write_markdown(
        output_dir=output_dir,
        summary=fit_holdout,
        monthly=period_rows,
        proxy_ic=proxy_ic,
        manifest={**manifest, "outputs": {**manifest["outputs"], "markdown": str(output_dir / "rounda_support_then_utility_proxy_ablation.md")}},
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
    parser.add_argument("--max-feature-store-features", type=int, default=498)
    parser.add_argument("--months", default=",".join(DEFAULT_MONTHS))
    parser.add_argument("--fit-months", default=",".join(DEFAULT_FIT_MONTHS))
    parser.add_argument("--holdout-month", default=DEFAULT_HOLDOUT_MONTH)
    parser.add_argument("--support-targets", default=",".join(DEFAULT_SUPPORT_TARGETS))
    parser.add_argument("--utility-targets", default=",".join(DEFAULT_UTILITY_TARGETS))
    parser.add_argument("--score-rules", default=",".join(DEFAULT_SCORE_RULES))
    parser.add_argument("--support-thresholds", default=",".join(str(v) for v in DEFAULT_SUPPORT_THRESHOLDS))
    parser.add_argument("--utility-thresholds", default=",".join(str(v) for v in DEFAULT_UTILITY_THRESHOLDS))
    parser.add_argument("--top-ks", default=",".join(str(v) for v in DEFAULT_TOP_KS))
    parser.add_argument("--proxy-top-k", type=int, default=DEFAULT_PROXY_TOP_K)
    parser.add_argument("--proxy-objective", choices=("target_ic", "economic_ic", "economic_score"), default="economic_ic")
    parser.add_argument("--proxy-min-target-ic", type=float, default=0.0)
    parser.add_argument("--proxy-min-utility-ic", type=float, default=0.0)
    parser.add_argument("--proxy-max-bad-mae-ic", type=float, default=0.0)
    parser.add_argument("--proxy-max-wide-ic", type=float, default=0.0)
    parser.add_argument("--proxy-max-timeout-ic", type=float, default=0.0)
    parser.add_argument("--proxy-utility-weight", type=float, default=1.0)
    parser.add_argument("--proxy-bad-mae-weight", type=float, default=1.0)
    parser.add_argument("--proxy-wide-weight", type=float, default=0.5)
    parser.add_argument("--proxy-timeout-weight", type=float, default=0.5)
    parser.add_argument("--min-week-rows", type=int, default=3)
    parser.add_argument("--include-causal-outcome-priors", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-causal-state-path-priors", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-event-confirmation-features", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-adverse-path-composites", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prior-windows-days", default=",".join(str(v) for v in DEFAULT_PRIOR_WINDOWS_DAYS))
    parser.add_argument("--prior-embargo-hours", type=float, default=24.0)
    parser.add_argument("--state-path-prior-features", default=",".join(DEFAULT_STATE_PATH_PRIOR_FEATURES))
    parser.add_argument("--event-feature-store-features", default=",".join(DEFAULT_EVENT_FEATURE_STORE_FEATURES))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = run_report(
        labels_path=args.labels_path,
        output_dir=args.output_dir,
        feature_dir=args.feature_dir,
        feature_list_csv=args.feature_list_csv,
        max_feature_store_features=args.max_feature_store_features,
        months=_parse_csv(args.months, DEFAULT_MONTHS),
        fit_months=_parse_csv(args.fit_months, DEFAULT_FIT_MONTHS),
        holdout_month=str(args.holdout_month),
        support_targets=_parse_csv(args.support_targets, DEFAULT_SUPPORT_TARGETS),
        utility_targets=_parse_csv(args.utility_targets, DEFAULT_UTILITY_TARGETS),
        score_rules=_parse_csv(args.score_rules, DEFAULT_SCORE_RULES),
        support_thresholds=_parse_float_csv(args.support_thresholds),
        utility_thresholds=_parse_float_csv(args.utility_thresholds),
        top_ks=_parse_int_csv(args.top_ks, DEFAULT_TOP_KS),
        proxy_top_k=int(args.proxy_top_k),
        proxy_objective=str(args.proxy_objective),
        proxy_min_target_ic=float(args.proxy_min_target_ic),
        proxy_min_utility_ic=float(args.proxy_min_utility_ic),
        proxy_max_bad_mae_ic=float(args.proxy_max_bad_mae_ic),
        proxy_max_wide_ic=float(args.proxy_max_wide_ic),
        proxy_max_timeout_ic=float(args.proxy_max_timeout_ic),
        proxy_utility_weight=float(args.proxy_utility_weight),
        proxy_bad_mae_weight=float(args.proxy_bad_mae_weight),
        proxy_wide_weight=float(args.proxy_wide_weight),
        proxy_timeout_weight=float(args.proxy_timeout_weight),
        min_week_rows=int(args.min_week_rows),
        include_causal_outcome_priors=bool(args.include_causal_outcome_priors),
        include_causal_state_path_priors=bool(args.include_causal_state_path_priors),
        include_event_confirmation_features=bool(args.include_event_confirmation_features),
        include_adverse_path_composites=bool(args.include_adverse_path_composites),
        prior_windows_days=_parse_float_csv(args.prior_windows_days),
        prior_embargo_hours=float(args.prior_embargo_hours),
        state_path_prior_features=_parse_csv(args.state_path_prior_features, DEFAULT_STATE_PATH_PRIOR_FEATURES),
        event_feature_store_features=_parse_csv(args.event_feature_store_features, DEFAULT_EVENT_FEATURE_STORE_FEATURES),
    )
    print(json.dumps(_json_safe({"output_dir": manifest["output_dir"], "outputs": manifest["outputs"]}), indent=2))


if __name__ == "__main__":
    main()
