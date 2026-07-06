#!/usr/bin/env python3
"""Support-aware path-decisive label ablation.

Proxy-only diagnostic. This tests whether making causal state support and path
decisiveness part of the target itself produces labels that are learnable and
profitable after costs before any base/meta training.
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
    _score_period,
    _score_proxy,
    _slice_week_positions,
    _table,
)
from scripts.run_label_quality_proxy_diagnostics import (  # noqa: E402
    DEFAULT_FEATURE_DIR,
    _feature_columns,
    _json_safe,
    _safe_mean,
    _safe_quantile,
    _spearman,
)
from scripts.run_path_aware_label_target_grid import _sigmoid  # noqa: E402
from scripts.run_soft_label_candidate_source_ablation import (  # noqa: E402
    _causal_time_edge_prior_features,
    _source_context,
)
from scripts.run_soft_label_economic_proxy_ablation import (  # noqa: E402
    DEFAULT_EVENT_FEATURE_STORE_FEATURES,
    DEFAULT_PRIOR_WINDOWS_DAYS,
    DEFAULT_STATE_PATH_PRIOR_FEATURES,
)


DEFAULT_OUTPUT_DIR = Path("data_perp/reports/support_path_decisive_label_stage100_v1")
DEFAULT_MONTHS = ("2026-04", "2026-05", "2026-06")
DEFAULT_FIT_MONTHS = ("2026-04", "2026-05")
DEFAULT_HOLDOUT_MONTH = "2026-06"
DEFAULT_TOP_FRACS = (0.005, 0.01, 0.02, 0.03, 0.05)


@dataclass(frozen=True)
class SupportArm:
    name: str
    family: str
    support_kind: str
    window_days: float
    min_count: float
    support_threshold: float
    support_power: float
    support_mode: str
    u_floor: float
    bars_cap: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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


def _mfe_mae(metrics: pd.DataFrame) -> pd.Series:
    ratio = metrics["mfe_norm"] / metrics["mae_norm"].clip(lower=0.25)
    return ratio.replace([np.inf, -np.inf], np.nan).clip(upper=10.0)


def _dirty_execution_target(metrics: pd.DataFrame) -> pd.Series:
    u = _safe_numeric(metrics["u_policy_net"]).fillna(-0.02)
    mae = _safe_numeric(metrics["mae_norm"]).fillna(10.0)
    barrier = _safe_numeric(metrics["barrier"]).fillna(1.0)
    timeout = _safe_numeric(metrics["is_timeout"].astype(float)).fillna(1.0)
    mfe_mae = _mfe_mae(metrics).fillna(0.0)
    dirty = (
        (u <= 0.0)
        | (mae >= 1.0)
        | (barrier > 0.025)
        | (timeout > 0.0)
        | (mfe_mae < 1.25)
    )
    return dirty.fillna(True).astype(float).astype(np.float32)


def _rank_blend(raw_soft: pd.Series, ts: pd.Series) -> pd.Series:
    raw_soft = _safe_numeric(raw_soft).fillna(0.0).clip(0.0, 1.0).reset_index(drop=True)
    ts_reset = pd.to_datetime(ts, utc=True, errors="coerce").reset_index(drop=True)
    ts_rank = raw_soft.groupby(ts_reset, dropna=False).rank(method="average", pct=True)
    global_rank = raw_soft.rank(method="average", pct=True)
    return (0.70 * raw_soft + 0.30 * ts_rank.fillna(global_rank)).clip(0.0, 1.0)


def _prior_metric_groups(frame: pd.DataFrame, *, window_days: float) -> dict[str, dict[str, pd.Series]]:
    suffix = f"_{window_days:g}d"
    known = ("mean_u", "hit_u", "bad_mae", "wide_25", "timeout", "mae_norm", "mfe_mae", "clean", "bounded")
    groups: dict[str, dict[str, pd.Series]] = {}
    for col in frame.columns:
        if not str(col).startswith("prior_xs_state_") or not str(col).endswith(suffix):
            continue
        if str(col).endswith(f"_count{suffix}"):
            prefix = str(col)[: -len(f"_count{suffix}")]
            groups.setdefault(prefix, {})["count"] = _safe_numeric(frame[col])
            continue
        for metric in known:
            tail = f"_{metric}{suffix}"
            if str(col).endswith(tail):
                prefix = str(col)[: -len(tail)]
                groups.setdefault(prefix, {})[metric] = _safe_numeric(frame[col])
                break
    return groups


def _support_quality(
    frame: pd.DataFrame,
    *,
    kind: str,
    window_days: float,
    min_count: float,
) -> pd.Series:
    groups = _prior_metric_groups(frame, window_days=window_days)
    qualities: list[pd.Series] = []
    for metrics in groups.values():
        count = metrics.get("count")
        if count is None:
            continue
        count_score = _sigmoid((_safe_numeric(count).fillna(0.0) - float(min_count)) / max(float(min_count) * 0.50, 1.0))
        clean = _safe_numeric(metrics.get("clean", pd.Series(np.nan, index=frame.index))).fillna(0.0).clip(0.0, 1.0)
        bounded = _safe_numeric(metrics.get("bounded", pd.Series(np.nan, index=frame.index))).fillna(0.0).clip(0.0, 1.0)
        bad = _safe_numeric(metrics.get("bad_mae", pd.Series(np.nan, index=frame.index))).fillna(1.0).clip(0.0, 1.0)
        wide = _safe_numeric(metrics.get("wide_25", pd.Series(np.nan, index=frame.index))).fillna(1.0).clip(0.0, 1.0)
        timeout = _safe_numeric(metrics.get("timeout", pd.Series(np.nan, index=frame.index))).fillna(1.0).clip(0.0, 1.0)
        mfe_mae = (_safe_numeric(metrics.get("mfe_mae", pd.Series(np.nan, index=frame.index))).fillna(0.0) / 3.0).clip(0.0, 1.0)
        mean_u = _sigmoid(_safe_numeric(metrics.get("mean_u", pd.Series(np.nan, index=frame.index))).fillna(-0.02) / 0.004)
        hit_u = _safe_numeric(metrics.get("hit_u", pd.Series(np.nan, index=frame.index))).fillna(0.0).clip(0.0, 1.0)
        if str(kind) == "clean":
            quality = count_score * (
                0.30 * clean
                + 0.20 * bounded
                + 0.20 * (1.0 - bad)
                + 0.15 * (1.0 - timeout)
                + 0.10 * (1.0 - wide)
                + 0.05 * mean_u
            )
        elif str(kind) == "bounded":
            quality = count_score * (
                0.25 * bounded
                + 0.25 * mfe_mae
                + 0.20 * mean_u
                + 0.10 * hit_u
                + 0.10 * (1.0 - timeout)
                + 0.10 * (1.0 - wide)
            )
        else:
            raise ValueError(f"Unknown support kind: {kind}")
        qualities.append(quality.astype(np.float32))
    if not qualities:
        return pd.Series(0.0, index=frame.index, dtype=np.float32)
    return pd.concat(qualities, axis=1).max(axis=1).fillna(0.0).clip(0.0, 1.0).astype(np.float32)


def _time_edge_quality(frame: pd.DataFrame, *, window_days: float, min_count: float) -> pd.Series:
    suffix = f"{window_days:g}d"
    pieces: list[pd.Series] = []
    for scope in ("symbol", "global"):
        count_col = f"prior_{scope}_time_edge_count_{suffix}"
        if count_col not in frame.columns:
            continue
        count_score = _sigmoid((_safe_numeric(frame[count_col]).fillna(0.0) - float(min_count)) / max(float(min_count) * 0.50, 1.0))
        early_clean = _safe_numeric(frame.get(f"prior_{scope}_time_edge_early_clean_{suffix}", 0.0)).fillna(0.0)
        early_edge = _safe_numeric(frame.get(f"prior_{scope}_time_edge_early_edge_{suffix}", 0.0)).fillna(0.0)
        timeout = _safe_numeric(frame.get(f"prior_{scope}_time_edge_timeout_{suffix}", 1.0)).fillna(1.0)
        late = _safe_numeric(frame.get(f"prior_{scope}_time_edge_late_or_timeout_{suffix}", 1.0)).fillna(1.0)
        pieces.append((count_score * (0.35 * early_clean + 0.30 * early_edge + 0.20 * (1.0 - timeout) + 0.15 * (1.0 - late))).clip(0.0, 1.0))
    if not pieces:
        return pd.Series(0.0, index=frame.index, dtype=np.float32)
    return pd.concat(pieces, axis=1).max(axis=1).fillna(0.0).clip(0.0, 1.0).astype(np.float32)


def _target_for_arm(frame: pd.DataFrame, metrics: pd.DataFrame, arm: SupportArm) -> pd.DataFrame:
    u = _safe_numeric(metrics["u_policy_net"]).fillna(-0.02).reset_index(drop=True)
    mae = _safe_numeric(metrics["mae_norm"]).fillna(10.0).reset_index(drop=True)
    mfe = _safe_numeric(metrics["mfe_norm"]).fillna(0.0).reset_index(drop=True)
    barrier = _safe_numeric(metrics["barrier"]).fillna(1.0).reset_index(drop=True)
    timeout = _safe_numeric(metrics["is_timeout"].astype(float)).fillna(1.0).clip(0.0, 1.0).reset_index(drop=True)
    bars = _safe_numeric(metrics["bars_to_mfe"]).fillna(24.0).reset_index(drop=True)
    mfe_mae = _mfe_mae(metrics).fillna(0.0).reset_index(drop=True)

    support = _support_quality(frame, kind=arm.support_kind, window_days=arm.window_days, min_count=arm.min_count)
    if arm.family == "decisive":
        support = pd.concat(
            [
                support,
                _time_edge_quality(frame, window_days=arm.window_days, min_count=max(arm.min_count * 0.50, 10.0)),
            ],
            axis=1,
        ).max(axis=1)
    support = support.reset_index(drop=True).clip(0.0, 1.0)
    if str(arm.support_mode) == "soft":
        support_soft = support.pow(float(arm.support_power)).clip(0.0, 1.0)
    elif str(arm.support_mode) == "gate":
        support_soft = pd.Series(
            np.where(support >= float(arm.support_threshold), 1.0, 0.05),
            index=support.index,
            dtype=np.float32,
        )
    else:
        raise ValueError(f"Unknown support mode: {arm.support_mode}")
    utility_soft = _sigmoid((u - float(arm.u_floor)) / 0.003)

    if arm.family == "clean":
        path_soft = (
            _sigmoid((1.0 - mae) / 0.30)
            * _sigmoid((0.025 - barrier) / 0.006)
            * _sigmoid((mfe_mae - 1.25) / 0.35)
            * _sigmoid((float(arm.bars_cap) - bars) / 5.0)
            * (1.0 - timeout)
        ).clip(0.0, 1.0)
        hard = (
            (u > float(arm.u_floor))
            & (mae <= 1.0)
            & (barrier <= 0.025)
            & (mfe_mae >= 1.25)
            & (bars <= float(arm.bars_cap))
            & (timeout <= 0.0)
            & (support >= float(arm.support_threshold))
        )
    elif arm.family == "bounded":
        path_soft = (
            _sigmoid((4.0 - mae) / 0.90)
            * _sigmoid((0.035 - barrier) / 0.008)
            * _sigmoid((mfe_mae - 1.25) / 0.35)
            * _sigmoid((mfe - 1.50) / 0.50)
            * _sigmoid((float(arm.bars_cap) - bars) / 8.0)
            * (1.0 - timeout)
        ).clip(0.0, 1.0)
        hard = (
            (u > float(arm.u_floor))
            & (mae <= 4.0)
            & (barrier <= 0.035)
            & (mfe_mae >= 1.25)
            & (mfe >= 1.50)
            & (bars <= float(arm.bars_cap))
            & (timeout <= 0.0)
            & (support >= float(arm.support_threshold))
        )
    elif arm.family == "decisive":
        path_soft = (
            _sigmoid((1.5 - mae) / 0.40)
            * _sigmoid((0.025 - barrier) / 0.006)
            * _sigmoid((mfe - 1.0) / 0.35)
            * _sigmoid((mfe_mae - 1.0) / 0.30)
            * _sigmoid((float(arm.bars_cap) - bars) / 3.0)
            * (1.0 - timeout)
        ).clip(0.0, 1.0)
        hard = (
            (u > float(arm.u_floor))
            & (mae <= 1.5)
            & (barrier <= 0.025)
            & (mfe >= 1.0)
            & (mfe_mae >= 1.0)
            & (bars <= float(arm.bars_cap))
            & (timeout <= 0.0)
            & (support >= float(arm.support_threshold))
        )
    else:
        raise ValueError(f"Unknown family: {arm.family}")

    raw = (utility_soft * path_soft * support_soft).clip(0.0, 1.0)
    target_soft = _rank_blend(raw, frame["__ts__"])
    return pd.DataFrame(
        {
            "target_soft": target_soft.astype(np.float32),
            "target_hard": hard.fillna(False).astype(float).astype(np.float32),
            "support_quality": support.astype(np.float32),
        },
        index=frame.index,
    )


def _default_arms() -> list[SupportArm]:
    return [
        SupportArm("P1_clean_support_14d", "clean", "clean", 14.0, 80.0, 0.22, 0.75, "soft", 0.0, 12.0),
        SupportArm("P2_clean_support_30d", "clean", "clean", 30.0, 160.0, 0.22, 0.75, "soft", 0.0, 12.0),
        SupportArm("P3_bounded_rebound_support_14d", "bounded", "bounded", 14.0, 80.0, 0.25, 0.75, "soft", 0.0, 24.0),
        SupportArm("P4_bounded_rebound_support_30d", "bounded", "bounded", 30.0, 160.0, 0.25, 0.75, "soft", 0.0, 24.0),
        SupportArm("P5_decisive_time_edge_support_14d", "decisive", "clean", 14.0, 60.0, 0.20, 0.70, "soft", 0.0, 6.0),
        SupportArm("P6_decisive_time_edge_support_30d", "decisive", "clean", 30.0, 120.0, 0.20, 0.70, "soft", 0.0, 6.0),
        SupportArm("P7_clean_support_gate_14d", "clean", "clean", 14.0, 80.0, 0.22, 1.0, "gate", 0.0, 12.0),
        SupportArm("P8_clean_support_gate_30d", "clean", "clean", 30.0, 160.0, 0.22, 1.0, "gate", 0.0, 12.0),
        SupportArm("P9_bounded_rebound_support_gate_14d", "bounded", "bounded", 14.0, 80.0, 0.25, 1.0, "gate", 0.0, 24.0),
        SupportArm("P10_bounded_rebound_support_gate_30d", "bounded", "bounded", 30.0, 160.0, 0.25, 1.0, "gate", 0.0, 24.0),
        SupportArm("P11_decisive_time_edge_support_gate_14d", "decisive", "clean", 14.0, 60.0, 0.20, 1.0, "gate", 0.0, 6.0),
        SupportArm("P12_decisive_time_edge_support_gate_30d", "decisive", "clean", 30.0, 120.0, 0.20, 1.0, "gate", 0.0, 6.0),
    ]


def _prevalence_rows(
    *,
    frame: pd.DataFrame,
    metrics: pd.DataFrame,
    target: pd.DataFrame,
    arm: SupportArm,
    months: list[str],
) -> list[dict[str, Any]]:
    period = frame["__ts__"].dt.to_period("M").astype(str)
    mfe_mae = _mfe_mae(metrics)
    rows: list[dict[str, Any]] = []
    for month in months:
        mask = period.eq(str(month))
        local_metrics = metrics.loc[mask]
        local_target = target.loc[mask]
        rows.append(
            {
                "label_arm": arm.name,
                "family": arm.family,
                "month": str(month),
                "rows": int(mask.sum()),
                "target_soft_mean": _safe_mean(local_target["target_soft"]),
                "target_soft_std": float(local_target["target_soft"].std()) if len(local_target) else float("nan"),
                "target_hard_rate": _safe_mean(local_target["target_hard"]),
                "support_mean": _safe_mean(local_target["support_quality"]),
                "support_p90": _safe_quantile(local_target["support_quality"], 0.90),
                "mean_return_net": _safe_mean(local_metrics["ret_net"]),
                "mean_u": _safe_mean(local_metrics["u_policy_net"]),
                "bad_mae_1r_rate": _safe_mean(local_metrics["mae_norm"] >= 1.0),
                "p90_mae_norm": _safe_quantile(local_metrics["mae_norm"], 0.90),
                "timeout_rate": _safe_mean(local_metrics["is_timeout"].astype(float)),
                "future_clean_rate": _safe_mean(
                    (local_metrics["u_policy_net"] > 0.0)
                    & (local_metrics["mae_norm"] <= 1.0)
                    & (local_metrics["barrier"] <= 0.025)
                    & (local_metrics["is_timeout"].astype(float) <= 0.0)
                ),
                "future_bounded_rate": _safe_mean(
                    (local_metrics["u_policy_net"] > 0.0)
                    & (local_metrics["mae_norm"] <= 4.0)
                    & (local_metrics["barrier"] <= 0.035)
                    & (mfe_mae >= 1.25)
                    & (local_metrics["is_timeout"].astype(float) <= 0.0)
                ),
            }
        )
    return rows


def _run_arm(
    *,
    frame: pd.DataFrame,
    metrics: pd.DataFrame,
    target: pd.DataFrame,
    features: list[str],
    arm: SupportArm,
    months: list[str],
    top_fracs: list[float],
    proxy_top_k: int,
    min_train_rows: int,
    min_valid_rows: int,
    include_inverse_proxy_selector: bool,
    proxy_target_mode: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    period = frame["__ts__"].dt.to_period("M").astype(str)
    period_rows: list[dict[str, Any]] = []
    proxy_ic_rows: list[dict[str, Any]] = []
    for month in months:
        train_mask = period.lt(str(month))
        valid_mask = period.eq(str(month))
        if int(train_mask.sum()) < int(min_train_rows) or int(valid_mask.sum()) < int(min_valid_rows):
            continue
        train = frame.loc[train_mask].copy()
        valid_raw = frame.loc[valid_mask].copy()
        valid = valid_raw.reset_index(drop=True)
        valid_metrics = metrics.loc[valid_mask].copy().reset_index(drop=True)
        valid_target = target.loc[valid_mask].copy().reset_index(drop=True)
        mode = str(proxy_target_mode)
        if mode == "soft":
            proxy_train_target = target.loc[train_mask, "target_soft"]
            selector_name = "fit_ic_proxy_oos"
            inverse_selector_name = "fit_ic_proxy_inverse_oos"
        elif mode == "hard":
            proxy_train_target = target.loc[train_mask, "target_hard"]
            selector_name = "fit_ic_proxy_hard_oos"
            inverse_selector_name = "fit_ic_proxy_hard_inverse_oos"
        elif mode == "hard_soft":
            proxy_train_target = (
                _safe_numeric(target.loc[train_mask, "target_soft"])
                * _safe_numeric(target.loc[train_mask, "target_hard"])
            )
            selector_name = "fit_ic_proxy_hard_soft_oos"
            inverse_selector_name = "fit_ic_proxy_hard_soft_inverse_oos"
        elif mode == "contrast_dirty":
            hard = _safe_numeric(target.loc[train_mask, "target_hard"]).fillna(0.0).gt(0.5)
            dirty = _dirty_execution_target(metrics.loc[train_mask]).reindex(hard.index).fillna(1.0).gt(0.5)
            clean = hard & ~dirty
            proxy_train_target = pd.Series(0.0, index=hard.index, dtype=np.float32)
            proxy_train_target.loc[dirty] = -1.0
            proxy_train_target.loc[clean] = 1.0
            selector_name = "fit_ic_proxy_contrast_dirty_oos"
            inverse_selector_name = "fit_ic_proxy_contrast_dirty_inverse_oos"
        else:
            raise ValueError(f"Unknown proxy target mode: {proxy_target_mode}")

        proxy_score, diag = _score_proxy(
            train=train,
            valid=valid_raw,
            features=features,
            y_train=proxy_train_target,
            proxy_top_k=int(proxy_top_k),
        )
        proxy_score = proxy_score.reset_index(drop=True)
        proxy_features = ",".join(diag.get("proxy_features", []))
        proxy_ic_rows.append(
            {
                "label_arm": arm.name,
                "family": arm.family,
                "month": str(month),
                "train_rows": int(train_mask.sum()),
                "valid_rows": int(valid_mask.sum()),
                "proxy_ic_target": _spearman(proxy_score, valid_target["target_soft"]),
                "proxy_ic_u": _spearman(proxy_score, valid_metrics["u_policy_net"]),
                "proxy_ic_bad_mae": _spearman(proxy_score, (valid_metrics["mae_norm"] >= 1.0).astype(float)),
                "proxy_ic_timeout": _spearman(proxy_score, valid_metrics["is_timeout"].astype(float)),
                "proxy_ic_support": _spearman(proxy_score, valid_target["support_quality"]),
                "proxy_features": proxy_features,
                "proxy_top_abs_ic": diag.get("proxy_top_abs_ic"),
                "proxy_mean_top_abs_ic": diag.get("proxy_mean_top_abs_ic"),
            }
        )
        selectors = [
            ("oracle_target_sort", valid_target["target_soft"], ""),
            (selector_name, proxy_score, proxy_features),
        ]
        if include_inverse_proxy_selector:
            selectors.append((inverse_selector_name, -proxy_score, proxy_features))
        period_slices = [("month", str(month), np.arange(len(valid), dtype=np.int64))]
        period_slices.extend(("week", week, pos) for week, pos in _slice_week_positions(valid))
        for selector, score, features_used in selectors:
            score = _safe_numeric(score).reset_index(drop=True)
            for period_type, period_name, pos in period_slices:
                for frac in top_fracs:
                    row = _score_period(
                        frame=valid.iloc[pos].reset_index(drop=True),
                        metrics=valid_metrics.iloc[pos].reset_index(drop=True),
                        target=valid_target.iloc[pos].reset_index(drop=True),
                        score=score.iloc[pos].reset_index(drop=True),
                        period_type=period_type,
                        period=period_name,
                        month=str(month),
                        selector=selector,
                        label_arm=arm.name,
                        economic_arm=arm.family,
                        top_frac=float(frac),
                        label_score=valid_target["target_soft"].iloc[pos].reset_index(drop=True),
                        economic_score=None,
                        economic_target=None,
                        label_proxy_features=features_used,
                        economic_proxy_features="",
                    )
                    row["family"] = arm.family
                    period_rows.append(row)
    return period_rows, proxy_ic_rows


def _weighted_mean(frame: pd.DataFrame, col: str, weight_col: str = "selected_rows") -> float:
    if frame.empty or col not in frame.columns:
        return float("nan")
    values = _safe_numeric(frame[col])
    weights = _safe_numeric(frame.get(weight_col, pd.Series(1.0, index=frame.index))).fillna(0.0)
    mask = values.notna() & weights.gt(0.0)
    if not bool(mask.any()):
        return float("nan")
    return float(np.average(values[mask], weights=weights[mask]))


def _passes_timeout(row: dict[str, Any], key: str, max_timeout_rate: float | None) -> bool:
    if max_timeout_rate is None:
        return True
    value = float(row.get(key, float("nan")))
    return math.isfinite(value) and value <= float(max_timeout_rate)


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
    monthly = period_rows[period_rows["period_type"].astype(str).eq("month")].copy()
    weekly = period_rows[period_rows["period_type"].astype(str).eq("week")].copy()
    rows: list[dict[str, Any]] = []
    group_cols = ["selector", "label_arm", "economic_arm", "top_frac"]
    for key, group in monthly.groupby(group_cols, observed=True, dropna=False):
        selector, label_arm, economic_arm, top_frac = key
        week_group = weekly[
            weekly["selector"].astype(str).eq(str(selector))
            & weekly["label_arm"].astype(str).eq(str(label_arm))
            & weekly["economic_arm"].astype(str).eq(str(economic_arm))
            & _safe_numeric(weekly["top_frac"]).eq(float(top_frac))
        ].copy()
        fit_month = group[group["month"].astype(str).isin(fit_months)].copy()
        holdout_monthly = group[group["month"].astype(str).eq(str(holdout_month))].copy()
        fit_week = week_group[week_group["month"].astype(str).isin(fit_months)].copy()
        holdout_week = week_group[week_group["month"].astype(str).eq(str(holdout_month))].copy()
        if fit_month.empty or holdout_monthly.empty:
            continue

        def week_stats(frame: pd.DataFrame) -> tuple[int, float, float]:
            returns = _safe_numeric(frame["mean_return_net"])
            selected = _safe_numeric(frame["selected_rows"]).fillna(0.0)
            material = selected.ge(int(min_week_rows))
            if not bool(material.any()):
                return 0, float("nan"), float("nan")
            return (
                int(material.sum()),
                float((returns.gt(0.0) & material).sum() / material.sum()),
                _safe_quantile(returns[material], 0.25),
            )

        fit_returns = _safe_numeric(fit_month["mean_return_net"])
        holdout_returns = _safe_numeric(holdout_monthly["mean_return_net"])
        fit_material_weeks, fit_week_rate, fit_q25_week = week_stats(fit_week)
        holdout_material_weeks, holdout_week_rate, holdout_q25_week = week_stats(holdout_week)
        row: dict[str, Any] = {
            "selector": str(selector),
            "label_arm": str(label_arm),
            "family": str(economic_arm),
            "economic_arm": str(economic_arm),
            "top_frac": float(top_frac),
            "fit_mean_return_net": _safe_mean(fit_returns),
            "fit_worst_return_net": _safe_quantile(fit_returns, 0.0),
            "holdout_mean_return_net": _safe_mean(holdout_returns),
            "fit_positive_months": int(fit_returns.gt(0.0).sum()),
            "holdout_positive_months": int(holdout_returns.gt(0.0).sum()),
            "fit_material_weeks": fit_material_weeks,
            "holdout_material_weeks": holdout_material_weeks,
            "fit_material_positive_week_rate": fit_week_rate,
            "holdout_material_positive_week_rate": holdout_week_rate,
            "fit_q25_week_return_net": fit_q25_week,
            "holdout_q25_week_return_net": holdout_q25_week,
            "fit_bad_mae_1r_rate": _weighted_mean(fit_month, "bad_mae_1r_rate"),
            "holdout_bad_mae_1r_rate": _weighted_mean(holdout_monthly, "bad_mae_1r_rate"),
            "fit_p90_mae_norm": _weighted_mean(fit_month, "p90_mae_norm"),
            "holdout_p90_mae_norm": _weighted_mean(holdout_monthly, "p90_mae_norm"),
            "fit_wide_barrier_25bps_rate": _weighted_mean(fit_month, "wide_barrier_25bps_rate"),
            "holdout_wide_barrier_25bps_rate": _weighted_mean(holdout_monthly, "wide_barrier_25bps_rate"),
            "fit_timeout_rate": _weighted_mean(fit_month, "timeout_rate"),
            "holdout_timeout_rate": _weighted_mean(holdout_monthly, "timeout_rate"),
            "fit_mean_mfe_mae_ratio": _weighted_mean(fit_month, "mean_mfe_mae_ratio"),
            "holdout_mean_mfe_mae_ratio": _weighted_mean(holdout_monthly, "mean_mfe_mae_ratio"),
            "fit_bounded_row_rate": _weighted_mean(fit_month, "bounded_row_rate"),
            "holdout_bounded_row_rate": _weighted_mean(holdout_monthly, "bounded_row_rate"),
            "fit_score_ic_u": _safe_mean(fit_month["score_ic_u"]),
            "holdout_score_ic_u": _safe_mean(holdout_monthly["score_ic_u"]),
            "fit_selected_rows": int(_safe_numeric(fit_month["selected_rows"]).sum(skipna=True)),
            "holdout_selected_rows": int(_safe_numeric(holdout_monthly["selected_rows"]).sum(skipna=True)),
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
        strict_fit = (
            fit_sign
            and row["fit_score_ic_u"] > 0.0
            and row["fit_bad_mae_1r_rate"] <= 0.40
            and row["fit_p90_mae_norm"] <= 4.00
            and row["fit_wide_barrier_25bps_rate"] <= 0.05
            and _passes_timeout(row, "fit_timeout_rate", max_timeout_rate)
        )
        strict_holdout = (
            holdout_sign
            and row["holdout_score_ic_u"] > 0.0
            and row["holdout_bad_mae_1r_rate"] <= 0.40
            and row["holdout_p90_mae_norm"] <= 4.00
            and row["holdout_wide_barrier_25bps_rate"] <= 0.05
            and _passes_timeout(row, "holdout_timeout_rate", max_timeout_rate)
        )
        if str(economic_arm) == "bounded":
            family_fit = (
                fit_sign
                and row["fit_score_ic_u"] > 0.0
                and row["fit_bad_mae_1r_rate"] <= 0.80
                and row["fit_p90_mae_norm"] <= 4.00
                and row["fit_mean_mfe_mae_ratio"] >= 1.25
                and row["fit_wide_barrier_25bps_rate"] <= 0.35
                and row["fit_timeout_rate"] <= 0.20
                and row["fit_bounded_row_rate"] >= 0.10
            )
            family_holdout = (
                holdout_sign
                and row["holdout_score_ic_u"] > 0.0
                and row["holdout_bad_mae_1r_rate"] <= 0.80
                and row["holdout_p90_mae_norm"] <= 4.00
                and row["holdout_mean_mfe_mae_ratio"] >= 1.25
                and row["holdout_wide_barrier_25bps_rate"] <= 0.35
                and row["holdout_timeout_rate"] <= 0.20
                and row["holdout_bounded_row_rate"] >= 0.10
            )
        else:
            family_fit = strict_fit
            family_holdout = strict_holdout
        row["fit_sign_pass"] = bool(fit_sign)
        row["holdout_sign_pass"] = bool(holdout_sign)
        row["fit_economic_pass"] = bool(strict_fit)
        row["holdout_economic_pass"] = bool(strict_holdout)
        row["trainworthy_pass"] = bool(strict_fit and strict_holdout)
        row["fit_family_economic_pass"] = bool(family_fit)
        row["holdout_family_economic_pass"] = bool(family_holdout)
        row["family_trainworthy_pass"] = bool(family_fit and family_holdout)
        rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(
        ["family_trainworthy_pass", "trainworthy_pass", "holdout_family_economic_pass", "holdout_economic_pass", "fit_family_economic_pass", "holdout_mean_return_net"],
        ascending=[False, False, False, False, False, False],
    )


def _candidate_summary(
    *,
    arms: list[SupportArm],
    prevalence: pd.DataFrame,
    fit_holdout: pd.DataFrame,
    proxy_ic: pd.DataFrame,
    holdout_month: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for arm in arms:
        fit = fit_holdout[fit_holdout["label_arm"].astype(str).eq(arm.name)].copy()
        non_oracle = fit[~fit["selector"].astype(str).str.startswith("oracle_")].copy()
        oracle = fit[fit["selector"].astype(str).str.startswith("oracle_")].copy()
        prev = prevalence[prevalence["label_arm"].astype(str).eq(arm.name)].copy()
        june_prev = prev[prev["month"].astype(str).eq(str(holdout_month))]
        june_ic = proxy_ic[
            proxy_ic["label_arm"].astype(str).eq(arm.name)
            & proxy_ic["month"].astype(str).eq(str(holdout_month))
        ]
        best = (
            non_oracle.sort_values(
                ["family_trainworthy_pass", "holdout_family_economic_pass", "fit_family_economic_pass", "holdout_mean_return_net"],
                ascending=[False, False, False, False],
            ).head(1)
            if not non_oracle.empty
            else pd.DataFrame()
        )
        row: dict[str, Any] = {
            **arm.to_dict(),
            "june_target_hard_rate": float(june_prev["target_hard_rate"].iloc[0]) if len(june_prev) else float("nan"),
            "june_support_mean": float(june_prev["support_mean"].iloc[0]) if len(june_prev) else float("nan"),
            "june_support_p90": float(june_prev["support_p90"].iloc[0]) if len(june_prev) else float("nan"),
            "june_proxy_ic_target": _safe_mean(june_ic["proxy_ic_target"]) if len(june_ic) else float("nan"),
            "june_proxy_ic_u": _safe_mean(june_ic["proxy_ic_u"]) if len(june_ic) else float("nan"),
            "non_oracle_trainworthy": int(non_oracle["trainworthy_pass"].sum()) if not non_oracle.empty else 0,
            "non_oracle_family_trainworthy": int(non_oracle["family_trainworthy_pass"].sum()) if not non_oracle.empty else 0,
            "non_oracle_fit_family_economic": int(non_oracle["fit_family_economic_pass"].sum()) if not non_oracle.empty else 0,
            "non_oracle_holdout_family_economic": int(non_oracle["holdout_family_economic_pass"].sum()) if not non_oracle.empty else 0,
            "oracle_family_trainworthy": int(oracle["family_trainworthy_pass"].sum()) if not oracle.empty else 0,
        }
        if not best.empty:
            rec = best.iloc[0]
            for col in [
                "selector",
                "top_frac",
                "fit_mean_return_net",
                "holdout_mean_return_net",
                "fit_bad_mae_1r_rate",
                "holdout_bad_mae_1r_rate",
                "fit_p90_mae_norm",
                "holdout_p90_mae_norm",
                "fit_timeout_rate",
                "holdout_timeout_rate",
                "fit_mean_mfe_mae_ratio",
                "holdout_mean_mfe_mae_ratio",
                "fit_bounded_row_rate",
                "holdout_bounded_row_rate",
                "fit_score_ic_u",
                "holdout_score_ic_u",
                "fit_selected_rows",
                "holdout_selected_rows",
            ]:
                row[f"best_non_oracle_{col}"] = rec.get(col)
        rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(
        [
            "non_oracle_family_trainworthy",
            "non_oracle_holdout_family_economic",
            "non_oracle_fit_family_economic",
            "best_non_oracle_holdout_mean_return_net",
            "june_proxy_ic_u",
        ],
        ascending=[False, False, False, False, False],
    )


def _write_markdown(
    *,
    output_dir: Path,
    candidate_summary: pd.DataFrame,
    fit_holdout: pd.DataFrame,
    prevalence: pd.DataFrame,
    proxy_ic: pd.DataFrame,
    manifest: dict[str, Any],
) -> Path:
    path = output_dir / "support_path_decisive_label_ablation.md"
    non_oracle = fit_holdout[~fit_holdout["selector"].astype(str).str.startswith("oracle_")].copy()
    lines = [
        "# Support Path-Decisive Label Ablation",
        "",
        "Scope: proxy-only support-aware target ablation. No base/meta training, Optuna, or policy optimisation.",
        "",
        f"Fit months: `{', '.join(manifest['fit_months'])}`. Holdout: `{manifest['holdout_month']}`.",
        f"Arms: `{manifest['arm_count']}`. Features: `{manifest['feature_count']}`. Proxy top-k: `{manifest['proxy_top_k']}`.",
        "",
        "## Candidate Summary",
        "",
        _table(
            candidate_summary,
            [
                "name",
                "family",
                "non_oracle_family_trainworthy",
                "non_oracle_fit_family_economic",
                "non_oracle_holdout_family_economic",
                "oracle_family_trainworthy",
                "june_target_hard_rate",
                "june_support_mean",
                "june_proxy_ic_u",
                "best_non_oracle_selector",
                "best_non_oracle_top_frac",
                "best_non_oracle_fit_mean_return_net",
                "best_non_oracle_holdout_mean_return_net",
                "best_non_oracle_fit_bad_mae_1r_rate",
                "best_non_oracle_holdout_bad_mae_1r_rate",
                "best_non_oracle_fit_p90_mae_norm",
                "best_non_oracle_holdout_p90_mae_norm",
                "best_non_oracle_fit_timeout_rate",
                "best_non_oracle_holdout_timeout_rate",
            ],
            limit=80,
        ),
        "",
        "## Best Non-Oracle Rows",
        "",
        _table(
            non_oracle.sort_values(
                ["family_trainworthy_pass", "holdout_family_economic_pass", "fit_family_economic_pass", "holdout_mean_return_net"],
                ascending=[False, False, False, False],
            ),
            [
                "family_trainworthy_pass",
                "fit_family_economic_pass",
                "holdout_family_economic_pass",
                "trainworthy_pass",
                "fit_economic_pass",
                "holdout_economic_pass",
                "fit_sign_pass",
                "holdout_sign_pass",
                "selector",
                "label_arm",
                "family",
                "top_frac",
                "fit_mean_return_net",
                "holdout_mean_return_net",
                "fit_bad_mae_1r_rate",
                "holdout_bad_mae_1r_rate",
                "fit_p90_mae_norm",
                "holdout_p90_mae_norm",
                "fit_timeout_rate",
                "holdout_timeout_rate",
                "fit_mean_mfe_mae_ratio",
                "holdout_mean_mfe_mae_ratio",
                "fit_bounded_row_rate",
                "holdout_bounded_row_rate",
                "fit_score_ic_u",
                "holdout_score_ic_u",
                "fit_selected_rows",
                "holdout_selected_rows",
            ],
            limit=120,
        ),
        "",
        "## June Proxy IC",
        "",
        _table(
            proxy_ic[proxy_ic["month"].astype(str).eq(str(manifest["holdout_month"]))].sort_values(
                "proxy_ic_u",
                ascending=False,
            ),
            [
                "label_arm",
                "family",
                "valid_rows",
                "proxy_ic_target",
                "proxy_ic_u",
                "proxy_ic_bad_mae",
                "proxy_ic_timeout",
                "proxy_ic_support",
                "proxy_features",
            ],
            limit=80,
        ),
        "",
        "## June Prevalence",
        "",
        _table(
            prevalence[prevalence["month"].astype(str).eq(str(manifest["holdout_month"]))],
            [
                "label_arm",
                "family",
                "target_hard_rate",
                "target_soft_mean",
                "support_mean",
                "support_p90",
                "future_clean_rate",
                "future_bounded_rate",
                "bad_mae_1r_rate",
                "p90_mae_norm",
                "timeout_rate",
            ],
            limit=80,
        ),
        "",
        "## Outputs",
        "",
        f"- Candidate summary: `{manifest['outputs']['candidate_summary']}`",
        f"- Fit/holdout: `{manifest['outputs']['fit_holdout']}`",
        f"- Period rows: `{manifest['outputs']['period_rows']}`",
        f"- Prevalence: `{manifest['outputs']['prevalence']}`",
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
    max_feature_columns: int | None,
    months: list[str],
    fit_months: list[str],
    holdout_month: str,
    top_fracs: list[float],
    proxy_top_k: int,
    min_train_rows: int,
    min_valid_rows: int,
    min_week_rows: int,
    max_timeout_rate: float | None,
    include_causal_time_edge_priors: bool,
    include_causal_outcome_priors: bool,
    include_causal_state_path_priors: bool,
    include_event_confirmation_features: bool,
    include_adverse_path_composites: bool,
    include_inverse_proxy_selector: bool,
    proxy_target_mode: str,
    prior_windows_days: list[float],
    prior_embargo_hours: float,
    state_path_prior_features: list[str],
    event_feature_store_features: list[str],
    arms: list[SupportArm],
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
        time_edge, reports["causal_time_edge_priors"] = _causal_time_edge_prior_features(
            frame,
            metrics,
            windows_days=prior_windows_days,
            embargo_hours=prior_embargo_hours,
        )
        frame = pd.concat([frame, time_edge.astype(np.float32, copy=False)], axis=1).copy()
    context = _source_context(frame)
    frame = pd.concat([frame, context.astype(np.float32, copy=False)], axis=1).copy()

    features = _feature_columns(frame)
    if max_feature_columns is not None and int(max_feature_columns) > 0:
        features = features[: int(max_feature_columns)]

    period_rows: list[dict[str, Any]] = []
    proxy_ic_rows: list[dict[str, Any]] = []
    prevalence_rows: list[dict[str, Any]] = []
    for idx, arm in enumerate(arms, start=1):
        target = _target_for_arm(frame, metrics, arm)
        target.index = frame.index
        prevalence_rows.extend(_prevalence_rows(frame=frame, metrics=metrics, target=target, arm=arm, months=months))
        cur_period, cur_proxy = _run_arm(
            frame=frame,
            metrics=metrics,
            target=target,
            features=features,
            arm=arm,
            months=months,
            top_fracs=top_fracs,
            proxy_top_k=proxy_top_k,
            min_train_rows=min_train_rows,
            min_valid_rows=min_valid_rows,
            include_inverse_proxy_selector=include_inverse_proxy_selector,
            proxy_target_mode=proxy_target_mode,
        )
        period_rows.extend(cur_period)
        proxy_ic_rows.extend(cur_proxy)
        print(json.dumps({"arm": arm.name, "idx": idx, "period_rows_so_far": len(period_rows)}, sort_keys=True))

    period_frame = pd.DataFrame(period_rows)
    proxy_ic = pd.DataFrame(proxy_ic_rows)
    prevalence = pd.DataFrame(prevalence_rows)
    fit_holdout = _fit_holdout_summary(
        period_frame,
        fit_months=fit_months,
        holdout_month=holdout_month,
        min_week_rows=min_week_rows,
        max_timeout_rate=max_timeout_rate,
    )
    candidate_summary = _candidate_summary(
        arms=arms,
        prevalence=prevalence,
        fit_holdout=fit_holdout,
        proxy_ic=proxy_ic,
        holdout_month=holdout_month,
    )

    paths = {
        "candidate_summary": output_dir / "support_path_decisive_candidate_summary.csv",
        "fit_holdout": output_dir / "support_path_decisive_fit_holdout.csv",
        "period_rows": output_dir / "support_path_decisive_period_rows.csv",
        "prevalence": output_dir / "support_path_decisive_prevalence.csv",
        "proxy_ic": output_dir / "support_path_decisive_proxy_ic.csv",
        "manifest": output_dir / "manifest.json",
    }
    candidate_summary.to_csv(paths["candidate_summary"], index=False)
    fit_holdout.to_csv(paths["fit_holdout"], index=False)
    period_frame.to_csv(paths["period_rows"], index=False)
    prevalence.to_csv(paths["prevalence"], index=False)
    proxy_ic.to_csv(paths["proxy_ic"], index=False)

    manifest = {
        "scope": "support_path_decisive_label_ablation",
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
        "arm_count": int(len(arms)),
        "arms": [arm.to_dict() for arm in arms],
        "months": [str(v) for v in months],
        "fit_months": [str(v) for v in fit_months],
        "holdout_month": str(holdout_month),
        "top_fracs": [float(v) for v in top_fracs],
        "proxy_top_k": int(proxy_top_k),
        "min_train_rows": int(min_train_rows),
        "min_valid_rows": int(min_valid_rows),
        "min_week_rows": int(min_week_rows),
        "max_timeout_rate": max_timeout_rate,
        "include_inverse_proxy_selector": bool(include_inverse_proxy_selector),
        "proxy_target_mode": str(proxy_target_mode),
        "prior_windows_days": [float(v) for v in prior_windows_days],
        "prior_embargo_hours": float(prior_embargo_hours),
        "reports": reports,
        "outputs": {key: str(path) for key, path in paths.items()},
    }
    markdown = _write_markdown(
        output_dir=output_dir,
        candidate_summary=candidate_summary,
        fit_holdout=fit_holdout,
        prevalence=prevalence,
        proxy_ic=proxy_ic,
        manifest={**manifest, "outputs": {**manifest["outputs"], "markdown": str(output_dir / "support_path_decisive_label_ablation.md")}},
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
    parser.add_argument("--max-feature-store-features", type=int, default=160)
    parser.add_argument("--max-feature-columns", type=int, default=0)
    parser.add_argument("--months", type=lambda value: _parse_csv(value, DEFAULT_MONTHS), default=list(DEFAULT_MONTHS))
    parser.add_argument("--fit-months", type=lambda value: _parse_csv(value, DEFAULT_FIT_MONTHS), default=list(DEFAULT_FIT_MONTHS))
    parser.add_argument("--holdout-month", type=str, default=DEFAULT_HOLDOUT_MONTH)
    parser.add_argument("--top-fracs", type=lambda value: _parse_float_csv(value), default=list(DEFAULT_TOP_FRACS))
    parser.add_argument("--proxy-top-k", type=int, default=DEFAULT_PROXY_TOP_K)
    parser.add_argument("--min-train-rows", type=int, default=250)
    parser.add_argument("--min-valid-rows", type=int, default=25)
    parser.add_argument("--min-week-rows", type=int, default=5)
    parser.add_argument("--max-timeout-rate", type=float, default=0.50)
    parser.add_argument("--include-causal-time-edge-priors", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-causal-outcome-priors", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-causal-state-path-priors", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-event-confirmation-features", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-adverse-path-composites", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-inverse-proxy-selector", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--proxy-target-mode", choices=("soft", "hard", "hard_soft", "contrast_dirty"), default="soft")
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
        max_feature_columns=args.max_feature_columns,
        months=list(args.months),
        fit_months=list(args.fit_months),
        holdout_month=str(args.holdout_month),
        top_fracs=list(args.top_fracs),
        proxy_top_k=int(args.proxy_top_k),
        min_train_rows=int(args.min_train_rows),
        min_valid_rows=int(args.min_valid_rows),
        min_week_rows=int(args.min_week_rows),
        max_timeout_rate=args.max_timeout_rate,
        include_causal_time_edge_priors=bool(args.include_causal_time_edge_priors),
        include_causal_outcome_priors=bool(args.include_causal_outcome_priors),
        include_causal_state_path_priors=bool(args.include_causal_state_path_priors),
        include_event_confirmation_features=bool(args.include_event_confirmation_features),
        include_adverse_path_composites=bool(args.include_adverse_path_composites),
        include_inverse_proxy_selector=bool(args.include_inverse_proxy_selector),
        proxy_target_mode=str(args.proxy_target_mode),
        prior_windows_days=list(args.prior_windows_days),
        prior_embargo_hours=float(args.prior_embargo_hours),
        state_path_prior_features=list(args.state_path_prior_features),
        event_feature_store_features=list(args.event_feature_store_features),
        arms=_default_arms(),
    )
    print(json.dumps(_json_safe({"output_dir": manifest["output_dir"], "outputs": manifest["outputs"]}), indent=2))


if __name__ == "__main__":
    main()
