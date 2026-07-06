#!/usr/bin/env python3
"""Round-A soft-label top-k proxy diagnostics.

This is a pre-training diagnostic for the soft-label/sample-weight plan. It
keeps weights fixed at W0 and compares S0/S2/S3/S6/S7/S8 by timestamp-balanced
HR@10/20/30, NDCG@30, weekly lower-tail HR, and execution economics.
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
    _slice_week_positions,
    _table,
)
from scripts.run_label_quality_proxy_diagnostics import (  # noqa: E402
    DEFAULT_FEATURE_DIR,
    _feature_columns,
    _json_safe,
    _make_targets,
    _safe_mean,
    _safe_quantile,
    _spearman,
)
from scripts.run_soft_label_economic_proxy_ablation import (  # noqa: E402
    DEFAULT_EVENT_FEATURE_STORE_FEATURES,
    DEFAULT_PRIOR_WINDOWS_DAYS,
    DEFAULT_STATE_PATH_PRIOR_FEATURES,
    _proxy_score as _economic_proxy_score,
)


DEFAULT_OUTPUT_DIR = Path("data_perp/reports/soft_label_rounda_topk_proxy_stage119_v1")
DEFAULT_LABEL_ARMS = (
    "S0_current_y_bin",
    "S2_cost_aware_return",
    "S3_path_quality",
    "S6_asymmetric_downside",
    "S7_horizon_blended",
    "S8_timestamp_rank_path",
)
DEFAULT_MONTHS = ("2026-04", "2026-05", "2026-06")
DEFAULT_FIT_MONTHS = ("2026-04", "2026-05")
DEFAULT_HOLDOUT_MONTH = "2026-06"
DEFAULT_TOP_KS = (10, 20, 30)
DEFAULT_MIN_SCORES = (-1.0,)
DEFAULT_GATE_MIN_SCORES = (0.50, 0.60, 0.70, 0.80, 0.90)
STRICT_LABEL_ARMS = (
    "S120_s3_clean_utility_veto",
    "S121_s8_clean_rank_veto",
    "S122_clean_dirty_contrast_rank",
    "S123_fast_clean_path_rank",
    "S124_s3_net_floor_veto",
    "S125_s8_net_floor_rank_veto",
    "S126_clean_net_direct_rank",
    "S127_fast_clean_net_rank",
)
ROUND_TRIP_COST = 0.0030


def _parse_csv(value: str | list[str] | tuple[str, ...], default: tuple[str, ...] = ()) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(part).strip() for part in value if str(part).strip()]
    text = str(value or "").strip()
    if not text:
        return list(default)
    return [part.strip() for part in text.split(",") if part.strip()]


def _parse_float_csv(value: str | list[float] | tuple[float, ...]) -> list[float]:
    if isinstance(value, (list, tuple)):
        return [float(part) for part in value]
    return [float(part.strip()) for part in str(value).split(",") if part.strip()]


def _parse_int_csv(value: str | list[int] | tuple[int, ...], default: tuple[int, ...] = ()) -> list[int]:
    if isinstance(value, (list, tuple)):
        return [int(part) for part in value]
    text = str(value or "").strip()
    if not text:
        return list(default)
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def _safe_numeric(values: Any) -> pd.Series:
    if isinstance(values, pd.Series):
        return pd.to_numeric(values, errors="coerce")
    return pd.to_numeric(pd.Series(values), errors="coerce")


def _sigmoid_series(values: Any, index: pd.Index) -> pd.Series:
    arr = np.asarray(values, dtype=np.float64)
    return pd.Series(1.0 / (1.0 + np.exp(-np.clip(arr, -60.0, 60.0))), index=index).clip(0.0, 1.0)


def _mfe_mae(metrics: pd.DataFrame) -> pd.Series:
    ratio = metrics["mfe_norm"] / metrics["mae_norm"].clip(lower=0.25)
    return ratio.replace([np.inf, -np.inf], np.nan).clip(upper=10.0)


def _timestamp_rank(frame: pd.DataFrame, values: pd.Series) -> pd.Series:
    rank = _safe_numeric(values).groupby(frame["__ts__"], dropna=False).rank(method="average", pct=True)
    return rank.fillna(_safe_numeric(values).rank(method="average", pct=True)).clip(0.0, 1.0)


def _masked_timestamp_rank(frame: pd.DataFrame, values: pd.Series) -> pd.Series:
    raw = _safe_numeric(values).fillna(0.0).clip(0.0, 1.0)
    return (_timestamp_rank(frame, raw) * raw.gt(0.0).astype(float)).clip(0.0, 1.0)


def _fmt_score(value: float) -> str:
    return str(int(round(float(value) * 100))).zfill(2)


def _target_frame(soft: pd.Series, hard: pd.Series) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "target_soft": _safe_numeric(soft).fillna(0.0).clip(0.0, 1.0).astype(np.float32),
            "target_hard": pd.Series(hard, index=soft.index).fillna(False).astype(float),
        },
        index=soft.index,
    )


def _rounda_proxy_score(
    *,
    train: pd.DataFrame,
    valid: pd.DataFrame,
    features: list[str],
    y_train: pd.Series,
    metrics_train: pd.DataFrame,
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
) -> tuple[pd.Series, dict[str, Any]]:
    if str(proxy_objective) == "target_ic":
        score, diag = _score_proxy(
            train=train,
            valid=valid,
            features=features,
            y_train=y_train,
            proxy_top_k=proxy_top_k,
        )
        diag = {**diag, "proxy_objective": "target_ic"}
        return score, diag
    return _economic_proxy_score(
        train=train,
        valid=valid,
        features=features,
        target_train=y_train,
        metrics_train=metrics_train,
        top_k=int(proxy_top_k),
        proxy_objective=str(proxy_objective),
        min_target_ic=float(proxy_min_target_ic),
        min_utility_ic=float(proxy_min_utility_ic),
        max_bad_mae_ic=float(proxy_max_bad_mae_ic),
        max_wide_ic=float(proxy_max_wide_ic),
        max_timeout_ic=float(proxy_max_timeout_ic),
        utility_weight=float(proxy_utility_weight),
        bad_mae_weight=float(proxy_bad_mae_weight),
        wide_weight=float(proxy_wide_weight),
        timeout_weight=float(proxy_timeout_weight),
    )


def _strict_rounda_targets(
    *,
    frame: pd.DataFrame,
    metrics: pd.DataFrame,
    base_targets: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """Targets that make dirty profitable lookalikes explicit negatives."""
    idx = metrics.index
    u = _safe_numeric(metrics["u_policy_net"]).fillna(-0.02)
    mae = _safe_numeric(metrics["mae_norm"]).fillna(10.0)
    mfe = _safe_numeric(metrics["mfe_norm"]).fillna(0.0)
    barrier = _safe_numeric(metrics["barrier"]).fillna(1.0)
    bars = _safe_numeric(metrics["bars_to_mfe"]).fillna(24.0)
    timeout = _safe_numeric(metrics["is_timeout"].astype(float)).fillna(1.0).clip(0.0, 1.0)
    mfe_mae = _mfe_mae(metrics).fillna(0.0)

    utility = _sigmoid_series((u - 0.0010) / 0.0060, idx)
    margin_utility = _sigmoid_series((u - 0.0030) / 0.0060, idx)
    low_mae = _sigmoid_series((0.75 - mae) / 0.18, idx)
    bounded_mae = _sigmoid_series((1.00 - mae) / 0.22, idx)
    low_barrier = _sigmoid_series((0.024 - barrier) / 0.0045, idx)
    efficient = _sigmoid_series((mfe_mae - 1.45) / 0.30, idx)
    fast = _sigmoid_series((7.0 - bars) / 2.5, idx)
    enough_mfe = _sigmoid_series((mfe - 1.15) / 0.25, idx)
    no_timeout = (1.0 - timeout).clip(0.0, 1.0)
    clean_gate = (low_mae * low_barrier * efficient * no_timeout).clip(0.0, 1.0)
    fast_clean_gate = (low_mae * low_barrier * efficient * fast * enough_mfe * no_timeout).clip(0.0, 1.0)

    dirty_penalty = (
        0.35 * _sigmoid_series((mae - 1.00) / 0.20, idx)
        + 0.25 * timeout
        + 0.20 * _sigmoid_series((barrier - 0.025) / 0.004, idx)
        + 0.20 * _sigmoid_series((1.20 - mfe_mae) / 0.25, idx)
    ).clip(0.0, 1.0)

    s3 = base_targets["S3_path_quality"]["target_soft"]
    s8 = base_targets["S8_timestamp_rank_path"]["target_soft"]
    s3_clean = (s3 * utility * clean_gate).clip(0.0, 1.0)
    s8_clean = (s8 * utility * clean_gate).clip(0.0, 1.0)
    s8_clean_rank = _timestamp_rank(frame, s8_clean)
    contrast_raw = (margin_utility * (0.55 * s3 + 0.45 * s8) * clean_gate * (1.0 - dirty_penalty)).clip(0.0, 1.0)
    contrast_rank = _timestamp_rank(frame, contrast_raw)
    fast_clean = (margin_utility * fast_clean_gate).clip(0.0, 1.0)
    fast_clean_rank = _timestamp_rank(frame, fast_clean)

    clean_hard = (
        (u > 0.0010)
        & (mae <= 0.75)
        & (barrier <= 0.025)
        & (mfe_mae >= 1.35)
        & (timeout <= 0.0)
    )
    fast_hard = clean_hard & (bars <= 7.0) & (mfe >= 1.15)
    net_floor = ((u - 0.0010) / 0.0120).clip(0.0, 1.0)
    net_margin_floor = ((u - 0.0030) / 0.0120).clip(0.0, 1.0)
    mae_floor = ((0.95 - mae) / 0.95).clip(0.0, 1.0)
    barrier_floor = ((0.027 - barrier) / 0.027).clip(0.0, 1.0)
    efficiency_floor = ((mfe_mae - 1.05) / 2.00).clip(0.0, 1.0)
    fast_floor = ((9.0 - bars) / 9.0).clip(0.0, 1.0)
    economic_core = (
        (u > 0.0010)
        & (mae <= 0.95)
        & (barrier <= 0.027)
        & (mfe_mae >= 1.05)
        & (timeout <= 0.0)
    )
    economic_core_gate = economic_core.astype(float)
    net_clean_gate = (economic_core_gate * net_floor * mae_floor * barrier_floor * efficiency_floor).clip(0.0, 1.0)
    s3_net_floor = (s3 * net_clean_gate).clip(0.0, 1.0)
    s8_net_floor = (s8 * net_clean_gate).clip(0.0, 1.0)
    s8_net_floor_rank = _masked_timestamp_rank(frame, s8_net_floor)
    direct_clean = (net_margin_floor * net_clean_gate).clip(0.0, 1.0)
    direct_clean_rank = _masked_timestamp_rank(frame, direct_clean)
    fast_net = (direct_clean * fast_floor * enough_mfe).clip(0.0, 1.0)
    fast_net_rank = _masked_timestamp_rank(frame, fast_net)
    return {
        "S120_s3_clean_utility_veto": _target_frame(s3_clean, clean_hard & (s3_clean >= 0.35)),
        "S121_s8_clean_rank_veto": _target_frame((0.45 * s8_clean + 0.55 * s8_clean_rank).clip(0.0, 1.0), clean_hard),
        "S122_clean_dirty_contrast_rank": _target_frame((0.45 * contrast_raw + 0.55 * contrast_rank).clip(0.0, 1.0), clean_hard & (contrast_rank >= 0.85)),
        "S123_fast_clean_path_rank": _target_frame((0.40 * fast_clean + 0.60 * fast_clean_rank).clip(0.0, 1.0), fast_hard & (fast_clean_rank >= 0.85)),
        "S124_s3_net_floor_veto": _target_frame(s3_net_floor, clean_hard & (s3_net_floor > 0.0)),
        "S125_s8_net_floor_rank_veto": _target_frame((0.45 * s8_net_floor + 0.55 * s8_net_floor_rank).clip(0.0, 1.0), clean_hard & (s8_net_floor > 0.0)),
        "S126_clean_net_direct_rank": _target_frame((0.45 * direct_clean + 0.55 * direct_clean_rank).clip(0.0, 1.0), clean_hard & (direct_clean > 0.0)),
        "S127_fast_clean_net_rank": _target_frame((0.40 * fast_net + 0.60 * fast_net_rank).clip(0.0, 1.0), fast_hard & (fast_net > 0.0)),
    }


def _ndcg_at_k(relevance: pd.Series, score: pd.Series, k: int) -> float:
    rel = _safe_numeric(relevance).fillna(0.0).clip(0.0, 1.0)
    scr = _safe_numeric(score)
    valid = scr.notna() & rel.notna()
    if int(valid.sum()) <= 1:
        return float("nan")
    local = pd.DataFrame({"rel": rel[valid], "score": scr[valid]})
    k_eff = min(int(k), len(local))
    if k_eff <= 0:
        return float("nan")
    chosen = local.sort_values("score", ascending=False, kind="mergesort").head(k_eff)
    ideal = local.sort_values("rel", ascending=False, kind="mergesort").head(k_eff)
    discount = 1.0 / np.log2(np.arange(2, k_eff + 2, dtype=np.float64))
    gain = np.power(2.0, chosen["rel"].to_numpy(dtype=np.float64)) - 1.0
    ideal_gain = np.power(2.0, ideal["rel"].to_numpy(dtype=np.float64)) - 1.0
    dcg = float(np.sum(gain * discount))
    idcg = float(np.sum(ideal_gain * discount))
    return dcg / idcg if idcg > 0.0 else float("nan")


def _topk_positions_by_timestamp(
    frame: pd.DataFrame,
    score: pd.Series,
    k: int,
    *,
    min_score: float,
) -> list[tuple[str, np.ndarray]]:
    score = _safe_numeric(score).reset_index(drop=True)
    ts = pd.to_datetime(frame["__ts__"], errors="coerce").reset_index(drop=True)
    out: list[tuple[str, np.ndarray]] = []
    for timestamp, ids in pd.Series(np.arange(len(frame), dtype=np.int64)).groupby(ts.astype(str), sort=False):
        pos = ids.to_numpy(dtype=np.int64)
        local_score = score.iloc[pos]
        valid_pos = pos[local_score.notna().to_numpy()]
        if float(min_score) > -1.0 and len(valid_pos):
            valid_pos = valid_pos[score.iloc[valid_pos].to_numpy(dtype=np.float64) > float(min_score)]
        if len(valid_pos) == 0:
            continue
        local_valid_score = score.iloc[valid_pos].to_numpy(dtype=np.float64)
        order = np.argsort(-local_valid_score, kind="mergesort")
        chosen = valid_pos[order[: min(int(k), len(order))]]
        out.append((str(timestamp), chosen.astype(np.int64, copy=False)))
    return out


def _score_topk_period(
    *,
    frame: pd.DataFrame,
    metrics: pd.DataFrame,
    target: pd.DataFrame,
    score: pd.Series,
    period_type: str,
    period: str,
    month: str,
    selector: str,
    label_arm: str,
    top_k: int,
    min_score: float,
    proxy_features: str,
    train_rows: int,
    train_weight_ess: float,
) -> dict[str, Any]:
    frame = frame.reset_index(drop=True)
    metrics = metrics.reset_index(drop=True)
    target = target.reset_index(drop=True)
    score = _safe_numeric(score).reset_index(drop=True)
    selections = _topk_positions_by_timestamp(frame, score, top_k, min_score=min_score)
    selected_idx = np.concatenate([idx for _, idx in selections]) if selections else np.array([], dtype=np.int64)
    selected_metrics = metrics.iloc[selected_idx] if len(selected_idx) else metrics.iloc[:0]
    selected_target = target.iloc[selected_idx] if len(selected_idx) else target.iloc[:0]
    selected_frame = frame.iloc[selected_idx] if len(selected_idx) else frame.iloc[:0]
    ret_net = _safe_numeric(selected_metrics.get("ret_net"))
    gross_ret = _safe_numeric(selected_metrics.get("return"))
    u = _safe_numeric(selected_metrics.get("u_policy_net"))
    mae = _safe_numeric(selected_metrics.get("mae_norm"))
    barrier = _safe_numeric(selected_metrics.get("barrier"))
    timeout = _safe_numeric(selected_metrics.get("is_timeout", pd.Series(dtype=float)).astype(float))
    mfe_mae = _mfe_mae(selected_metrics) if len(selected_metrics) else pd.Series(dtype=float)
    if "y_outcome" in selected_metrics.columns:
        full_sl = _safe_numeric(selected_metrics["y_outcome"]).eq(0.0)
    else:
        full_sl = mae.ge(1.0)

    ts_hr_label: list[float] = []
    ts_hr_u: list[float] = []
    ts_ndcg: list[float] = []
    for _, idx in selections:
        ts_target = target["target_hard"].iloc[idx]
        ts_u = metrics["u_policy_net"].iloc[idx]
        ts_hr_label.append(_safe_mean(ts_target))
        ts_hr_u.append(_safe_mean(ts_u > 0.0))
    for _, ids in pd.Series(np.arange(len(frame), dtype=np.int64)).groupby(frame["__ts__"].astype(str), sort=False):
        pos = ids.to_numpy(dtype=np.int64)
        if len(pos):
            ts_ndcg.append(_ndcg_at_k(target["target_soft"].iloc[pos], score.iloc[pos], top_k))

    return {
        "period_type": period_type,
        "period": period,
        "month": month,
        "selector": selector,
        "label_arm": label_arm,
        "top_k": int(top_k),
        "min_score": float(min_score),
        "rows": int(len(frame)),
        "timestamp_count": int(len(selections)),
        "selected_rows": int(len(selected_idx)),
        "tb_hr_label": _safe_mean(ts_hr_label),
        "tb_hr_u": _safe_mean(ts_hr_u),
        "tb_ndcg_label": _safe_mean(ts_ndcg),
        "target_top_soft_mean": _safe_mean(selected_target.get("target_soft")),
        "target_top_hard_rate": _safe_mean(selected_target.get("target_hard")),
        "mean_u": _safe_mean(u),
        "hit_u": _safe_mean(u > 0.0),
        "q10_u": _safe_quantile(u, 0.10),
        "mean_return_net": _safe_mean(ret_net),
        "q05_return_net": _safe_quantile(ret_net, 0.05),
        "gross_pnl": float(gross_ret.sum(skipna=True)) if len(gross_ret) else 0.0,
        "costs": float(len(selected_idx) * ROUND_TRIP_COST),
        "net_pnl": float(ret_net.sum(skipna=True)) if len(ret_net) else 0.0,
        "full_sl_rate": _safe_mean(full_sl),
        "bad_mae_1r_rate": _safe_mean(mae >= 1.0),
        "p90_mae_norm": _safe_quantile(mae, 0.90),
        "wide_barrier_25bps_rate": _safe_mean(barrier > 0.025),
        "timeout_rate": _safe_mean(timeout),
        "mean_mfe_mae_ratio": _safe_mean(mfe_mae),
        "top_symbol_share": (
            float(selected_frame["__symbol__"].astype(str).value_counts(normalize=True).iloc[0])
            if len(selected_frame)
            else 0.0
        ),
        "score_ic_label": _spearman(score, target["target_soft"]),
        "score_ic_u": _spearman(score, metrics["u_policy_net"]),
        "score_ic_bad_mae": _spearman(score, (metrics["mae_norm"] >= 1.0).astype(float)),
        "score_ic_timeout": _spearman(score, metrics["is_timeout"].astype(float)),
        "proxy_features": proxy_features,
        "train_rows": int(train_rows),
        "weight_arm": "W0_base",
        "train_weight_mean": 1.0,
        "train_weight_ess": float(train_weight_ess),
        "train_weight_ess_frac": float(train_weight_ess / train_rows) if train_rows else float("nan"),
    }


def _gated_selector_scores(
    *,
    label_proxy: pd.Series,
    hard_proxy: pd.Series,
    support_proxy: pd.Series,
    label_features: str,
    hard_features: str,
    support_features: str,
    gate_min_scores: list[float],
) -> list[tuple[str, pd.Series, str]]:
    label = _safe_numeric(label_proxy).reset_index(drop=True)
    hard = _safe_numeric(hard_proxy).reset_index(drop=True)
    support = _safe_numeric(support_proxy).reset_index(drop=True)
    selectors: list[tuple[str, pd.Series, str]] = []
    for gate in gate_min_scores:
        suffix = _fmt_score(float(gate))
        hard_mask = hard >= float(gate)
        support_mask = support >= float(gate)
        hard_blend = (0.65 * label + 0.35 * hard).clip(0.0, 1.0)
        support_blend = (0.65 * label + 0.35 * support).clip(0.0, 1.0)
        selectors.extend(
            [
                (
                    f"hard_gate{suffix}_label_proxy_oos",
                    label.where(hard_mask),
                    f"label={label_features}; hard={hard_features}",
                ),
                (
                    f"hard_gate{suffix}_blend_proxy_oos",
                    hard_blend.where(hard_mask),
                    f"label={label_features}; hard={hard_features}",
                ),
                (
                    f"hard_gate{suffix}_gate_proxy_oos",
                    hard.where(hard_mask),
                    f"hard={hard_features}",
                ),
                (
                    f"support_gate{suffix}_label_proxy_oos",
                    label.where(support_mask),
                    f"label={label_features}; support={support_features}",
                ),
                (
                    f"support_gate{suffix}_blend_proxy_oos",
                    support_blend.where(support_mask),
                    f"label={label_features}; support={support_features}",
                ),
                (
                    f"support_gate{suffix}_gate_proxy_oos",
                    support.where(support_mask),
                    f"support={support_features}",
                ),
            ]
        )
    return selectors


def _fit_holdout_summary(
    rows: pd.DataFrame,
    *,
    fit_months: list[str],
    holdout_month: str,
    max_timeout_rate: float,
) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    monthly = rows[rows["period_type"].eq("month")].copy()
    weekly = rows[rows["period_type"].eq("week")].copy()
    out: list[dict[str, Any]] = []
    for key, group in monthly.groupby(["selector", "label_arm", "top_k", "min_score"], observed=True, dropna=False):
        selector, label_arm, top_k, min_score = key
        fit = group[group["month"].astype(str).isin(fit_months)].copy()
        holdout = group[group["month"].astype(str).eq(str(holdout_month))].copy()
        if fit.empty or holdout.empty:
            continue
        wgroup = weekly[
            weekly["selector"].astype(str).eq(str(selector))
            & weekly["label_arm"].astype(str).eq(str(label_arm))
            & pd.to_numeric(weekly["top_k"], errors="coerce").eq(int(top_k))
            & pd.to_numeric(weekly["min_score"], errors="coerce").eq(float(min_score))
        ].copy()
        fit_week = wgroup[wgroup["month"].astype(str).isin(fit_months)]
        holdout_week = wgroup[wgroup["month"].astype(str).eq(str(holdout_month))]

        def q(frame: pd.DataFrame, col: str, value: float) -> float:
            return _safe_quantile(frame[col], value) if col in frame.columns and not frame.empty else float("nan")

        row: dict[str, Any] = {
            "selector": str(selector),
            "label_arm": str(label_arm),
            "top_k": int(top_k),
            "min_score": float(min_score),
            "fit_tb_hr_label": _safe_mean(fit["tb_hr_label"]),
            "holdout_tb_hr_label": _safe_mean(holdout["tb_hr_label"]),
            "fit_tb_ndcg_label": _safe_mean(fit["tb_ndcg_label"]),
            "holdout_tb_ndcg_label": _safe_mean(holdout["tb_ndcg_label"]),
            "fit_q25_week_tb_hr_label": q(fit_week, "tb_hr_label", 0.25),
            "fit_q10_week_tb_hr_label": q(fit_week, "tb_hr_label", 0.10),
            "holdout_q25_week_tb_hr_label": q(holdout_week, "tb_hr_label", 0.25),
            "holdout_q10_week_tb_hr_label": q(holdout_week, "tb_hr_label", 0.10),
            "fit_mean_return_net": _safe_mean(fit["mean_return_net"]),
            "holdout_mean_return_net": _safe_mean(holdout["mean_return_net"]),
            "fit_q05_return_net": _safe_mean(fit["q05_return_net"]),
            "holdout_q05_return_net": _safe_mean(holdout["q05_return_net"]),
            "fit_net_pnl": float(pd.to_numeric(fit["net_pnl"], errors="coerce").sum(skipna=True)),
            "holdout_net_pnl": float(pd.to_numeric(holdout["net_pnl"], errors="coerce").sum(skipna=True)),
            "fit_full_sl_rate": _safe_mean(fit["full_sl_rate"]),
            "holdout_full_sl_rate": _safe_mean(holdout["full_sl_rate"]),
            "fit_bad_mae_1r_rate": _safe_mean(fit["bad_mae_1r_rate"]),
            "holdout_bad_mae_1r_rate": _safe_mean(holdout["bad_mae_1r_rate"]),
            "fit_p90_mae_norm": _safe_mean(fit["p90_mae_norm"]),
            "holdout_p90_mae_norm": _safe_mean(holdout["p90_mae_norm"]),
            "fit_timeout_rate": _safe_mean(fit["timeout_rate"]),
            "holdout_timeout_rate": _safe_mean(holdout["timeout_rate"]),
            "fit_wide_barrier_25bps_rate": _safe_mean(fit["wide_barrier_25bps_rate"]),
            "holdout_wide_barrier_25bps_rate": _safe_mean(holdout["wide_barrier_25bps_rate"]),
            "fit_score_ic_u": _safe_mean(fit["score_ic_u"]),
            "holdout_score_ic_u": _safe_mean(holdout["score_ic_u"]),
            "fit_score_ic_bad_mae": _safe_mean(fit["score_ic_bad_mae"]),
            "holdout_score_ic_bad_mae": _safe_mean(holdout["score_ic_bad_mae"]),
            "fit_selected_rows": int(pd.to_numeric(fit["selected_rows"], errors="coerce").sum(skipna=True)),
            "holdout_selected_rows": int(pd.to_numeric(holdout["selected_rows"], errors="coerce").sum(skipna=True)),
            "proxy_features": str(group["proxy_features"].dropna().iloc[0]) if group["proxy_features"].dropna().size else "",
        }
        fit_econ = (
            row["fit_mean_return_net"] > 0.0
            and row["fit_score_ic_u"] > 0.0
            and row["fit_score_ic_bad_mae"] < 0.0
            and row["fit_bad_mae_1r_rate"] <= 0.40
            and row["fit_p90_mae_norm"] <= 4.0
            and row["fit_wide_barrier_25bps_rate"] <= 0.05
            and row["fit_timeout_rate"] <= float(max_timeout_rate)
        )
        holdout_econ = (
            row["holdout_mean_return_net"] > 0.0
            and row["holdout_score_ic_u"] > 0.0
            and row["holdout_score_ic_bad_mae"] < 0.0
            and row["holdout_bad_mae_1r_rate"] <= 0.40
            and row["holdout_p90_mae_norm"] <= 4.0
            and row["holdout_wide_barrier_25bps_rate"] <= 0.05
            and row["holdout_timeout_rate"] <= float(max_timeout_rate)
        )
        row["fit_economic_pass"] = bool(fit_econ)
        row["holdout_economic_pass"] = bool(holdout_econ)
        row["trainworthy_pass"] = bool(fit_econ and holdout_econ)
        row["rounda_objective"] = (
            row["fit_tb_hr_label"]
            + 0.50 * row["fit_tb_ndcg_label"]
            + 0.25 * row["fit_q25_week_tb_hr_label"]
            + 0.15 * row["fit_q10_week_tb_hr_label"]
            - 0.30 * row["fit_full_sl_rate"]
            - 0.20 * row["fit_timeout_rate"]
        )
        out.append(row)
    summary = pd.DataFrame(out)
    if summary.empty:
        return summary
    return summary.sort_values(
        ["trainworthy_pass", "holdout_economic_pass", "fit_economic_pass", "rounda_objective"],
        ascending=[False, False, False, False],
    )


def _write_markdown(output_dir: Path, summary: pd.DataFrame, period_rows: pd.DataFrame, manifest: dict[str, Any]) -> Path:
    path = output_dir / "soft_label_rounda_topk_proxy_diagnostics.md"
    cols = [
        "trainworthy_pass",
        "fit_economic_pass",
        "holdout_economic_pass",
        "selector",
        "label_arm",
        "top_k",
        "min_score",
        "rounda_objective",
        "fit_tb_hr_label",
        "holdout_tb_hr_label",
        "fit_tb_ndcg_label",
        "holdout_tb_ndcg_label",
        "fit_q25_week_tb_hr_label",
        "holdout_q25_week_tb_hr_label",
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
        "fit_score_ic_bad_mae",
        "holdout_score_ic_bad_mae",
    ]
    month_cols = [
        "month",
        "selector",
        "label_arm",
        "top_k",
        "min_score",
        "tb_hr_label",
        "tb_ndcg_label",
        "mean_return_net",
        "q05_return_net",
        "net_pnl",
        "full_sl_rate",
        "bad_mae_1r_rate",
        "p90_mae_norm",
        "timeout_rate",
        "score_ic_u",
        "score_ic_bad_mae",
        "proxy_features",
    ]
    month_rows = period_rows[period_rows["period_type"].eq("month")].copy()
    lines = [
        "# Soft Label Round-A Top-K Proxy Diagnostics",
        "",
        "Scope: proxy-only Round A. Weight arm is fixed at `W0_base`; no model training, Optuna, or policy optimisation is run.",
        "",
        f"Label arms: `{', '.join(manifest['label_arms'])}`.",
        f"Top-k values: `{manifest['top_ks']}`.",
        f"Minimum score thresholds: `{manifest['min_scores']}`.",
        f"Gate selectors: `{manifest['include_gate_selectors']}`. Gate thresholds: `{manifest['gate_min_scores']}`.",
        f"Proxy objective: `{manifest['proxy_objective']}`.",
        f"Fit months: `{', '.join(manifest['fit_months'])}`. Holdout: `{manifest['holdout_month']}`.",
        f"Features: `{manifest['feature_count']}`. Proxy top-k features: `{manifest['proxy_top_k']}`.",
        "",
        "## Fit/Holdout Summary",
        "",
        _table(summary, cols, limit=80),
        "",
        "## Month Detail",
        "",
        _table(
            month_rows.sort_values(["month", "selector", "label_arm", "top_k", "min_score"]),
            month_cols,
            limit=180,
        ),
        "",
        "## Outputs",
        "",
        f"- Period rows: `{manifest['outputs']['period_rows']}`",
        f"- Fit/holdout: `{manifest['outputs']['fit_holdout']}`",
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
    months: list[str],
    fit_months: list[str],
    holdout_month: str,
    top_ks: list[int],
    min_scores: list[float],
    include_gate_selectors: bool,
    gate_min_scores: list[float],
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
    max_timeout_rate: float,
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
    features = _feature_columns(frame)
    targets = _make_targets(frame, metrics)
    strict_targets = _strict_rounda_targets(frame=frame, metrics=metrics, base_targets=targets)
    targets.update(strict_targets)
    missing = sorted(set(label_arms) - set(targets))
    if missing:
        raise ValueError(f"Unknown label arms: {missing}")
    period = frame["__ts__"].dt.to_period("M").astype(str)
    rows: list[dict[str, Any]] = []
    for month in months:
        train_mask = period.lt(str(month))
        valid_mask = period.eq(str(month))
        if int(train_mask.sum()) < 500 or int(valid_mask.sum()) < 100:
            continue
        train = frame.loc[train_mask].copy()
        train_metrics = metrics.loc[train_mask].copy()
        valid = frame.loc[valid_mask].copy().reset_index(drop=True)
        valid_metrics = metrics.loc[valid_mask].copy().reset_index(drop=True)
        valid_indices = np.arange(len(valid), dtype=np.int64)
        period_slices = [("month", str(month), valid_indices)]
        period_slices.extend(("week", week, pos) for week, pos in _slice_week_positions(valid))
        for arm in label_arms:
            target = targets[arm]
            target_valid = target.loc[valid_mask].copy().reset_index(drop=True)
            proxy_score, diag = _rounda_proxy_score(
                train=train,
                valid=frame.loc[valid_mask].copy(),
                features=features,
                y_train=target.loc[train_mask, "target_soft"],
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
            proxy_score = proxy_score.reset_index(drop=True)
            label_proxy_features = ",".join(diag.get("proxy_features", []))
            selector_scores = [
                ("oracle_label_sort", target_valid["target_soft"], ""),
                ("label_ic_proxy_oos", proxy_score, label_proxy_features),
            ]
            if include_gate_selectors:
                hard_proxy, hard_diag = _rounda_proxy_score(
                    train=train,
                    valid=frame.loc[valid_mask].copy(),
                    features=features,
                    y_train=target.loc[train_mask, "target_hard"],
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
                support_proxy, support_diag = _rounda_proxy_score(
                    train=train,
                    valid=frame.loc[valid_mask].copy(),
                    features=features,
                    y_train=target.loc[train_mask, "target_soft"].gt(0.0).astype(float),
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
                selector_scores.extend(
                    _gated_selector_scores(
                        label_proxy=proxy_score,
                        hard_proxy=hard_proxy.reset_index(drop=True),
                        support_proxy=support_proxy.reset_index(drop=True),
                        label_features=label_proxy_features,
                        hard_features=",".join(hard_diag.get("proxy_features", [])),
                        support_features=",".join(support_diag.get("proxy_features", [])),
                        gate_min_scores=gate_min_scores,
                    )
                )
            train_rows = int(train_mask.sum())
            train_weight_ess = float(train_rows)
            for selector, score, proxy_features in selector_scores:
                score = _safe_numeric(score).reset_index(drop=True)
                for min_score in min_scores:
                    for period_type, period_name, pos in period_slices:
                        local_frame = valid.iloc[pos].reset_index(drop=True)
                        local_metrics = valid_metrics.iloc[pos].reset_index(drop=True)
                        local_target = target_valid.iloc[pos].reset_index(drop=True)
                        local_score = score.iloc[pos].reset_index(drop=True)
                        for top_k in top_ks:
                            rows.append(
                                _score_topk_period(
                                    frame=local_frame,
                                    metrics=local_metrics,
                                    target=local_target,
                                    score=local_score,
                                    period_type=period_type,
                                    period=str(period_name),
                                    month=str(month),
                                    selector=selector,
                                    label_arm=arm,
                                    top_k=int(top_k),
                                    min_score=float(min_score),
                                    proxy_features=proxy_features,
                                    train_rows=train_rows,
                                    train_weight_ess=train_weight_ess,
                                )
                            )
        print(json.dumps({"month": str(month), "progress": "complete"}))

    period_rows = pd.DataFrame(rows)
    fit_holdout = _fit_holdout_summary(
        period_rows,
        fit_months=fit_months,
        holdout_month=holdout_month,
        max_timeout_rate=max_timeout_rate,
    )
    paths = {
        "period_rows": output_dir / "soft_label_rounda_topk_period_rows.csv",
        "fit_holdout": output_dir / "soft_label_rounda_topk_fit_holdout.csv",
        "manifest": output_dir / "manifest.json",
    }
    period_rows.to_csv(paths["period_rows"], index=False)
    fit_holdout.to_csv(paths["fit_holdout"], index=False)
    manifest = {
        "scope": "soft_label_rounda_topk_proxy_diagnostics",
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
        "label_arms": list(label_arms),
        "strict_label_arms": list(STRICT_LABEL_ARMS),
        "strict_label_descriptions": {
            "S120_s3_clean_utility_veto": "S3 path quality multiplied by positive utility and strict clean-path gates.",
            "S121_s8_clean_rank_veto": "S8 timestamp-rank path quality after strict clean utility gating.",
            "S122_clean_dirty_contrast_rank": "S3/S8 clean utility contrast with high-MAE/timeout/wide-barrier penalty before timestamp rank.",
            "S123_fast_clean_path_rank": "Fast positive utility path requiring low MAE, low barrier, no timeout, and early MFE.",
            "S124_s3_net_floor_veto": "S3 path quality with sub-threshold net utility set to zero before selection.",
            "S125_s8_net_floor_rank_veto": "S8 path quality with sub-threshold net utility set to zero before timestamp ranking.",
            "S126_clean_net_direct_rank": "Direct clean net-utility rank with zero support below the economic floor.",
            "S127_fast_clean_net_rank": "Fast clean net-utility rank with zero support below the economic floor.",
        },
        "months": list(months),
        "fit_months": list(fit_months),
        "holdout_month": str(holdout_month),
        "top_ks": [int(v) for v in top_ks],
        "min_scores": [float(v) for v in min_scores],
        "include_gate_selectors": bool(include_gate_selectors),
        "gate_min_scores": [float(v) for v in gate_min_scores],
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
        "weight_arm": "W0_base",
        "max_timeout_rate": float(max_timeout_rate),
        "reports": reports,
        "outputs": {key: str(path) for key, path in paths.items()},
    }
    markdown = _write_markdown(
        output_dir=output_dir,
        summary=fit_holdout,
        period_rows=period_rows,
        manifest={**manifest, "outputs": {**manifest["outputs"], "markdown": str(output_dir / "soft_label_rounda_topk_proxy_diagnostics.md")}},
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
    parser.add_argument("--label-arms", default=",".join(DEFAULT_LABEL_ARMS))
    parser.add_argument("--months", default=",".join(DEFAULT_MONTHS))
    parser.add_argument("--fit-months", default=",".join(DEFAULT_FIT_MONTHS))
    parser.add_argument("--holdout-month", default=DEFAULT_HOLDOUT_MONTH)
    parser.add_argument("--top-ks", default=",".join(str(v) for v in DEFAULT_TOP_KS))
    parser.add_argument("--min-scores", default=",".join(str(v) for v in DEFAULT_MIN_SCORES))
    parser.add_argument("--include-gate-selectors", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--gate-min-scores", default=",".join(str(v) for v in DEFAULT_GATE_MIN_SCORES))
    parser.add_argument("--proxy-top-k", type=int, default=DEFAULT_PROXY_TOP_K)
    parser.add_argument("--proxy-objective", choices=("target_ic", "economic_ic", "economic_score"), default="target_ic")
    parser.add_argument("--proxy-min-target-ic", type=float, default=0.0)
    parser.add_argument("--proxy-min-utility-ic", type=float, default=0.0)
    parser.add_argument("--proxy-max-bad-mae-ic", type=float, default=0.0)
    parser.add_argument("--proxy-max-wide-ic", type=float, default=0.0)
    parser.add_argument("--proxy-max-timeout-ic", type=float, default=0.0)
    parser.add_argument("--proxy-utility-weight", type=float, default=1.0)
    parser.add_argument("--proxy-bad-mae-weight", type=float, default=1.0)
    parser.add_argument("--proxy-wide-weight", type=float, default=0.5)
    parser.add_argument("--proxy-timeout-weight", type=float, default=0.5)
    parser.add_argument("--max-timeout-rate", type=float, default=0.50)
    parser.add_argument("--include-causal-outcome-priors", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-causal-state-path-priors", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-event-confirmation-features", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-adverse-path-composites", action=argparse.BooleanOptionalAction, default=False)
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
        label_arms=_parse_csv(args.label_arms, DEFAULT_LABEL_ARMS),
        months=_parse_csv(args.months, DEFAULT_MONTHS),
        fit_months=_parse_csv(args.fit_months, DEFAULT_FIT_MONTHS),
        holdout_month=str(args.holdout_month),
        top_ks=_parse_int_csv(args.top_ks, DEFAULT_TOP_KS),
        min_scores=_parse_float_csv(args.min_scores),
        include_gate_selectors=bool(args.include_gate_selectors),
        gate_min_scores=_parse_float_csv(args.gate_min_scores),
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
        max_timeout_rate=float(args.max_timeout_rate),
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
