#!/usr/bin/env python3
"""Proxy-only label quality report under economic limits.

This is a pre-training diagnostic. It reuses the current label builders and
feature-store proxy code, but it does not fit LightGBM, Optuna, policy geometry,
or the cheap ExtraTrees smoke model.
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

from scripts.run_label_economic_proxy_ablation import (  # noqa: E402
    ECONOMIC_ARMS,
    LABEL_ARMS,
    _economic_targets,
    _label_targets,
)
from scripts.run_label_quality_proxy_diagnostics import (  # noqa: E402
    DEFAULT_FEATURE_DIR,
    _decile_diagnostics,
    _feature_columns,
    _json_safe,
    _load_feature_store_columns,
    _load_labels,
    _path_metrics,
    _proxy_score as _base_proxy_score,
    _read_feature_list,
    _safe_mean,
    _safe_quantile,
    _selection_metrics,
    _spearman,
    _make_targets as _base_label_targets,
)
from scripts.run_soft_label_economic_proxy_ablation import (  # noqa: E402
    DEFAULT_EVENT_FEATURE_STORE_FEATURES,
    DEFAULT_PRIOR_WINDOWS_DAYS,
    DEFAULT_STATE_PATH_PRIOR_FEATURES,
    _causal_outcome_prior_features,
    _causal_state_path_prior_features,
    _event_confirmation_features,
    _proxy_score as _topk_proxy_score,
)


DEFAULT_LABELS_PATH = Path(
    "data_perp/artifacts/20260702_094500_first_touch_c0_fast6_s10_policy_net_labels/labels"
)
DEFAULT_FEATURE_LIST_CSV = Path(
    "data_perp/artifacts/20260629_050000_lgbm_mda/quality_reports/base_model_feature_importance.csv"
)
DEFAULT_OUTPUT_DIR = Path(
    "data_perp/reports/label_proxy_quality_economic_limits_stage4_v1"
)
DEFAULT_LABEL_ARMS = (
    "S10_policy_net_soft",
    "S14_policy_net_path_blend",
    "S34_exec_guard_broad_policy",
    "S53_timeout_barrier_cap_path_blend",
    "S55_timeout_barrier_cap_exec_guard",
    "S57_timeout_tpnet_cap_path_blend",
)
DEFAULT_ECONOMIC_ARMS = ("raw_u_policy_net", "risk_u_mild", "risk_u_strict_fast")
DEFAULT_TOP_FRACS = (0.30, 0.10, 0.05, 0.03, 0.01)
DEFAULT_COMBINE_LABEL_WEIGHT = 0.50
DEFAULT_ECONOMIC_GATE_FRAC = 0.30
DEFAULT_PROXY_TOP_K = 8
DEFAULT_MAX_TIMEOUT_RATE: float | None = None
DEFAULT_FIT_MONTHS = ("2026-04", "2026-05")
DEFAULT_HOLDOUT_MONTH = "2026-06"
DEFAULT_STATE_PATH_RISK_GATE_FRACS = (0.30, 0.50)
DEFAULT_STATE_PATH_RISK_PENALTIES = (0.25, 0.50, 1.00)


def _parse_csv(value: str | None, default: tuple[str, ...]) -> list[str]:
    if value is None or not str(value).strip():
        return list(default)
    lowered = str(value).strip().lower()
    if lowered == "all":
        return []
    return [part.strip() for part in str(value).split(",") if part.strip()]


def _parse_float_csv(value: str | None, default: tuple[float, ...]) -> list[float]:
    if value is None or not str(value).strip():
        return list(default)
    return [float(part.strip()) for part in str(value).split(",") if part.strip()]


def _baseline(metrics: pd.DataFrame) -> dict[str, float]:
    return {
        "period_baseline_mean_u": _safe_mean(metrics["u_policy_net"]),
        "period_baseline_hit_u": _safe_mean(metrics["u_policy_net"] > 0.0),
        "period_baseline_q10_u": _safe_quantile(metrics["u_policy_net"], 0.10),
    }


def _add_delta(row: dict[str, Any], base: dict[str, float]) -> None:
    row.update(base)
    for source, base_key, out_key in (
        ("mean_u", "period_baseline_mean_u", "delta_mean_u_vs_period"),
        ("hit_u", "period_baseline_hit_u", "delta_hit_u_vs_period"),
        ("q10_u", "period_baseline_q10_u", "delta_q10_u_vs_period"),
    ):
        left = float(row.get(source, float("nan")))
        right = float(base.get(base_key, float("nan")))
        row[out_key] = left - right if math.isfinite(left) and math.isfinite(right) else float("nan")


def _top_gate(score: pd.Series, frac: float) -> pd.Series:
    score = pd.to_numeric(score, errors="coerce")
    out = pd.Series(False, index=score.index)
    valid = score.dropna()
    if valid.empty:
        return out
    k = max(1, int(math.ceil(float(frac) * len(valid))))
    chosen = valid.sort_values(ascending=False, kind="mergesort").head(k).index
    out.loc[chosen] = True
    return out


def _score_proxy(
    *,
    train: pd.DataFrame,
    valid: pd.DataFrame,
    features: list[str],
    y_train: pd.Series,
    proxy_top_k: int,
) -> tuple[pd.Series, dict[str, Any]]:
    if int(proxy_top_k) == DEFAULT_PROXY_TOP_K:
        return _base_proxy_score(train, valid, features, y_train)
    return _topk_proxy_score(
        train=train,
        valid=valid,
        features=features,
        target_train=y_train,
        top_k=int(proxy_top_k),
    )


def _mfe_mae(metrics: pd.DataFrame) -> pd.Series:
    ratio = metrics["mfe_norm"] / metrics["mae_norm"].clip(lower=0.25)
    return ratio.replace([np.inf, -np.inf], np.nan).clip(upper=10.0)


def _state_path_risk_targets(metrics: pd.DataFrame) -> dict[str, pd.Series]:
    mfe_mae = _mfe_mae(metrics)
    timeout = metrics["is_timeout"].astype(float).fillna(1.0)
    u = metrics["u_policy_net"].fillna(-0.02)
    mae = metrics["mae_norm"].fillna(10.0)
    barrier = metrics["barrier"].fillna(1.0)
    bounded = (
        (u > 0.0)
        & (mae <= 1.0)
        & (barrier <= 0.025)
        & (mfe_mae >= 1.25)
        & (timeout <= 0.0)
    ).astype(float)
    non_timeout = (timeout <= 0.0).astype(float)
    bad_mae = (mae >= 1.0).astype(float)
    dirty = (
        (mae >= 1.0)
        | (barrier > 0.025)
        | (mfe_mae < 1.25)
        | (timeout > 0.0)
        | (u <= 0.0)
    ).astype(float)
    profit_low_mae = ((u > 0.0) & (mae <= 1.0)).astype(float)
    profit_low_mae_no_timeout = ((u > 0.0) & (mae <= 1.0) & (timeout <= 0.0)).astype(float)
    decisive_profit_low_mae = (
        (u > 0.0)
        & (mae <= 1.0)
        & (mfe_mae >= 1.25)
        & (timeout <= 0.0)
    ).astype(float)
    path_quality = (
        0.35 * bounded
        + 0.25 * (1.0 - bad_mae)
        + 0.20 * (1.0 - dirty)
        + 0.10 * (1.0 - timeout.clip(0.0, 1.0))
        + 0.10 * (mfe_mae / 3.0).clip(0.0, 1.0)
    ).clip(0.0, 1.0)
    return {
        "bounded": bounded.astype(float),
        "non_timeout": non_timeout.astype(float),
        "bad_mae": bad_mae.astype(float),
        "dirty": dirty.astype(float),
        "profit_low_mae": profit_low_mae.astype(float),
        "profit_low_mae_no_timeout": profit_low_mae_no_timeout.astype(float),
        "decisive_profit_low_mae": decisive_profit_low_mae.astype(float),
        "path_quality": path_quality.astype(float),
    }


def _state_path_feature_columns(features: list[str]) -> list[str]:
    preferred_tokens = (
        "_bad_mae_",
        "_mae_norm_",
        "_bounded_",
        "_clean_",
        "_mfe_mae_",
        "_timeout_",
        "_wide_25_",
        "_hit_u_",
        "_mean_u_",
        "_count_",
    )
    cols = [
        feature
        for feature in features
        if feature.startswith("prior_xs_state_") and any(token in feature for token in preferred_tokens)
    ]
    return cols or [feature for feature in features if feature.startswith("prior_xs_state_")]


def _score_period(
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
    economic_arm: str,
    top_frac: float,
    label_score: pd.Series | None,
    economic_score: pd.Series | None,
    economic_target: pd.Series | None,
    label_proxy_features: str,
    economic_proxy_features: str,
) -> dict[str, Any]:
    score = pd.to_numeric(score.reset_index(drop=True), errors="coerce")
    frame = frame.reset_index(drop=True)
    metrics = metrics.reset_index(drop=True)
    target = target.reset_index(drop=True)
    row = _selection_metrics(
        frame=frame,
        metrics=metrics,
        target=target,
        score=score,
        arm=(
            f"{selector}::{label_arm}"
            if economic_arm == "none"
            else f"{selector}::{label_arm}::{economic_arm}"
        ),
        selector=selector,
        period=period,
        top_frac=top_frac,
    )
    _add_delta(row, _baseline(metrics))
    row.update(
        {
            "period_type": period_type,
            "month": month,
            "label_arm": label_arm,
            "economic_arm": economic_arm,
            "label_proxy_features": label_proxy_features,
            "economic_proxy_features": economic_proxy_features,
            "score_ic_u": _spearman(score, metrics["u_policy_net"]),
            "score_ic_label": (
                _spearman(score, label_score.reset_index(drop=True))
                if label_score is not None
                else float("nan")
            ),
            "score_ic_economic": (
                _spearman(score, economic_target.reset_index(drop=True))
                if economic_target is not None
                else float("nan")
            ),
        }
    )
    row.update(_decile_diagnostics(score, metrics["u_policy_net"]))
    return row


def _slice_week_positions(valid: pd.DataFrame) -> list[tuple[str, np.ndarray]]:
    weeks = valid["__ts__"].dt.to_period("W-SUN").astype(str)
    out: list[tuple[str, np.ndarray]] = []
    for week, ids in pd.Series(np.arange(len(valid)), index=valid.index).groupby(weeks, dropna=False):
        pos = ids.to_numpy(dtype=np.int64)
        if len(pos) >= 20:
            out.append((str(week), pos))
    return out


def _run_month(
    *,
    frame: pd.DataFrame,
    metrics: pd.DataFrame,
    targets: dict[str, pd.DataFrame],
    economic_targets: dict[str, pd.Series],
    features: list[str],
    month: str,
    label_arms: list[str],
    economic_arms: list[str],
    top_fracs: list[float],
    combine_label_weight: float,
    economic_gate_frac: float,
    proxy_top_k: int,
    include_state_path_risk_selectors: bool,
    state_path_risk_gate_fracs: list[float],
    state_path_risk_penalties: list[float],
) -> list[dict[str, Any]]:
    month_period = frame["__ts__"].dt.to_period("M").astype(str)
    train_mask = month_period < month
    valid_mask = month_period == month
    if int(train_mask.sum()) < 100 or int(valid_mask.sum()) < 50:
        return []

    train = frame.loc[train_mask].copy()
    valid = frame.loc[valid_mask].copy().reset_index(drop=True)
    valid_metrics = metrics.loc[valid_mask].copy().reset_index(drop=True)
    valid_indices = np.arange(len(valid), dtype=np.int64)

    label_scores: dict[str, pd.Series] = {}
    label_feature_names: dict[str, str] = {}
    for arm in label_arms:
        score, diag = _score_proxy(
            train=train,
            valid=frame.loc[valid_mask].copy(),
            features=features,
            y_train=targets[arm].loc[train_mask, "target_soft"],
            proxy_top_k=proxy_top_k,
        )
        label_scores[arm] = score.reset_index(drop=True)
        label_feature_names[arm] = ",".join(diag.get("proxy_features", []))

    economic_scores: dict[str, pd.Series] = {}
    economic_feature_names: dict[str, str] = {}
    for arm in economic_arms:
        score, diag = _score_proxy(
            train=train,
            valid=frame.loc[valid_mask].copy(),
            features=features,
            y_train=economic_targets[arm].loc[train_mask],
            proxy_top_k=proxy_top_k,
        )
        economic_scores[arm] = score.reset_index(drop=True)
        economic_feature_names[arm] = ",".join(diag.get("proxy_features", []))

    state_path_scores: dict[str, pd.Series] = {}
    state_path_feature_names: dict[str, str] = {}
    state_path_valid_targets: dict[str, pd.Series] = {}
    if include_state_path_risk_selectors:
        state_path_features = _state_path_feature_columns(features)
        path_targets = _state_path_risk_targets(metrics)
        if state_path_features:
            for arm in ("bad_mae", "dirty", "bounded", "profit_low_mae", "path_quality"):
                score, diag = _score_proxy(
                    train=train,
                    valid=frame.loc[valid_mask].copy(),
                    features=state_path_features,
                    y_train=path_targets[arm].loc[train_mask],
                    proxy_top_k=proxy_top_k,
                )
                state_path_scores[arm] = score.reset_index(drop=True)
                state_path_feature_names[arm] = ",".join(diag.get("proxy_features", []))
                state_path_valid_targets[arm] = path_targets[arm].loc[valid_mask].copy().reset_index(drop=True)

    selector_scores: list[dict[str, Any]] = []
    for label_arm in label_arms:
        target_valid = targets[label_arm].loc[valid_mask].copy().reset_index(drop=True)
        label_proxy = label_scores[label_arm]
        selector_scores.append(
            {
                "selector": "oracle_label_sort",
                "label_arm": label_arm,
                "economic_arm": "none",
                "score": target_valid["target_soft"],
                "target": target_valid,
                "label_score": target_valid["target_soft"],
                "economic_score": None,
                "economic_target": None,
                "label_proxy_features": "",
                "economic_proxy_features": "",
            }
        )
        selector_scores.append(
            {
                "selector": "label_ic_proxy_oos",
                "label_arm": label_arm,
                "economic_arm": "none",
                "score": label_proxy,
                "target": target_valid,
                "label_score": target_valid["target_soft"],
                "economic_score": None,
                "economic_target": None,
                "label_proxy_features": label_feature_names[label_arm],
                "economic_proxy_features": "",
            }
        )
        if state_path_scores:
            bounded_proxy = state_path_scores["bounded"]
            profit_low_mae_proxy = state_path_scores["profit_low_mae"]
            quality_proxy = state_path_scores["path_quality"]
            bad_mae_proxy = state_path_scores["bad_mae"]
            dirty_proxy = state_path_scores["dirty"]
            state_features = (
                "bounded="
                + state_path_feature_names.get("bounded", "")
                + "; profit_low_mae="
                + state_path_feature_names.get("profit_low_mae", "")
                + "; quality="
                + state_path_feature_names.get("path_quality", "")
                + "; bad_mae="
                + state_path_feature_names.get("bad_mae", "")
                + "; dirty="
                + state_path_feature_names.get("dirty", "")
            )
            selector_scores.extend(
                [
                    {
                        "selector": "state_path_bounded_proxy_oos",
                        "label_arm": label_arm,
                        "economic_arm": "state_path_bounded",
                        "score": bounded_proxy,
                        "target": target_valid,
                        "label_score": target_valid["target_soft"],
                        "economic_score": bounded_proxy,
                        "economic_target": state_path_valid_targets["bounded"],
                        "label_proxy_features": label_feature_names[label_arm],
                        "economic_proxy_features": state_features,
                    },
                    {
                        "selector": "state_path_profit_lowmae_proxy_oos",
                        "label_arm": label_arm,
                        "economic_arm": "state_path_profit_lowmae",
                        "score": profit_low_mae_proxy,
                        "target": target_valid,
                        "label_score": target_valid["target_soft"],
                        "economic_score": profit_low_mae_proxy,
                        "economic_target": state_path_valid_targets["profit_low_mae"],
                        "label_proxy_features": label_feature_names[label_arm],
                        "economic_proxy_features": state_features,
                    },
                    {
                        "selector": "state_path_quality_proxy_oos",
                        "label_arm": label_arm,
                        "economic_arm": "state_path_quality",
                        "score": quality_proxy,
                        "target": target_valid,
                        "label_score": target_valid["target_soft"],
                        "economic_score": quality_proxy,
                        "economic_target": state_path_valid_targets["path_quality"],
                        "label_proxy_features": label_feature_names[label_arm],
                        "economic_proxy_features": state_features,
                    },
                    {
                        "selector": "state_path_low_badmae_proxy_oos",
                        "label_arm": label_arm,
                        "economic_arm": "state_path_bad_mae",
                        "score": -bad_mae_proxy,
                        "target": target_valid,
                        "label_score": target_valid["target_soft"],
                        "economic_score": bad_mae_proxy,
                        "economic_target": state_path_valid_targets["bad_mae"],
                        "label_proxy_features": label_feature_names[label_arm],
                        "economic_proxy_features": state_features,
                    },
                    {
                        "selector": "state_path_low_dirty_proxy_oos",
                        "label_arm": label_arm,
                        "economic_arm": "state_path_dirty",
                        "score": -dirty_proxy,
                        "target": target_valid,
                        "label_score": target_valid["target_soft"],
                        "economic_score": dirty_proxy,
                        "economic_target": state_path_valid_targets["dirty"],
                        "label_proxy_features": label_feature_names[label_arm],
                        "economic_proxy_features": state_features,
                    },
                    {
                        "selector": f"combined_l{combine_label_weight:.2f}_label_state_path_quality_proxy_oos",
                        "label_arm": label_arm,
                        "economic_arm": "state_path_quality",
                        "score": combine_label_weight * label_proxy + (1.0 - combine_label_weight) * quality_proxy,
                        "target": target_valid,
                        "label_score": target_valid["target_soft"],
                        "economic_score": quality_proxy,
                        "economic_target": state_path_valid_targets["path_quality"],
                        "label_proxy_features": label_feature_names[label_arm],
                        "economic_proxy_features": state_features,
                    },
                    {
                        "selector": f"combined_l{combine_label_weight:.2f}_label_state_path_bounded_proxy_oos",
                        "label_arm": label_arm,
                        "economic_arm": "state_path_bounded",
                        "score": combine_label_weight * label_proxy + (1.0 - combine_label_weight) * bounded_proxy,
                        "target": target_valid,
                        "label_score": target_valid["target_soft"],
                        "economic_score": bounded_proxy,
                        "economic_target": state_path_valid_targets["bounded"],
                        "label_proxy_features": label_feature_names[label_arm],
                        "economic_proxy_features": state_features,
                    },
                ]
            )
            for gate_frac in state_path_risk_gate_fracs:
                label_gate = _top_gate(label_proxy, gate_frac)
                selector_scores.extend(
                    [
                        {
                            "selector": f"dual_label_state_path_bounded_gate{gate_frac:.2f}_oos",
                            "label_arm": label_arm,
                            "economic_arm": "state_path_bounded",
                            "score": label_proxy.where(label_gate & _top_gate(bounded_proxy, gate_frac)),
                            "target": target_valid,
                            "label_score": target_valid["target_soft"],
                            "economic_score": bounded_proxy,
                            "economic_target": state_path_valid_targets["bounded"],
                            "label_proxy_features": label_feature_names[label_arm],
                            "economic_proxy_features": state_features,
                        },
                        {
                            "selector": f"dual_label_state_path_profit_lowmae_gate{gate_frac:.2f}_oos",
                            "label_arm": label_arm,
                            "economic_arm": "state_path_profit_lowmae",
                            "score": label_proxy.where(label_gate & _top_gate(profit_low_mae_proxy, gate_frac)),
                            "target": target_valid,
                            "label_score": target_valid["target_soft"],
                            "economic_score": profit_low_mae_proxy,
                            "economic_target": state_path_valid_targets["profit_low_mae"],
                            "label_proxy_features": label_feature_names[label_arm],
                            "economic_proxy_features": state_features,
                        },
                        {
                            "selector": f"dual_label_state_path_low_badmae_gate{gate_frac:.2f}_oos",
                            "label_arm": label_arm,
                            "economic_arm": "state_path_bad_mae",
                            "score": label_proxy.where(label_gate & _top_gate(-bad_mae_proxy, gate_frac)),
                            "target": target_valid,
                            "label_score": target_valid["target_soft"],
                            "economic_score": bad_mae_proxy,
                            "economic_target": state_path_valid_targets["bad_mae"],
                            "label_proxy_features": label_feature_names[label_arm],
                            "economic_proxy_features": state_features,
                        },
                        {
                            "selector": f"state_path_quality_gate{gate_frac:.2f}_then_label_proxy_oos",
                            "label_arm": label_arm,
                            "economic_arm": "state_path_quality",
                            "score": label_proxy.where(_top_gate(quality_proxy, gate_frac)),
                            "target": target_valid,
                            "label_score": target_valid["target_soft"],
                            "economic_score": quality_proxy,
                            "economic_target": state_path_valid_targets["path_quality"],
                            "label_proxy_features": label_feature_names[label_arm],
                            "economic_proxy_features": state_features,
                        },
                        {
                            "selector": f"state_path_bounded_gate{gate_frac:.2f}_then_label_proxy_oos",
                            "label_arm": label_arm,
                            "economic_arm": "state_path_bounded",
                            "score": label_proxy.where(_top_gate(bounded_proxy, gate_frac)),
                            "target": target_valid,
                            "label_score": target_valid["target_soft"],
                            "economic_score": bounded_proxy,
                            "economic_target": state_path_valid_targets["bounded"],
                            "label_proxy_features": label_feature_names[label_arm],
                            "economic_proxy_features": state_features,
                        },
                        {
                            "selector": f"state_path_low_badmae_gate{gate_frac:.2f}_then_label_proxy_oos",
                            "label_arm": label_arm,
                            "economic_arm": "state_path_bad_mae",
                            "score": label_proxy.where(_top_gate(-bad_mae_proxy, gate_frac)),
                            "target": target_valid,
                            "label_score": target_valid["target_soft"],
                            "economic_score": bad_mae_proxy,
                            "economic_target": state_path_valid_targets["bad_mae"],
                            "label_proxy_features": label_feature_names[label_arm],
                            "economic_proxy_features": state_features,
                        },
                        {
                            "selector": f"state_path_low_dirty_gate{gate_frac:.2f}_then_label_proxy_oos",
                            "label_arm": label_arm,
                            "economic_arm": "state_path_dirty",
                            "score": label_proxy.where(_top_gate(-dirty_proxy, gate_frac)),
                            "target": target_valid,
                            "label_score": target_valid["target_soft"],
                            "economic_score": dirty_proxy,
                            "economic_target": state_path_valid_targets["dirty"],
                            "label_proxy_features": label_feature_names[label_arm],
                            "economic_proxy_features": state_features,
                        },
                    ]
                )
            for penalty in state_path_risk_penalties:
                selector_scores.extend(
                    [
                        {
                            "selector": f"label_minus_state_path_badmae_{penalty:.2f}",
                            "label_arm": label_arm,
                            "economic_arm": "state_path_bad_mae",
                            "score": label_proxy - float(penalty) * bad_mae_proxy,
                            "target": target_valid,
                            "label_score": target_valid["target_soft"],
                            "economic_score": bad_mae_proxy,
                            "economic_target": state_path_valid_targets["bad_mae"],
                            "label_proxy_features": label_feature_names[label_arm],
                            "economic_proxy_features": state_features,
                        },
                        {
                            "selector": f"label_minus_state_path_dirty_{penalty:.2f}",
                            "label_arm": label_arm,
                            "economic_arm": "state_path_dirty",
                            "score": label_proxy - float(penalty) * dirty_proxy,
                            "target": target_valid,
                            "label_score": target_valid["target_soft"],
                            "economic_score": dirty_proxy,
                            "economic_target": state_path_valid_targets["dirty"],
                            "label_proxy_features": label_feature_names[label_arm],
                            "economic_proxy_features": state_features,
                        },
                    ]
                )
        for economic_arm, economic_proxy in economic_scores.items():
            economic_valid = economic_targets[economic_arm].loc[valid_mask].copy().reset_index(drop=True)
            combined = combine_label_weight * label_proxy + (1.0 - combine_label_weight) * economic_proxy
            gated = label_proxy.where(_top_gate(economic_proxy, economic_gate_frac))
            selector_scores.append(
                {
                    "selector": f"combined_l{combine_label_weight:.2f}_label_economic_proxy_oos",
                    "label_arm": label_arm,
                    "economic_arm": economic_arm,
                    "score": combined,
                    "target": target_valid,
                    "label_score": target_valid["target_soft"],
                    "economic_score": economic_proxy,
                    "economic_target": economic_valid,
                    "label_proxy_features": label_feature_names[label_arm],
                    "economic_proxy_features": economic_feature_names[economic_arm],
                }
            )
            selector_scores.append(
                {
                    "selector": f"econ_gate{economic_gate_frac:.2f}_then_label_proxy_oos",
                    "label_arm": label_arm,
                    "economic_arm": economic_arm,
                    "score": gated,
                    "target": target_valid,
                    "label_score": target_valid["target_soft"],
                    "economic_score": economic_proxy,
                    "economic_target": economic_valid,
                    "label_proxy_features": label_feature_names[label_arm],
                    "economic_proxy_features": economic_feature_names[economic_arm],
                }
            )

    rows: list[dict[str, Any]] = []
    period_slices = [("month", month, valid_indices)]
    period_slices.extend(("week", week, pos) for week, pos in _slice_week_positions(valid))
    for spec in selector_scores:
        score = pd.to_numeric(spec["score"], errors="coerce").reset_index(drop=True)
        target = spec["target"].reset_index(drop=True)
        economic_score = (
            pd.to_numeric(spec["economic_score"], errors="coerce").reset_index(drop=True)
            if spec["economic_score"] is not None
            else None
        )
        economic_target = (
            pd.to_numeric(spec["economic_target"], errors="coerce").reset_index(drop=True)
            if spec["economic_target"] is not None
            else None
        )
        label_score = (
            pd.to_numeric(spec["label_score"], errors="coerce").reset_index(drop=True)
            if spec["label_score"] is not None
            else None
        )
        for period_type, period, pos in period_slices:
            local_frame = valid.iloc[pos].reset_index(drop=True)
            local_metrics = valid_metrics.iloc[pos].reset_index(drop=True)
            local_target = target.iloc[pos].reset_index(drop=True)
            local_score = score.iloc[pos].reset_index(drop=True)
            local_label_score = label_score.iloc[pos].reset_index(drop=True) if label_score is not None else None
            local_economic_score = (
                economic_score.iloc[pos].reset_index(drop=True) if economic_score is not None else None
            )
            local_economic_target = (
                economic_target.iloc[pos].reset_index(drop=True) if economic_target is not None else None
            )
            for top_frac in top_fracs:
                rows.append(
                    _score_period(
                        frame=local_frame,
                        metrics=local_metrics,
                        target=local_target,
                        score=local_score,
                        period_type=period_type,
                        period=period,
                        month=month,
                        selector=spec["selector"],
                        label_arm=spec["label_arm"],
                        economic_arm=spec["economic_arm"],
                        top_frac=top_frac,
                        label_score=local_label_score,
                        economic_score=local_economic_score,
                        economic_target=local_economic_target,
                        label_proxy_features=spec["label_proxy_features"],
                        economic_proxy_features=spec["economic_proxy_features"],
                    )
                )
    return rows


def _passes_timeout(row: dict[str, Any], key: str, max_timeout_rate: float | None) -> bool:
    if max_timeout_rate is None:
        return True
    value = float(row.get(key, float("nan")))
    return bool(math.isfinite(value) and value <= float(max_timeout_rate))


def _aggregate(
    rows: pd.DataFrame,
    *,
    min_material_selected_rows: int,
    max_timeout_rate: float | None,
) -> pd.DataFrame:
    if rows.empty:
        return rows
    out_rows: list[dict[str, Any]] = []
    group_cols = [
        "period_type",
        "selector",
        "label_arm",
        "economic_arm",
        "top_frac",
    ]
    for key, group in rows.groupby(group_cols, dropna=False, observed=True):
        period_type, selector, label_arm, economic_arm, top_frac = key
        mean_u = pd.to_numeric(group["mean_u"], errors="coerce")
        mean_return_net = pd.to_numeric(group["mean_return_net"], errors="coerce")
        selected_rows = pd.to_numeric(group["selected_rows"], errors="coerce")
        sum_return_net = (mean_return_net * selected_rows).replace([np.inf, -np.inf], np.nan)
        sum_return_net_plus10 = ((mean_return_net - 0.0010) * selected_rows).replace([np.inf, -np.inf], np.nan)
        sum_return_net_plus25 = ((mean_return_net - 0.0025) * selected_rows).replace([np.inf, -np.inf], np.nan)
        material = group[selected_rows.ge(int(min_material_selected_rows)).to_numpy()].copy()
        material_return = pd.to_numeric(material["mean_return_net"], errors="coerce") if not material.empty else pd.Series(dtype=float)
        material_rows = pd.to_numeric(material["selected_rows"], errors="coerce") if not material.empty else pd.Series(dtype=float)
        material_sum_return_net = (material_return * material_rows).replace([np.inf, -np.inf], np.nan)
        material_sum_return_net_plus10 = ((material_return - 0.0010) * material_rows).replace(
            [np.inf, -np.inf],
            np.nan,
        )
        material_sum_return_net_plus25 = ((material_return - 0.0025) * material_rows).replace(
            [np.inf, -np.inf],
            np.nan,
        )
        periods = int(len(group))
        positive_periods = int((mean_u > 0.0).sum())
        positive_return_periods = int((mean_return_net > 0.0).sum())
        material_periods = int(len(material))
        positive_material_return_periods = int((material_return > 0.0).sum()) if material_periods else 0
        row = {
            "period_type": period_type,
            "selector": selector,
            "label_arm": label_arm,
            "economic_arm": economic_arm,
            "top_frac": float(top_frac),
            "periods": periods,
            "positive_periods": positive_periods,
            "positive_period_rate": positive_periods / periods if periods else float("nan"),
            "positive_return_periods": positive_return_periods,
            "positive_return_period_rate": positive_return_periods / periods if periods else float("nan"),
            "material_periods": material_periods,
            "material_period_rate": material_periods / periods if periods else float("nan"),
            "positive_material_return_periods": positive_material_return_periods,
            "positive_material_return_period_rate": (
                positive_material_return_periods / material_periods if material_periods else float("nan")
            ),
            "mean_u": _safe_mean(mean_u),
            "worst_period_mean_u": _safe_quantile(mean_u, 0.0),
            "q25_period_mean_u": _safe_quantile(mean_u, 0.25),
            "mean_return_net": _safe_mean(mean_return_net),
            "worst_period_return_net": _safe_quantile(mean_return_net, 0.0),
            "q25_period_return_net": _safe_quantile(mean_return_net, 0.25),
            "sum_return_net": float(sum_return_net.sum(skipna=True)),
            "sum_return_net_plus10bps": float(sum_return_net_plus10.sum(skipna=True)),
            "sum_return_net_plus25bps": float(sum_return_net_plus25.sum(skipna=True)),
            "material_mean_return_net": _safe_mean(material_return) if material_periods else float("nan"),
            "material_worst_period_return_net": _safe_quantile(material_return, 0.0) if material_periods else float("nan"),
            "material_q25_period_return_net": _safe_quantile(material_return, 0.25) if material_periods else float("nan"),
            "material_sum_return_net": float(material_sum_return_net.sum(skipna=True)) if material_periods else 0.0,
            "material_sum_return_net_plus10bps": (
                float(material_sum_return_net_plus10.sum(skipna=True)) if material_periods else 0.0
            ),
            "material_sum_return_net_plus25bps": (
                float(material_sum_return_net_plus25.sum(skipna=True)) if material_periods else 0.0
            ),
            "hit_u": _safe_mean(group["hit_u"]),
            "hit_return_net": _safe_mean(group["hit_return_net"]),
            "q10_u": _safe_mean(group["q10_u"]),
            "delta_mean_u_vs_period": _safe_mean(group["delta_mean_u_vs_period"]),
            "delta_hit_u_vs_period": _safe_mean(group["delta_hit_u_vs_period"]),
            "delta_q10_u_vs_period": _safe_mean(group["delta_q10_u_vs_period"]),
            "score_ic_u": _safe_mean(group["score_ic_u"]),
            "score_ic_label": _safe_mean(group["score_ic_label"]),
            "score_ic_economic": _safe_mean(group["score_ic_economic"]),
            "decile_spearman_u": _safe_mean(group["decile_spearman_u"]),
            "top_bottom_decile_spread_u": _safe_mean(group["top_bottom_decile_spread_u"]),
            "bad_mae_1r_rate": _safe_mean(group["bad_mae_1r_rate"]),
            "p90_mae_norm": _safe_mean(group["p90_mae_norm"]),
            "mean_mfe_mae_ratio": _safe_mean(group["mean_mfe_mae_ratio"]),
            "clean_row_rate": _safe_mean(group["clean_row_rate"]),
            "strict_clean_row_rate": _safe_mean(group["strict_clean_row_rate"]),
            "bounded_row_rate": _safe_mean(group["bounded_row_rate"]),
            "wide_barrier_25bps_rate": _safe_mean(group["wide_barrier_25bps_rate"]),
            "wide_barrier_35bps_rate": _safe_mean(group["wide_barrier_35bps_rate"]),
            "timeout_rate": _safe_mean(group["timeout_rate"]),
            "mean_bars_to_mfe": _safe_mean(group["mean_bars_to_mfe"]),
            "mean_selected_rows": _safe_mean(group["selected_rows"]),
            "min_selected_rows": int(pd.to_numeric(group["selected_rows"], errors="coerce").min()),
            "material_bad_mae_1r_rate": _safe_mean(material["bad_mae_1r_rate"]) if material_periods else float("nan"),
            "material_p90_mae_norm": _safe_mean(material["p90_mae_norm"]) if material_periods else float("nan"),
            "material_mean_mfe_mae_ratio": (
                _safe_mean(material["mean_mfe_mae_ratio"]) if material_periods else float("nan")
            ),
            "material_clean_row_rate": _safe_mean(material["clean_row_rate"]) if material_periods else float("nan"),
            "material_strict_clean_row_rate": (
                _safe_mean(material["strict_clean_row_rate"]) if material_periods else float("nan")
            ),
            "material_bounded_row_rate": _safe_mean(material["bounded_row_rate"]) if material_periods else float("nan"),
            "material_wide_barrier_25bps_rate": (
                _safe_mean(material["wide_barrier_25bps_rate"]) if material_periods else float("nan")
            ),
            "material_timeout_rate": _safe_mean(material["timeout_rate"]) if material_periods else float("nan"),
            "material_mean_selected_rows": _safe_mean(material_rows) if material_periods else float("nan"),
            "material_min_selected_rows": int(material_rows.min()) if material_periods else 0,
            "top_symbol_share": _safe_mean(group["top_symbol_share"]),
            "label_proxy_features": str(group["label_proxy_features"].dropna().iloc[0])
            if group["label_proxy_features"].dropna().size
            else "",
            "economic_proxy_features": str(group["economic_proxy_features"].dropna().iloc[0])
            if group["economic_proxy_features"].dropna().size
            else "",
        }
        return_stable = (
            bool(row["positive_return_period_rate"] >= (1.0 if period_type == "month" else 0.60))
            and bool(row["q25_period_return_net"] > 0.0 if period_type == "week" else row["worst_period_return_net"] > 0.0)
            and bool(row["sum_return_net"] > 0.0)
            and bool(row["sum_return_net_plus10bps"] > 0.0)
        )
        row["acceptance_gate"] = (
            bool(row["mean_u"] > 0.0)
            and bool(row["mean_return_net"] > 0.0)
            and bool(row["delta_mean_u_vs_period"] > 0.0)
            and bool(row["positive_period_rate"] >= (1.0 if period_type == "month" else 0.60))
            and bool(row["q25_period_mean_u"] > 0.0 if period_type == "week" else row["worst_period_mean_u"] > 0.0)
            and return_stable
            and bool(row["score_ic_u"] > 0.0 if selector != "oracle_label_sort" else True)
            and bool(row["wide_barrier_25bps_rate"] <= 0.05)
            and bool(row["bad_mae_1r_rate"] <= 0.40)
            and bool(row["p90_mae_norm"] <= 4.00)
            and _passes_timeout(row, "timeout_rate", max_timeout_rate)
        )
        material_return_stable = (
            bool(row["material_period_rate"] >= 0.50)
            and bool(row["positive_material_return_period_rate"] >= 0.60)
            and bool(row["material_q25_period_return_net"] > 0.0)
            and bool(row["material_sum_return_net"] > 0.0)
            and bool(row["material_sum_return_net_plus10bps"] > 0.0)
        )
        row["material_acceptance_gate"] = (
            row["acceptance_gate"]
            if period_type != "week"
            else bool(row["mean_u"] > 0.0)
            and bool(row["mean_return_net"] > 0.0)
            and bool(row["delta_mean_u_vs_period"] > 0.0)
            and material_return_stable
            and bool(row["score_ic_u"] > 0.0 if selector != "oracle_label_sort" else True)
            and bool(row["material_wide_barrier_25bps_rate"] <= 0.05)
            and bool(row["material_bad_mae_1r_rate"] <= 0.40)
            and bool(row["material_p90_mae_norm"] <= 4.00)
            and _passes_timeout(row, "material_timeout_rate", max_timeout_rate)
        )
        out_rows.append(row)
    aggregate = pd.DataFrame(out_rows)
    return aggregate.sort_values(
        ["period_type", "top_frac", "mean_u", "worst_period_mean_u"],
        ascending=[True, True, False, False],
    )


def _weighted_mean(frame: pd.DataFrame, value_col: str, weight_col: str) -> float:
    if frame.empty or value_col not in frame.columns or weight_col not in frame.columns:
        return float("nan")
    values = pd.to_numeric(frame[value_col], errors="coerce")
    weights = pd.to_numeric(frame[weight_col], errors="coerce").fillna(0.0)
    mask = values.notna() & weights.gt(0.0)
    if not bool(mask.any()):
        return float("nan")
    return float(np.average(values[mask], weights=weights[mask]))


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
    monthly = period_rows[period_rows["period_type"].eq("month")].copy()
    weekly = period_rows[period_rows["period_type"].eq("week")].copy()
    if monthly.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    group_cols = ["selector", "label_arm", "economic_arm", "top_frac"]
    for key, group in monthly.groupby(group_cols, observed=True, dropna=False):
        selector, label_arm, economic_arm, top_frac = key
        week_group = weekly[
            weekly["selector"].astype(str).eq(str(selector))
            & weekly["label_arm"].astype(str).eq(str(label_arm))
            & weekly["economic_arm"].astype(str).eq(str(economic_arm))
            & pd.to_numeric(weekly["top_frac"], errors="coerce").eq(float(top_frac))
        ].copy()
        fit_month = group[group["month"].astype(str).isin(fit_months)].copy()
        holdout_monthly = group[group["month"].astype(str).eq(str(holdout_month))].copy()
        fit_week = week_group[week_group["month"].astype(str).isin(fit_months)].copy()
        holdout_week = week_group[week_group["month"].astype(str).eq(str(holdout_month))].copy()
        if fit_month.empty or holdout_monthly.empty:
            continue

        def week_stats(frame: pd.DataFrame) -> tuple[int, float, float]:
            returns = pd.to_numeric(frame["mean_return_net"], errors="coerce")
            selected = pd.to_numeric(frame["selected_rows"], errors="coerce").fillna(0.0)
            material = selected.ge(int(min_week_rows))
            if not bool(material.any()):
                return 0, float("nan"), float("nan")
            return (
                int(material.sum()),
                float((returns.gt(0.0) & material).sum() / material.sum()),
                _safe_quantile(returns[material], 0.25),
            )

        fit_returns = pd.to_numeric(fit_month["mean_return_net"], errors="coerce")
        holdout_returns = pd.to_numeric(holdout_monthly["mean_return_net"], errors="coerce")
        fit_material_weeks, fit_week_rate, fit_q25_week = week_stats(fit_week)
        holdout_material_weeks, holdout_week_rate, holdout_q25_week = week_stats(holdout_week)
        row: dict[str, Any] = {
            "selector": str(selector),
            "label_arm": str(label_arm),
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
            "fit_bad_mae_1r_rate": _weighted_mean(fit_month, "bad_mae_1r_rate", "selected_rows"),
            "holdout_bad_mae_1r_rate": _weighted_mean(holdout_monthly, "bad_mae_1r_rate", "selected_rows"),
            "fit_p90_mae_norm": _weighted_mean(fit_month, "p90_mae_norm", "selected_rows"),
            "holdout_p90_mae_norm": _weighted_mean(holdout_monthly, "p90_mae_norm", "selected_rows"),
            "fit_wide_barrier_25bps_rate": _weighted_mean(fit_month, "wide_barrier_25bps_rate", "selected_rows"),
            "holdout_wide_barrier_25bps_rate": _weighted_mean(
                holdout_monthly,
                "wide_barrier_25bps_rate",
                "selected_rows",
            ),
            "fit_timeout_rate": _weighted_mean(fit_month, "timeout_rate", "selected_rows"),
            "holdout_timeout_rate": _weighted_mean(holdout_monthly, "timeout_rate", "selected_rows"),
            "fit_score_ic_u": _safe_mean(fit_month["score_ic_u"]),
            "holdout_score_ic_u": _safe_mean(holdout_monthly["score_ic_u"]),
            "fit_decile_spearman_u": _safe_mean(fit_month["decile_spearman_u"]),
            "holdout_decile_spearman_u": _safe_mean(holdout_monthly["decile_spearman_u"]),
            "fit_selected_rows": int(pd.to_numeric(fit_month["selected_rows"], errors="coerce").sum(skipna=True)),
            "holdout_selected_rows": int(
                pd.to_numeric(holdout_monthly["selected_rows"], errors="coerce").sum(skipna=True)
            ),
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
        fit_economic = (
            fit_sign
            and row["fit_score_ic_u"] > 0.0
            and row["fit_bad_mae_1r_rate"] <= 0.40
            and row["fit_p90_mae_norm"] <= 4.00
            and row["fit_wide_barrier_25bps_rate"] <= 0.05
            and _passes_timeout(row, "fit_timeout_rate", max_timeout_rate)
        )
        holdout_economic = (
            holdout_sign
            and row["holdout_score_ic_u"] > 0.0
            and row["holdout_bad_mae_1r_rate"] <= 0.40
            and row["holdout_p90_mae_norm"] <= 4.00
            and row["holdout_wide_barrier_25bps_rate"] <= 0.05
            and _passes_timeout(row, "holdout_timeout_rate", max_timeout_rate)
        )
        row["fit_sign_pass"] = bool(fit_sign)
        row["holdout_sign_pass"] = bool(holdout_sign)
        row["fit_economic_pass"] = bool(fit_economic)
        row["holdout_economic_pass"] = bool(holdout_economic)
        row["trainworthy_pass"] = bool(fit_economic and holdout_economic)
        rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(
        ["trainworthy_pass", "holdout_economic_pass", "fit_economic_pass", "holdout_mean_return_net"],
        ascending=[False, False, False, False],
    )


def _table(frame: pd.DataFrame, cols: list[str], limit: int | None = None) -> str:
    if frame.empty:
        return "No rows."
    view = frame[[c for c in cols if c in frame.columns]].copy()
    if limit is not None:
        view = view.head(limit)
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda v: f"{float(v):.4f}" if pd.notna(v) else "")
    return view.to_markdown(index=False)


def _write_markdown(
    output_dir: Path,
    aggregate: pd.DataFrame,
    monthly: pd.DataFrame,
    fit_holdout: pd.DataFrame,
    manifest: dict[str, Any],
) -> Path:
    path = output_dir / "label_proxy_quality_economic_limits.md"
    cols = [
        "acceptance_gate",
        "material_acceptance_gate",
        "selector",
        "label_arm",
        "economic_arm",
        "periods",
        "positive_periods",
        "positive_period_rate",
        "material_periods",
        "material_period_rate",
        "mean_u",
        "worst_period_mean_u",
        "q25_period_mean_u",
        "mean_return_net",
        "worst_period_return_net",
        "q25_period_return_net",
        "positive_return_period_rate",
        "positive_material_return_period_rate",
        "material_q25_period_return_net",
        "sum_return_net",
        "sum_return_net_plus10bps",
        "sum_return_net_plus25bps",
        "material_sum_return_net_plus10bps",
        "delta_mean_u_vs_period",
        "score_ic_u",
        "score_ic_label",
        "decile_spearman_u",
        "bad_mae_1r_rate",
        "p90_mae_norm",
        "mean_mfe_mae_ratio",
        "clean_row_rate",
        "strict_clean_row_rate",
        "bounded_row_rate",
        "material_bad_mae_1r_rate",
        "material_p90_mae_norm",
        "material_mean_mfe_mae_ratio",
        "material_clean_row_rate",
        "material_strict_clean_row_rate",
        "material_bounded_row_rate",
        "wide_barrier_25bps_rate",
        "timeout_rate",
        "mean_selected_rows",
    ]
    fit_cols = [
        "trainworthy_pass",
        "fit_economic_pass",
        "holdout_economic_pass",
        "fit_sign_pass",
        "holdout_sign_pass",
        "selector",
        "label_arm",
        "economic_arm",
        "top_frac",
        "fit_mean_return_net",
        "holdout_mean_return_net",
        "fit_bad_mae_1r_rate",
        "holdout_bad_mae_1r_rate",
        "fit_p90_mae_norm",
        "holdout_p90_mae_norm",
        "fit_wide_barrier_25bps_rate",
        "holdout_wide_barrier_25bps_rate",
        "fit_score_ic_u",
        "holdout_score_ic_u",
        "fit_decile_spearman_u",
        "holdout_decile_spearman_u",
        "fit_selected_rows",
        "holdout_selected_rows",
    ]
    lines = [
        "# Label Proxy Quality Within Economic Limits",
        "",
        "Scope: proxy-only label QA. No LightGBM, Optuna, policy geometry optimization, or cheap tree smoke is run.",
        "",
        "The acceptance gate requires positive utility versus the period baseline, positive return-net including +10 bps stress, positive proxy IC for causal selectors, monthly or weekly stability, and bounded path risk.",
        "",
        "Path-risk thresholds: `wide_barrier_25bps_rate <= 5%`, `bad_mae_1r_rate <= 40%`, and `p90_mae_norm <= 4R`.",
        f"Timeout threshold: `{manifest['max_timeout_rate']}` (`null` means disabled).",
        f"Proxy top-k features: `{manifest['proxy_top_k']}`",
        f"Causal outcome priors: `{manifest['include_causal_outcome_priors']}`",
        f"Causal state-path priors: `{manifest['include_causal_state_path_priors']}`",
        f"Event confirmation features: `{manifest['include_event_confirmation_features']}`",
        f"Material week selected-row floor: `{manifest['min_material_selected_rows']}`",
        f"Fit months: `{', '.join(manifest['fit_months'])}`. Holdout month: `{manifest['holdout_month']}`.",
        f"State-path risk selectors: `{manifest['include_state_path_risk_selectors']}`",
        f"State-path risk gate fractions: `{manifest['state_path_risk_gate_fracs']}`",
        f"State-path risk penalties: `{manifest['state_path_risk_penalties']}`",
        "",
    ]
    for period_type in ("month", "week"):
        for frac in manifest["top_fracs"]:
            subset = aggregate[
                aggregate["period_type"].eq(period_type) & aggregate["top_frac"].eq(float(frac))
            ].sort_values(["acceptance_gate", "mean_u", "worst_period_mean_u"], ascending=[False, False, False])
            frac_label = f"{float(frac):.0%}" if float(frac) >= 0.01 else f"{float(frac):.1%}"
            lines.extend(
                [
                    f"## {period_type.title()} Aggregate Top {frac_label}",
                    "",
                    _table(subset, cols, limit=30),
                    "",
                ]
            )
    month_focus = monthly[
        monthly["selector"].isin(["oracle_label_sort", "label_ic_proxy_oos"])
        & monthly["top_frac"].isin([0.10, 0.05, 0.03])
    ].sort_values(["period", "selector", "label_arm", "top_frac"])
    lines.extend(
        [
            "## Fit / Holdout Gate",
            "",
            "Train-worthy rows require April-May fit economics, June holdout economics, positive score IC, bad-MAE <= 40%, p90 MAE <= 4R, and wide-barrier <= 5%.",
            "",
            _table(fit_holdout, fit_cols, limit=60),
            "",
        ]
    )
    lines.extend(
        [
            "## Month Detail",
            "",
            _table(
                month_focus,
                [
                    "period",
                    "selector",
                    "label_arm",
                    "top_frac",
                    "selected_rows",
                    "mean_u",
                    "delta_mean_u_vs_period",
                    "hit_u",
                    "q10_u",
                    "score_ic_u",
                    "bad_mae_1r_rate",
                    "wide_barrier_25bps_rate",
                    "timeout_rate",
                ],
                limit=120,
            ),
            "",
            "## Outputs",
            "",
            f"- Period rows: `{manifest['outputs']['period_rows']}`",
            f"- Aggregate: `{manifest['outputs']['aggregate']}`",
            f"- Fit/holdout: `{manifest['outputs']['fit_holdout']}`",
            f"- Markdown: `{manifest['outputs']['markdown']}`",
            f"- Manifest: `{manifest['outputs']['manifest']}`",
        ]
    )
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
    economic_arms: list[str],
    top_fracs: list[float],
    combine_label_weight: float,
    economic_gate_frac: float,
    proxy_top_k: int,
    include_causal_outcome_priors: bool,
    include_causal_state_path_priors: bool,
    include_event_confirmation_features: bool,
    prior_windows_days: list[float],
    prior_embargo_hours: float,
    state_path_prior_features: list[str],
    event_feature_store_features: list[str],
    min_material_selected_rows: int,
    max_timeout_rate: float | None,
    fit_months: list[str],
    holdout_month: str,
    include_state_path_risk_selectors: bool,
    state_path_risk_gate_fracs: list[float],
    state_path_risk_penalties: list[float],
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

    metrics = _path_metrics(frame)
    causal_outcome_report: dict[str, Any] = {"enabled": False}
    causal_state_report: dict[str, Any] = {"enabled": False}
    event_confirmation_report: dict[str, Any] = {"enabled": False}
    if include_causal_outcome_priors:
        prior_features, causal_outcome_report = _causal_outcome_prior_features(
            frame,
            metrics,
            windows_days=prior_windows_days,
            embargo_hours=prior_embargo_hours,
        )
        if not prior_features.empty:
            frame = pd.concat([frame, prior_features.astype(np.float32, copy=False)], axis=1).copy()
    if include_causal_state_path_priors:
        state_features, causal_state_report = _causal_state_path_prior_features(
            frame,
            metrics,
            state_features=state_path_prior_features,
            windows_days=prior_windows_days,
            embargo_hours=prior_embargo_hours,
        )
        if not state_features.empty:
            frame = pd.concat([frame, state_features.astype(np.float32, copy=False)], axis=1).copy()
    if include_event_confirmation_features:
        event_features, event_confirmation_report = _event_confirmation_features(
            frame,
            event_features=event_feature_store_features,
        )
        if not event_features.empty:
            frame = pd.concat([frame, event_features.astype(np.float32, copy=False)], axis=1).copy()

    features = _feature_columns(frame)
    targets = _base_label_targets(frame, metrics)
    targets.update(_label_targets(frame, metrics))
    economic_targets = _economic_targets(metrics)
    if not label_arms:
        label_arms = list(LABEL_ARMS)
    if not economic_arms:
        economic_arms = list(ECONOMIC_ARMS)
    missing_labels = sorted(set(label_arms) - set(targets))
    missing_economic = sorted(set(economic_arms) - set(economic_targets))
    if missing_labels:
        raise ValueError(f"Unknown label arms: {missing_labels}")
    if missing_economic:
        raise ValueError(f"Unknown economic arms: {missing_economic}")

    months = sorted(frame["__ts__"].dt.to_period("M").dropna().astype(str).unique())
    rows: list[dict[str, Any]] = []
    for month in months[1:]:
        rows.extend(
            _run_month(
                frame=frame,
                metrics=metrics,
                targets=targets,
                economic_targets=economic_targets,
                features=features,
                month=month,
                label_arms=label_arms,
                economic_arms=economic_arms,
                top_fracs=top_fracs,
                combine_label_weight=combine_label_weight,
                economic_gate_frac=economic_gate_frac,
                proxy_top_k=proxy_top_k,
                include_state_path_risk_selectors=include_state_path_risk_selectors,
                state_path_risk_gate_fracs=state_path_risk_gate_fracs,
                state_path_risk_penalties=state_path_risk_penalties,
            )
        )
    period_rows = pd.DataFrame(rows)
    aggregate = _aggregate(
        period_rows,
        min_material_selected_rows=min_material_selected_rows,
        max_timeout_rate=max_timeout_rate,
    )
    fit_holdout = _fit_holdout_summary(
        period_rows,
        fit_months=fit_months,
        holdout_month=holdout_month,
        min_week_rows=min_material_selected_rows,
        max_timeout_rate=max_timeout_rate,
    )

    paths = {
        "period_rows": output_dir / "label_proxy_quality_period_rows.csv",
        "aggregate": output_dir / "label_proxy_quality_aggregate.csv",
        "fit_holdout": output_dir / "label_proxy_quality_fit_holdout.csv",
        "manifest": output_dir / "manifest.json",
    }
    period_rows.to_csv(paths["period_rows"], index=False)
    aggregate.to_csv(paths["aggregate"], index=False)
    fit_holdout.to_csv(paths["fit_holdout"], index=False)
    manifest = {
        "scope": "proxy_only_label_quality_with_economic_limits",
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
        "include_causal_outcome_priors": bool(include_causal_outcome_priors),
        "include_causal_state_path_priors": bool(include_causal_state_path_priors),
        "include_event_confirmation_features": bool(include_event_confirmation_features),
        "prior_windows_days": [float(v) for v in prior_windows_days],
        "prior_embargo_hours": float(prior_embargo_hours),
        "causal_outcome_priors": causal_outcome_report,
        "causal_state_path_priors": causal_state_report,
        "event_confirmation_features": event_confirmation_report,
        "feature_count": int(len(features)),
        "label_arms": label_arms,
        "economic_arms": economic_arms,
        "top_fracs": [float(v) for v in top_fracs],
        "combine_label_weight": float(combine_label_weight),
        "economic_gate_frac": float(economic_gate_frac),
        "proxy_top_k": int(proxy_top_k),
        "min_material_selected_rows": int(min_material_selected_rows),
        "max_timeout_rate": None if max_timeout_rate is None else float(max_timeout_rate),
        "fit_months": [str(value) for value in fit_months],
        "holdout_month": str(holdout_month),
        "include_state_path_risk_selectors": bool(include_state_path_risk_selectors),
        "state_path_risk_gate_fracs": [float(value) for value in state_path_risk_gate_fracs],
        "state_path_risk_penalties": [float(value) for value in state_path_risk_penalties],
        "months": months[1:],
        "outputs": {key: str(value) for key, value in paths.items()},
    }
    markdown = _write_markdown(
        output_dir=output_dir,
        aggregate=aggregate,
        monthly=period_rows[period_rows["period_type"].eq("month")].copy(),
        fit_holdout=fit_holdout,
        manifest={**manifest, "outputs": {**manifest["outputs"], "markdown": str(output_dir / "label_proxy_quality_economic_limits.md")}},
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
    parser.add_argument("--label-arms", type=str, default=",".join(DEFAULT_LABEL_ARMS))
    parser.add_argument("--economic-arms", type=str, default=",".join(DEFAULT_ECONOMIC_ARMS))
    parser.add_argument("--top-fracs", type=str, default=",".join(str(v) for v in DEFAULT_TOP_FRACS))
    parser.add_argument("--combine-label-weight", type=float, default=DEFAULT_COMBINE_LABEL_WEIGHT)
    parser.add_argument("--economic-gate-frac", type=float, default=DEFAULT_ECONOMIC_GATE_FRAC)
    parser.add_argument("--proxy-top-k", type=int, default=DEFAULT_PROXY_TOP_K)
    parser.add_argument("--min-material-selected-rows", type=int, default=5)
    parser.add_argument("--max-timeout-rate", type=float, default=DEFAULT_MAX_TIMEOUT_RATE)
    parser.add_argument("--fit-months", type=str, default=",".join(DEFAULT_FIT_MONTHS))
    parser.add_argument("--holdout-month", type=str, default=DEFAULT_HOLDOUT_MONTH)
    parser.add_argument("--include-state-path-risk-selectors", action="store_true")
    parser.add_argument(
        "--state-path-risk-gate-fracs",
        type=str,
        default=",".join(str(v) for v in DEFAULT_STATE_PATH_RISK_GATE_FRACS),
    )
    parser.add_argument(
        "--state-path-risk-penalties",
        type=str,
        default=",".join(str(v) for v in DEFAULT_STATE_PATH_RISK_PENALTIES),
    )
    parser.add_argument("--include-causal-outcome-priors", action="store_true")
    parser.add_argument("--include-causal-state-path-priors", action="store_true")
    parser.add_argument("--include-event-confirmation-features", action="store_true")
    parser.add_argument("--prior-windows-days", type=str, default=",".join(str(v) for v in DEFAULT_PRIOR_WINDOWS_DAYS))
    parser.add_argument("--prior-embargo-hours", type=float, default=24.0)
    parser.add_argument(
        "--state-path-prior-features",
        type=str,
        default=",".join(DEFAULT_STATE_PATH_PRIOR_FEATURES),
    )
    parser.add_argument(
        "--event-feature-store-features",
        type=str,
        default=",".join(DEFAULT_EVENT_FEATURE_STORE_FEATURES),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = run_report(
        labels_path=args.labels_path,
        output_dir=args.output_dir,
        feature_dir=args.feature_dir,
        feature_list_csv=args.feature_list_csv,
        max_feature_store_features=args.max_feature_store_features,
        label_arms=_parse_csv(args.label_arms, DEFAULT_LABEL_ARMS),
        economic_arms=_parse_csv(args.economic_arms, DEFAULT_ECONOMIC_ARMS),
        top_fracs=_parse_float_csv(args.top_fracs, DEFAULT_TOP_FRACS),
        combine_label_weight=float(args.combine_label_weight),
        economic_gate_frac=float(args.economic_gate_frac),
        proxy_top_k=int(args.proxy_top_k),
        include_causal_outcome_priors=bool(args.include_causal_outcome_priors),
        include_causal_state_path_priors=bool(args.include_causal_state_path_priors),
        include_event_confirmation_features=bool(args.include_event_confirmation_features),
        prior_windows_days=_parse_float_csv(args.prior_windows_days, DEFAULT_PRIOR_WINDOWS_DAYS),
        prior_embargo_hours=float(args.prior_embargo_hours),
        state_path_prior_features=_parse_csv(args.state_path_prior_features, DEFAULT_STATE_PATH_PRIOR_FEATURES),
        event_feature_store_features=_parse_csv(args.event_feature_store_features, DEFAULT_EVENT_FEATURE_STORE_FEATURES),
        min_material_selected_rows=int(args.min_material_selected_rows),
        max_timeout_rate=args.max_timeout_rate,
        fit_months=_parse_csv(args.fit_months, DEFAULT_FIT_MONTHS),
        holdout_month=str(args.holdout_month),
        include_state_path_risk_selectors=bool(args.include_state_path_risk_selectors),
        state_path_risk_gate_fracs=_parse_float_csv(
            args.state_path_risk_gate_fracs,
            DEFAULT_STATE_PATH_RISK_GATE_FRACS,
        ),
        state_path_risk_penalties=_parse_float_csv(
            args.state_path_risk_penalties,
            DEFAULT_STATE_PATH_RISK_PENALTIES,
        ),
    )
    print(json.dumps(_json_safe(manifest), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
