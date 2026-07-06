#!/usr/bin/env python3
"""Dual-target execution-risk smoke for label candidates.

This is a pre-production diagnostic. It tests whether the path-risk part of
execution can be learned separately from the upside label: for each OOT month it
fits a cheap upside model plus bad-MAE / wide-barrier / timeout models on prior
months only, then selects by a composite score.
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

from scripts.run_label_economic_proxy_ablation import LABEL_ARMS, _label_targets
from scripts.run_label_feature_store_model_smoke import (
    _add_delta_fields,
    _fit_predict,
    _month_model_frame,
    _parse_csv,
    _parse_int_csv,
)
from scripts.run_label_quality_proxy_diagnostics import (
    DEFAULT_FEATURE_DIR,
    DEFAULT_FEATURE_LIST_CSV,
    DEFAULT_LABELS_DIR,
    TOP_FRACS,
    _decile_diagnostics,
    _effective_n,
    _feature_columns,
    _json_safe,
    _load_feature_store_columns,
    _load_labels,
    _path_metrics,
    _rank_top_indices,
    _read_feature_list,
    _safe_mean,
    _safe_quantile,
    _selection_metrics,
    _spearman,
)
from scripts.run_label_weighted_proxy_ablation import (
    PROXY_METHODS,
    WEIGHT_ARMS,
    _effective_sample_size,
    _weight_series,
    _weighted_proxy_score,
)
from scripts.run_soft_label_economic_proxy_ablation import (  # noqa: E402
    DEFAULT_EVENT_FEATURE_STORE_FEATURES,
    DEFAULT_PRIOR_WINDOWS_DAYS,
    DEFAULT_STATE_PATH_PRIOR_FEATURES,
    _causal_outcome_prior_features,
    _causal_state_path_prior_features,
    _event_confirmation_features,
)


DEFAULT_OUTPUT_DIR = Path("data_perp/reports/label_dual_target_execution_smoke_v1")
DEFAULT_TOP_FRACS = tuple(float(v) for v in TOP_FRACS)
DEFAULT_LABEL_ARMS = (
    "S32_econ_limited_broad_policy",
    "S34_exec_guard_broad_policy",
)
DEFAULT_WEIGHT_ARMS = (
    "W9_tail_utility",
    "W10_payoff_clean",
    "W12_tail_timestamp_balanced",
)
DEFAULT_MAE_PENALTIES = (0.0, 0.25, 0.50, 0.75, 1.00)
DEFAULT_WIDE_PENALTIES = (0.0, 0.25, 0.50)
DEFAULT_TIMEOUT_PENALTIES = (0.0, 0.15)
DEFAULT_MAE_KEEP_FRACS = (1.0,)
DEFAULT_WIDE_KEEP_FRACS = (1.0,)
DEFAULT_TIMEOUT_KEEP_FRACS = (1.0,)


def _parse_float_csv(value: str | None, default: tuple[float, ...]) -> list[float]:
    if value is None or not str(value).strip():
        return list(default)
    return [float(part.strip()) for part in str(value).split(",") if part.strip()]


def _rank_pct(values: Any) -> pd.Series:
    return pd.to_numeric(pd.Series(values), errors="coerce").rank(method="average", pct=True).fillna(0.5)


def _risk_targets(metrics: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "bad_mae_1r": (metrics["mae_norm"] >= 1.0).astype(float),
        "wide_barrier_25bps": (metrics["barrier"] > 0.025).astype(float),
        "timeout": metrics["is_timeout"].astype(float),
    }


def _baseline_row(valid_metrics: pd.DataFrame) -> dict[str, float]:
    return {
        "period_baseline_mean_u": _safe_mean(valid_metrics["u_policy_net"]),
        "period_baseline_hit_u": _safe_mean(valid_metrics["u_policy_net"] > 0.0),
        "period_baseline_q10_u": _safe_quantile(valid_metrics["u_policy_net"], 0.10),
    }


def _gate_mask_from_ranks(
    *,
    bad_mae_rank: pd.Series,
    wide_rank: pd.Series,
    timeout_rank: pd.Series,
    mae_keep_frac: float,
    wide_keep_frac: float,
    timeout_keep_frac: float,
) -> pd.Series:
    return (
        (bad_mae_rank <= float(mae_keep_frac))
        & (wide_rank <= float(wide_keep_frac))
        & (timeout_rank <= float(timeout_keep_frac))
    ).fillna(False)


def _gated_selection_metrics(
    *,
    frame: pd.DataFrame,
    metrics: pd.DataFrame,
    target: pd.DataFrame,
    score: pd.Series,
    gate_mask: pd.Series,
    arm: str,
    period: str,
    top_frac: float,
) -> dict[str, Any]:
    gate_mask = pd.Series(gate_mask).reset_index(drop=True).fillna(False).astype(bool)
    gated_score = pd.Series(score).reset_index(drop=True).where(gate_mask)
    gate_rows = int(gate_mask.sum())
    gate_rate = gate_rows / float(len(gate_mask)) if len(gate_mask) else 0.0
    selection_frac_within_gate = min(1.0, float(top_frac) / gate_rate) if gate_rate > 0.0 else float(top_frac)
    row = _selection_metrics(
        frame=frame,
        metrics=metrics,
        target=target,
        score=gated_score,
        arm=arm,
        selector="dual_target_execution_risk_gate_seed_ensemble_smoke_oos",
        period=period,
        top_frac=selection_frac_within_gate,
    )
    row["top_frac"] = float(top_frac)
    row["selection_frac_within_gate"] = float(selection_frac_within_gate)
    row["gate_rows"] = gate_rows
    row["gate_candidate_rate"] = float(gate_rate)
    return row


def _selection_weekly_rows(
    *,
    frame: pd.DataFrame,
    metrics: pd.DataFrame,
    target: pd.DataFrame,
    score: pd.Series,
    arm: str,
    selector: str,
    period: str,
    top_frac: float,
    gate_mask: pd.Series | None = None,
    selection_frac: float | None = None,
) -> list[dict[str, Any]]:
    score_series = pd.Series(score).reset_index(drop=True)
    if gate_mask is not None:
        score_series = score_series.where(pd.Series(gate_mask).reset_index(drop=True).fillna(False).astype(bool))
    idx = _rank_top_indices(score_series, float(selection_frac if selection_frac is not None else top_frac))
    if not len(idx):
        return []

    selected_frame = frame.iloc[idx].reset_index(drop=True)
    selected_metrics = metrics.iloc[idx].reset_index(drop=True)
    selected_target = target.iloc[idx].reset_index(drop=True)
    weeks = selected_frame["__ts__"].dt.to_period("W-SUN").astype(str)
    rows: list[dict[str, Any]] = []
    for week, ids in pd.Series(np.arange(len(selected_frame))).groupby(weeks, dropna=False):
        pos = ids.to_numpy(dtype=np.int64)
        week_metrics = selected_metrics.iloc[pos]
        week_frame = selected_frame.iloc[pos]
        week_target = selected_target.iloc[pos]
        symbols = week_frame.get("__symbol__", pd.Series(dtype=object))
        timestamps = week_frame.get("__ts__", pd.Series(dtype="datetime64[ns]"))
        utility = week_metrics["u_policy_net"]
        mfe_mae_ratio = (
            week_metrics["mfe_norm"] / week_metrics["mae_norm"].clip(lower=0.25)
        ).replace([np.inf, -np.inf], np.nan).clip(upper=10.0)
        rows.append(
            {
                "arm": arm,
                "selector": selector,
                "period": period,
                "week": str(week),
                "top_frac": float(top_frac),
                "rows": int(len(frame)),
                "selected_rows": int(len(idx)),
                "week_selected_rows": int(len(pos)),
                "week_selected_share": float(len(pos) / len(idx)) if len(idx) else 0.0,
                "target_top_soft_mean": _safe_mean(week_target.get("target_soft")),
                "target_top_hard_rate": _safe_mean(week_target.get("target_hard")),
                "mean_u": _safe_mean(utility),
                "median_u": _safe_quantile(utility, 0.50),
                "q10_u": _safe_quantile(utility, 0.10),
                "hit_u": _safe_mean(utility > 0.0),
                "mean_return_net": _safe_mean(week_metrics["ret_net"]),
                "hit_return_net": _safe_mean(week_metrics["ret_net"] > 0.0),
                "mean_barrier": _safe_mean(week_metrics["barrier"]),
                "wide_barrier_25bps_rate": _safe_mean(week_metrics["barrier"] > 0.025),
                "bad_mae_1r_rate": _safe_mean(week_metrics["mae_norm"] >= 1.0),
                "p90_mae_norm": _safe_quantile(week_metrics["mae_norm"], 0.90),
                "mean_mfe_mae_ratio": _safe_mean(mfe_mae_ratio),
                "clean_row_rate": _safe_mean(
                    (week_metrics["u_policy_net"] > 0.0)
                    & (week_metrics["mae_norm"] <= 1.0)
                    & (week_metrics["barrier"] <= 0.025)
                    & (week_metrics["is_timeout"].astype(float) <= 0.0)
                ),
                "strict_clean_row_rate": _safe_mean(
                    (week_metrics["u_policy_net"] > 0.0)
                    & (week_metrics["mae_norm"] <= 0.85)
                    & (week_metrics["barrier"] <= 0.024)
                    & (mfe_mae_ratio >= 1.35)
                    & (week_metrics["is_timeout"].astype(float) <= 0.0)
                ),
                "bounded_row_rate": _safe_mean(
                    (week_metrics["u_policy_net"] > 0.0)
                    & (week_metrics["mae_norm"] <= 1.0)
                    & (week_metrics["barrier"] <= 0.035)
                    & (mfe_mae_ratio >= 1.25)
                    & (week_metrics["is_timeout"].astype(float) <= 0.0)
                ),
                "timeout_rate": _safe_mean(week_metrics["is_timeout"].astype(float)),
                "symbol_effective_n": _effective_n(symbols),
                "top_symbol_share": float(symbols.value_counts(normalize=True, dropna=False).iloc[0])
                if len(symbols)
                else 0.0,
                "timestamp_effective_n": _effective_n(timestamps.astype(str)),
                "top_timestamp_share": float(timestamps.astype(str).value_counts(normalize=True, dropna=False).iloc[0])
                if len(timestamps)
                else 0.0,
            }
        )
    return rows


def _selected_ledger_rows(
    *,
    frame: pd.DataFrame,
    metrics: pd.DataFrame,
    target: pd.DataFrame,
    score: pd.Series,
    upside: pd.Series,
    bad_mae_pred: pd.Series,
    wide_pred: pd.Series,
    timeout_pred: pd.Series,
    upside_rank: pd.Series,
    bad_mae_rank: pd.Series,
    wide_rank: pd.Series,
    timeout_rank: pd.Series,
    arm: str,
    selector: str,
    period: str,
    top_frac: float,
    gate_mask: pd.Series | None = None,
    selection_frac: float | None = None,
) -> list[dict[str, Any]]:
    score_series = pd.Series(score).reset_index(drop=True)
    gate_series = pd.Series(True, index=score_series.index)
    if gate_mask is not None:
        gate_series = pd.Series(gate_mask).reset_index(drop=True).fillna(False).astype(bool)
        score_series = score_series.where(gate_series)
    idx = _rank_top_indices(score_series, float(selection_frac if selection_frac is not None else top_frac))
    if not len(idx):
        return []

    score_values = pd.to_numeric(score_series, errors="coerce")
    selected_scores = score_values.iloc[idx].to_numpy(dtype=np.float64)
    order_rank = pd.Series((-selected_scores).argsort(kind="mergesort")).rank(method="first").astype(int)
    rows: list[dict[str, Any]] = []
    for rank_pos, pos in enumerate(idx, start=1):
        row_metrics = metrics.iloc[int(pos)]
        row_target = target.iloc[int(pos)]
        row_frame = frame.iloc[int(pos)]
        rows.append(
            {
                "arm": arm,
                "selector": selector,
                "period": period,
                "week": str(pd.Timestamp(row_frame["__ts__"]).to_period("W-SUN")),
                "top_frac": float(top_frac),
                "selection_frac_within_gate": float(selection_frac if selection_frac is not None else top_frac),
                "selected_rank": int(rank_pos),
                "selected_rows": int(len(idx)),
                "row_index": int(pos),
                "timestamp": row_frame["__ts__"],
                "symbol": row_frame["__symbol__"],
                "score": float(score_values.iloc[int(pos)]) if pd.notna(score_values.iloc[int(pos)]) else float("nan"),
                "upside_pred": float(upside.iloc[int(pos)]),
                "bad_mae_pred": float(bad_mae_pred.iloc[int(pos)]),
                "wide_pred": float(wide_pred.iloc[int(pos)]),
                "timeout_pred": float(timeout_pred.iloc[int(pos)]),
                "upside_rank": float(upside_rank.iloc[int(pos)]),
                "bad_mae_rank": float(bad_mae_rank.iloc[int(pos)]),
                "wide_rank": float(wide_rank.iloc[int(pos)]),
                "timeout_rank": float(timeout_rank.iloc[int(pos)]),
                "passed_risk_gate": bool(gate_series.iloc[int(pos)]),
                "target_soft": float(row_target.get("target_soft", float("nan"))),
                "target_hard": float(row_target.get("target_hard", float("nan"))),
                "u_policy_net": float(row_metrics["u_policy_net"]),
                "ret_net": float(row_metrics["ret_net"]),
                "barrier": float(row_metrics["barrier"]),
                "mae_norm": float(row_metrics["mae_norm"]),
                "mfe_norm": float(row_metrics["mfe_norm"]),
                "bars_to_mfe": float(row_metrics["bars_to_mfe"]),
                "is_timeout": bool(row_metrics["is_timeout"]),
                "bad_mae_1r": bool(row_metrics["mae_norm"] >= 1.0),
                "wide_barrier_25bps": bool(row_metrics["barrier"] > 0.025),
            }
        )
    return rows


def _seed_average_predict(
    *,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    w_train: pd.Series,
    x_valid: pd.DataFrame,
    seeds: list[int],
) -> tuple[pd.Series, float, float]:
    preds = [
        _fit_predict(
            x_train=x_train,
            y_train=y_train,
            w_train=w_train,
            x_valid=x_valid,
            seed=seed,
        )
        for seed in seeds
    ]
    matrix = np.vstack(preds)
    pred = np.mean(matrix, axis=0).astype(np.float32)
    std = np.std(matrix, axis=0).astype(np.float32) if len(preds) > 1 else np.zeros_like(pred)
    return pd.Series(pred), float(np.mean(std)), float(np.percentile(std, 90))


def _run_month(
    *,
    frame: pd.DataFrame,
    metrics: pd.DataFrame,
    targets: dict[str, pd.DataFrame],
    features: list[str],
    month: str,
    label_arms: list[str],
    weight_arms: list[str],
    seeds: list[int],
    model_feature_selector: str,
    model_feature_tail_frac: float,
    mae_penalties: list[float],
    wide_penalties: list[float],
    timeout_penalties: list[float],
    mae_keep_fracs: list[float],
    wide_keep_fracs: list[float],
    timeout_keep_fracs: list[float],
    top_fracs: list[float],
    emit_selected_ledger: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    month_period = frame["__ts__"].dt.to_period("M").astype(str)
    train_mask = month_period < month
    valid_mask = month_period == month
    if int(train_mask.sum()) < 500 or int(valid_mask.sum()) < 100:
        return [], [], [
            {
                "period": month,
                "skipped": True,
                "train_rows": int(train_mask.sum()),
                "valid_rows": int(valid_mask.sum()),
            }
        ]

    x_train_all, x_valid_all = _month_model_frame(
        frame,
        train_mask=train_mask,
        valid_mask=valid_mask,
        features=features,
    )
    train = frame.loc[train_mask].copy()
    valid = frame.loc[valid_mask].copy().reset_index(drop=True)
    train_metrics = metrics.loc[train_mask].copy()
    valid_metrics = metrics.loc[valid_mask].copy().reset_index(drop=True)
    risk_train = _risk_targets(train_metrics)
    risk_valid = _risk_targets(valid_metrics)
    baseline = _baseline_row(valid_metrics)
    rows: list[dict[str, Any]] = []
    weekly_rows: list[dict[str, Any]] = []
    selected_ledger_rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []

    for label_arm in label_arms:
        target_train = targets[label_arm].loc[train_mask].copy()
        target_valid = targets[label_arm].loc[valid_mask].copy().reset_index(drop=True)
        for weight_arm in weight_arms:
            weights = _weight_series(
                frame=train,
                metrics=train_metrics,
                target=target_train,
                arm=weight_arm,
            )
            selector_diag: dict[str, Any] = {}
            model_features = list(features)
            if model_feature_selector != "all":
                _unused_score, selector_diag = _weighted_proxy_score(
                    train,
                    frame.loc[valid_mask].copy(),
                    features,
                    target_train["target_soft"],
                    weights,
                    method=model_feature_selector,
                    tail_frac=model_feature_tail_frac,
                )
                selected = [
                    feature
                    for feature in selector_diag.get("proxy_features", [])
                    if feature in x_train_all.columns
                ]
                if selected:
                    model_features = selected

            x_train = x_train_all[model_features]
            x_valid = x_valid_all[model_features]
            upside, upside_seed_std_mean, upside_seed_std_p90 = _seed_average_predict(
                x_train=x_train,
                y_train=target_train["target_soft"],
                w_train=weights,
                x_valid=x_valid,
                seeds=seeds,
            )
            bad_mae_pred, bad_mae_seed_std_mean, _bad_mae_seed_std_p90 = _seed_average_predict(
                x_train=x_train,
                y_train=risk_train["bad_mae_1r"],
                w_train=weights,
                x_valid=x_valid,
                seeds=seeds,
            )
            wide_pred, wide_seed_std_mean, _wide_seed_std_p90 = _seed_average_predict(
                x_train=x_train,
                y_train=risk_train["wide_barrier_25bps"],
                w_train=weights,
                x_valid=x_valid,
                seeds=seeds,
            )
            timeout_pred, timeout_seed_std_mean, _timeout_seed_std_p90 = _seed_average_predict(
                x_train=x_train,
                y_train=risk_train["timeout"],
                w_train=weights,
                x_valid=x_valid,
                seeds=seeds,
            )
            upside = upside.reset_index(drop=True)
            bad_mae_pred = bad_mae_pred.reset_index(drop=True)
            wide_pred = wide_pred.reset_index(drop=True)
            timeout_pred = timeout_pred.reset_index(drop=True)
            upside_rank = _rank_pct(upside)
            bad_mae_rank = _rank_pct(bad_mae_pred)
            wide_rank = _rank_pct(wide_pred)
            timeout_rank = _rank_pct(timeout_pred)
            diagnostics.append(
                {
                    "period": month,
                    "label_arm": label_arm,
                    "weight_arm": weight_arm,
                    "model_feature_selector": model_feature_selector,
                    "model_feature_count": int(len(model_features)),
                    "model_features": ",".join(model_features),
                    "train_rows": int(train_mask.sum()),
                    "valid_rows": int(valid_mask.sum()),
                    "weight_effective_frac": _effective_sample_size(weights) / float(len(weights))
                    if len(weights)
                    else float("nan"),
                    "upside_ic_u": _spearman(upside, valid_metrics["u_policy_net"]),
                    "upside_ic_label": _spearman(upside, target_valid["target_soft"]),
                    "bad_mae_ic_actual": _spearman(bad_mae_pred, risk_valid["bad_mae_1r"]),
                    "wide_ic_actual": _spearman(wide_pred, risk_valid["wide_barrier_25bps"]),
                    "timeout_ic_actual": _spearman(timeout_pred, risk_valid["timeout"]),
                    "upside_seed_std_mean": upside_seed_std_mean,
                    "upside_seed_std_p90": upside_seed_std_p90,
                    "bad_mae_seed_std_mean": bad_mae_seed_std_mean,
                    "wide_seed_std_mean": wide_seed_std_mean,
                    "timeout_seed_std_mean": timeout_seed_std_mean,
                    "proxy_features": ",".join(selector_diag.get("proxy_features", [])),
                }
            )
            for mae_penalty in mae_penalties:
                for wide_penalty in wide_penalties:
                    for timeout_penalty in timeout_penalties:
                        score = (
                            upside_rank
                            - float(mae_penalty) * bad_mae_rank
                            - float(wide_penalty) * wide_rank
                            - float(timeout_penalty) * timeout_rank
                        )
                        decile = _decile_diagnostics(score, valid_metrics["u_policy_net"])
                        for top_frac in top_fracs:
                            arm = (
                                f"{label_arm}::{weight_arm}::dual_exec"
                                f"::mae{mae_penalty:.2f}"
                                f"::wide{wide_penalty:.2f}"
                                f"::timeout{timeout_penalty:.2f}"
                            )
                            row = _selection_metrics(
                                frame=valid,
                                metrics=valid_metrics,
                                target=target_valid,
                                score=score,
                                arm=arm,
                                selector="dual_target_execution_seed_ensemble_smoke_oos",
                                period=month,
                                top_frac=top_frac,
                            )
                            _add_delta_fields(row, baseline)
                            row.update(
                                {
                                    "label_arm": label_arm,
                                    "weight_arm": weight_arm,
                                    "model_feature_selector": model_feature_selector,
                                    "model_feature_count": int(len(model_features)),
                                    "model_features": ",".join(model_features),
                                    "selection_mode": "penalty_score",
                                    "mae_penalty": float(mae_penalty),
                                    "wide_penalty": float(wide_penalty),
                                    "timeout_penalty": float(timeout_penalty),
                                    "mae_keep_frac": 1.0,
                                    "wide_keep_frac": 1.0,
                                    "timeout_keep_frac": 1.0,
                                    "selection_frac_within_gate": float(top_frac),
                                    "gate_rows": int(len(valid)),
                                    "gate_candidate_rate": 1.0,
                                    "score_ic_u": _spearman(score, valid_metrics["u_policy_net"]),
                                    "score_ic_label": _spearman(score, target_valid["target_soft"]),
                                    "upside_ic_u": _spearman(upside, valid_metrics["u_policy_net"]),
                                    "bad_mae_ic_actual": _spearman(
                                        bad_mae_pred,
                                        risk_valid["bad_mae_1r"],
                                    ),
                                    "wide_ic_actual": _spearman(
                                        wide_pred,
                                        risk_valid["wide_barrier_25bps"],
                                    ),
                                    "timeout_ic_actual": _spearman(
                                        timeout_pred,
                                        risk_valid["timeout"],
                                    ),
                                    **decile,
                                }
                            )
                            rows.append(row)
                            if emit_selected_ledger:
                                for ledger_row in _selected_ledger_rows(
                                    frame=valid,
                                    metrics=valid_metrics,
                                    target=target_valid,
                                    score=score,
                                    upside=upside,
                                    bad_mae_pred=bad_mae_pred,
                                    wide_pred=wide_pred,
                                    timeout_pred=timeout_pred,
                                    upside_rank=upside_rank,
                                    bad_mae_rank=bad_mae_rank,
                                    wide_rank=wide_rank,
                                    timeout_rank=timeout_rank,
                                    arm=arm,
                                    selector="dual_target_execution_seed_ensemble_smoke_oos",
                                    period=month,
                                    top_frac=top_frac,
                                ):
                                    ledger_row.update(
                                        {
                                            "label_arm": label_arm,
                                            "weight_arm": weight_arm,
                                            "model_feature_selector": model_feature_selector,
                                            "model_feature_count": int(len(model_features)),
                                            "selection_mode": "penalty_score",
                                            "mae_penalty": float(mae_penalty),
                                            "wide_penalty": float(wide_penalty),
                                            "timeout_penalty": float(timeout_penalty),
                                            "mae_keep_frac": 1.0,
                                            "wide_keep_frac": 1.0,
                                            "timeout_keep_frac": 1.0,
                                            "gate_rows": int(len(valid)),
                                            "gate_candidate_rate": 1.0,
                                        }
                                    )
                                    selected_ledger_rows.append(ledger_row)
                            for weekly_row in _selection_weekly_rows(
                                frame=valid,
                                metrics=valid_metrics,
                                target=target_valid,
                                score=score,
                                arm=arm,
                                selector="dual_target_execution_seed_ensemble_smoke_oos",
                                period=month,
                                top_frac=top_frac,
                            ):
                                weekly_row.update(
                                    {
                                        "label_arm": label_arm,
                                        "weight_arm": weight_arm,
                                        "model_feature_selector": model_feature_selector,
                                        "model_feature_count": int(len(model_features)),
                                        "model_features": ",".join(model_features),
                                        "selection_mode": "penalty_score",
                                        "mae_penalty": float(mae_penalty),
                                        "wide_penalty": float(wide_penalty),
                                        "timeout_penalty": float(timeout_penalty),
                                        "mae_keep_frac": 1.0,
                                        "wide_keep_frac": 1.0,
                                        "timeout_keep_frac": 1.0,
                                        "selection_frac_within_gate": float(top_frac),
                                        "gate_rows": int(len(valid)),
                                        "gate_candidate_rate": 1.0,
                                    }
                                )
                                weekly_rows.append(weekly_row)
                        for mae_keep_frac in mae_keep_fracs:
                            for wide_keep_frac in wide_keep_fracs:
                                for timeout_keep_frac in timeout_keep_fracs:
                                    if (
                                        float(mae_keep_frac) >= 1.0
                                        and float(wide_keep_frac) >= 1.0
                                        and float(timeout_keep_frac) >= 1.0
                                    ):
                                        continue
                                    gate_mask = _gate_mask_from_ranks(
                                        bad_mae_rank=bad_mae_rank,
                                        wide_rank=wide_rank,
                                        timeout_rank=timeout_rank,
                                        mae_keep_frac=float(mae_keep_frac),
                                        wide_keep_frac=float(wide_keep_frac),
                                        timeout_keep_frac=float(timeout_keep_frac),
                                    )
                                    gated_decile = _decile_diagnostics(
                                        score.where(gate_mask),
                                        valid_metrics["u_policy_net"],
                                    )
                                    for top_frac in top_fracs:
                                        arm = (
                                            f"{label_arm}::{weight_arm}::dual_exec_gate"
                                            f"::mae{mae_penalty:.2f}"
                                            f"::wide{wide_penalty:.2f}"
                                            f"::timeout{timeout_penalty:.2f}"
                                            f"::keepmae{float(mae_keep_frac):.2f}"
                                            f"::keepwide{float(wide_keep_frac):.2f}"
                                            f"::keeptime{float(timeout_keep_frac):.2f}"
                                        )
                                        row = _gated_selection_metrics(
                                            frame=valid,
                                            metrics=valid_metrics,
                                            target=target_valid,
                                            score=score,
                                            gate_mask=gate_mask,
                                            arm=arm,
                                            period=month,
                                            top_frac=top_frac,
                                        )
                                        _add_delta_fields(row, baseline)
                                        row.update(
                                            {
                                                "label_arm": label_arm,
                                                "weight_arm": weight_arm,
                                                "model_feature_selector": model_feature_selector,
                                                "model_feature_count": int(len(model_features)),
                                                "model_features": ",".join(model_features),
                                                "selection_mode": "risk_gate",
                                                "mae_penalty": float(mae_penalty),
                                                "wide_penalty": float(wide_penalty),
                                                "timeout_penalty": float(timeout_penalty),
                                                "mae_keep_frac": float(mae_keep_frac),
                                                "wide_keep_frac": float(wide_keep_frac),
                                                "timeout_keep_frac": float(timeout_keep_frac),
                                                "score_ic_u": _spearman(
                                                    score.where(gate_mask),
                                                    valid_metrics["u_policy_net"],
                                                ),
                                                "score_ic_label": _spearman(
                                                    score.where(gate_mask),
                                                    target_valid["target_soft"],
                                                ),
                                                "upside_ic_u": _spearman(upside, valid_metrics["u_policy_net"]),
                                                "bad_mae_ic_actual": _spearman(
                                                    bad_mae_pred,
                                                    risk_valid["bad_mae_1r"],
                                                ),
                                                "wide_ic_actual": _spearman(
                                                    wide_pred,
                                                    risk_valid["wide_barrier_25bps"],
                                                ),
                                                "timeout_ic_actual": _spearman(
                                                    timeout_pred,
                                                    risk_valid["timeout"],
                                                ),
                                                **gated_decile,
                                            }
                                        )
                                        rows.append(row)
                                        if emit_selected_ledger:
                                            for ledger_row in _selected_ledger_rows(
                                                frame=valid,
                                                metrics=valid_metrics,
                                                target=target_valid,
                                                score=score,
                                                upside=upside,
                                                bad_mae_pred=bad_mae_pred,
                                                wide_pred=wide_pred,
                                                timeout_pred=timeout_pred,
                                                upside_rank=upside_rank,
                                                bad_mae_rank=bad_mae_rank,
                                                wide_rank=wide_rank,
                                                timeout_rank=timeout_rank,
                                                gate_mask=gate_mask,
                                                arm=arm,
                                                selector="dual_target_execution_risk_gate_seed_ensemble_smoke_oos",
                                                period=month,
                                                top_frac=top_frac,
                                                selection_frac=row.get("selection_frac_within_gate", top_frac),
                                            ):
                                                ledger_row.update(
                                                    {
                                                        "label_arm": label_arm,
                                                        "weight_arm": weight_arm,
                                                        "model_feature_selector": model_feature_selector,
                                                        "model_feature_count": int(len(model_features)),
                                                        "selection_mode": "risk_gate",
                                                        "mae_penalty": float(mae_penalty),
                                                        "wide_penalty": float(wide_penalty),
                                                        "timeout_penalty": float(timeout_penalty),
                                                        "mae_keep_frac": float(mae_keep_frac),
                                                        "wide_keep_frac": float(wide_keep_frac),
                                                        "timeout_keep_frac": float(timeout_keep_frac),
                                                        "gate_rows": int(row.get("gate_rows", 0)),
                                                        "gate_candidate_rate": float(row.get("gate_candidate_rate", 0.0)),
                                                    }
                                                )
                                                selected_ledger_rows.append(ledger_row)
                                        for weekly_row in _selection_weekly_rows(
                                            frame=valid,
                                            metrics=valid_metrics,
                                            target=target_valid,
                                            score=score,
                                            gate_mask=gate_mask,
                                            arm=arm,
                                            selector="dual_target_execution_risk_gate_seed_ensemble_smoke_oos",
                                            period=month,
                                            top_frac=top_frac,
                                            selection_frac=row.get("selection_frac_within_gate", top_frac),
                                        ):
                                            weekly_row.update(
                                                {
                                                    "label_arm": label_arm,
                                                    "weight_arm": weight_arm,
                                                    "model_feature_selector": model_feature_selector,
                                                    "model_feature_count": int(len(model_features)),
                                                    "model_features": ",".join(model_features),
                                                    "selection_mode": "risk_gate",
                                                    "mae_penalty": float(mae_penalty),
                                                    "wide_penalty": float(wide_penalty),
                                                    "timeout_penalty": float(timeout_penalty),
                                                    "mae_keep_frac": float(mae_keep_frac),
                                                    "wide_keep_frac": float(wide_keep_frac),
                                                    "timeout_keep_frac": float(timeout_keep_frac),
                                                    "selection_frac_within_gate": float(
                                                        row.get("selection_frac_within_gate", top_frac)
                                                    ),
                                                    "gate_rows": int(row.get("gate_rows", 0)),
                                                    "gate_candidate_rate": float(row.get("gate_candidate_rate", 0.0)),
                                                }
                                            )
                                            weekly_rows.append(weekly_row)
    return rows, weekly_rows, selected_ledger_rows, diagnostics


def _aggregate(monthly: pd.DataFrame) -> pd.DataFrame:
    if monthly.empty:
        return monthly
    rows: list[dict[str, Any]] = []
    groups = monthly.groupby(
        [
            "arm",
            "label_arm",
            "weight_arm",
            "model_feature_selector",
            "selection_mode",
            "mae_penalty",
            "wide_penalty",
            "timeout_penalty",
            "mae_keep_frac",
            "wide_keep_frac",
            "timeout_keep_frac",
            "top_frac",
        ],
        dropna=False,
        observed=True,
    )
    for key, group in groups:
        (
            arm,
            label_arm,
            weight_arm,
            model_feature_selector,
            selection_mode,
            mae_penalty,
            wide_penalty,
            timeout_penalty,
            mae_keep_frac,
            wide_keep_frac,
            timeout_keep_frac,
            top_frac,
        ) = key
        mean_u = pd.to_numeric(group["mean_u"], errors="coerce")
        mean_return_net = pd.to_numeric(group["mean_return_net"], errors="coerce")
        selected_rows = pd.to_numeric(group["selected_rows"], errors="coerce")
        worst_month = float(mean_u.min()) if len(mean_u.dropna()) else float("nan")
        worst_month_return = (
            float(mean_return_net.min()) if len(mean_return_net.dropna()) else float("nan")
        )
        sum_return_net = float((mean_return_net.fillna(0.0) * selected_rows.fillna(0.0)).sum())
        sum_return_net_plus10 = float(
            ((mean_return_net.fillna(0.0) - 0.0010) * selected_rows.fillna(0.0)).sum()
        )
        sum_return_net_plus25 = float(
            ((mean_return_net.fillna(0.0) - 0.0025) * selected_rows.fillna(0.0)).sum()
        )
        avg_bad_mae = _safe_mean(group["bad_mae_1r_rate"])
        avg_p90_mae = _safe_mean(group["p90_mae_norm"])
        avg_wide25 = _safe_mean(group["wide_barrier_25bps_rate"])
        avg_timeout = _safe_mean(group["timeout_rate"])
        avg_clean = _safe_mean(group["clean_row_rate"]) if "clean_row_rate" in group else float("nan")
        avg_strict_clean = (
            _safe_mean(group["strict_clean_row_rate"]) if "strict_clean_row_rate" in group else float("nan")
        )
        avg_bounded = _safe_mean(group["bounded_row_rate"]) if "bounded_row_rate" in group else float("nan")
        avg_score_ic_u = _safe_mean(group["score_ic_u"])
        rows.append(
            {
                "arm": arm,
                "label_arm": label_arm,
                "weight_arm": weight_arm,
                "model_feature_selector": model_feature_selector,
                "selection_mode": selection_mode,
                "mae_penalty": float(mae_penalty),
                "wide_penalty": float(wide_penalty),
                "timeout_penalty": float(timeout_penalty),
                "mae_keep_frac": float(mae_keep_frac),
                "wide_keep_frac": float(wide_keep_frac),
                "timeout_keep_frac": float(timeout_keep_frac),
                "top_frac": float(top_frac),
                "months": int(group["period"].nunique()),
                "positive_months": int((mean_u > 0.0).sum()),
                "positive_return_months": int((mean_return_net > 0.0).sum()),
                "positive_return_month_rate": _safe_mean(mean_return_net > 0.0),
                "mean_u": _safe_mean(mean_u),
                "worst_month_mean_u": worst_month,
                "mean_return_net": _safe_mean(mean_return_net),
                "worst_month_return_net": worst_month_return,
                "q25_month_return_net": _safe_quantile(mean_return_net, 0.25),
                "sum_return_net": sum_return_net,
                "sum_return_net_plus10bps": sum_return_net_plus10,
                "sum_return_net_plus25bps": sum_return_net_plus25,
                "hit_u": _safe_mean(group["hit_u"]),
                "hit_return_net": _safe_mean(group["hit_return_net"]),
                "q10_u": _safe_mean(group["q10_u"]),
                "delta_mean_u_vs_period": _safe_mean(group["delta_mean_u_vs_period"]),
                "score_ic_u": avg_score_ic_u,
                "score_ic_label": _safe_mean(group["score_ic_label"]),
                "upside_ic_u": _safe_mean(group["upside_ic_u"]),
                "bad_mae_ic_actual": _safe_mean(group["bad_mae_ic_actual"]),
                "wide_ic_actual": _safe_mean(group["wide_ic_actual"]),
                "timeout_ic_actual": _safe_mean(group["timeout_ic_actual"]),
                "decile_spearman_u": _safe_mean(group["decile_spearman_u"]),
                "top_bottom_decile_spread_u": _safe_mean(group["top_bottom_decile_spread_u"]),
                "bad_mae_1r_rate": avg_bad_mae,
                "p90_mae_norm": avg_p90_mae,
                "mean_mfe_mae_ratio": _safe_mean(group["mean_mfe_mae_ratio"])
                if "mean_mfe_mae_ratio" in group
                else float("nan"),
                "clean_row_rate": avg_clean,
                "strict_clean_row_rate": avg_strict_clean,
                "bounded_row_rate": avg_bounded,
                "wide_barrier_25bps_rate": avg_wide25,
                "wide_barrier_35bps_rate": _safe_mean(group["wide_barrier_35bps_rate"]),
                "timeout_rate": avg_timeout,
                "gate_candidate_rate": _safe_mean(group["gate_candidate_rate"]),
                "selection_frac_within_gate": _safe_mean(group["selection_frac_within_gate"]),
                "top_symbol_share": _safe_mean(group["top_symbol_share"]),
                "mean_selected_rows": _safe_mean(selected_rows),
                "min_selected_rows": int(selected_rows.min()) if len(selected_rows.dropna()) else 0,
                "mean_model_feature_count": _safe_mean(group["model_feature_count"]),
                "model_features": str(group["model_features"].dropna().iloc[0])
                if group["model_features"].dropna().size
                else "",
                "decision": (
                    "promote_to_locked_walkforward_candidate"
                    if int((mean_u > 0.0).sum()) >= 3
                    and _safe_mean(mean_u) > 0.0
                    and math.isfinite(worst_month)
                    and worst_month > 0.0
                    and int((mean_return_net > 0.0).sum()) >= 3
                    and _safe_mean(mean_return_net) > 0.0
                    and math.isfinite(worst_month_return)
                    and worst_month_return > 0.0
                    and sum_return_net_plus10 > 0.0
                    and avg_score_ic_u > 0.0
                    and avg_wide25 <= 0.05
                    and avg_bad_mae <= 0.40
                    and avg_p90_mae <= 4.00
                    else "reject_or_rework"
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["top_frac", "mean_u", "worst_month_mean_u"],
        ascending=[True, False, False],
    )


def _write_markdown(output_dir: Path, aggregate: pd.DataFrame, manifest: dict[str, Any]) -> Path:
    path = output_dir / "label_dual_target_execution_smoke.md"

    def table(frame: pd.DataFrame, cols: list[str], limit: int | None = None) -> str:
        if frame.empty:
            return "No rows."
        view = frame[[c for c in cols if c in frame.columns]].copy()
        if limit is not None:
            view = view.head(limit)
        for col in view.columns:
            if pd.api.types.is_float_dtype(view[col]):
                view[col] = view[col].map(lambda v: f"{float(v):.4f}" if pd.notna(v) else "")
        return view.to_markdown(index=False)

    cols = [
        "decision",
        "label_arm",
        "weight_arm",
        "selection_mode",
        "mae_penalty",
        "wide_penalty",
        "timeout_penalty",
        "mae_keep_frac",
        "wide_keep_frac",
        "timeout_keep_frac",
        "months",
        "positive_months",
        "mean_u",
        "worst_month_mean_u",
        "positive_return_months",
        "mean_return_net",
        "worst_month_return_net",
        "sum_return_net_plus10bps",
        "q10_u",
        "bad_mae_1r_rate",
        "p90_mae_norm",
        "mean_mfe_mae_ratio",
        "clean_row_rate",
        "strict_clean_row_rate",
        "bounded_row_rate",
        "wide_barrier_25bps_rate",
        "timeout_rate",
        "gate_candidate_rate",
        "score_ic_u",
        "bad_mae_ic_actual",
        "wide_ic_actual",
        "mean_selected_rows",
    ]
    lines = [
        "# Label Dual-Target Execution Smoke",
        "",
        "Scope: cheap causal month-forward diagnostic; not production LightGBM training or final OOS.",
        "",
    ]
    for frac in manifest["top_fracs"]:
        subset = aggregate[aggregate["top_frac"].eq(frac)].sort_values(
            ["mean_u", "worst_month_mean_u"],
            ascending=[False, False],
        )
        lines.extend([f"## Top {frac:.0%}", "", table(subset, cols, limit=30), ""])
    lines.extend(
        [
            "## Outputs",
            "",
            f"- Monthly: `{manifest['outputs']['monthly']}`",
            f"- Weekly: `{manifest['outputs']['weekly']}`",
            f"- Selected ledger: `{manifest['outputs']['selected_ledger']}`",
            f"- Aggregate: `{manifest['outputs']['aggregate']}`",
            f"- Diagnostics: `{manifest['outputs']['diagnostics']}`",
            f"- Manifest: `{manifest['outputs']['manifest']}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_smoke(
    *,
    labels_path: Path,
    output_dir: Path,
    feature_dir: Path,
    feature_list_csv: Path,
    max_feature_store_features: int | None,
    label_arms: list[str],
    weight_arms: list[str],
    seeds: list[int],
    model_feature_selector: str,
    model_feature_tail_frac: float,
    mae_penalties: list[float],
    wide_penalties: list[float],
    timeout_penalties: list[float],
    mae_keep_fracs: list[float],
    wide_keep_fracs: list[float],
    timeout_keep_fracs: list[float],
    top_fracs: list[float],
    emit_selected_ledger: bool,
    include_causal_outcome_priors: bool,
    include_causal_state_path_priors: bool,
    include_event_confirmation_features: bool,
    prior_windows_days: list[float],
    prior_embargo_hours: float,
    state_path_prior_features: list[str],
    event_feature_store_features: list[str],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = _load_labels(labels_path)
    selected_features = _read_feature_list(feature_list_csv, max_features=max_feature_store_features)
    selected_features = list(dict.fromkeys(list(selected_features) + list(event_feature_store_features)))
    feature_matrix, feature_store_report = _load_feature_store_columns(
        frame,
        feature_dir=feature_dir,
        selected_features=selected_features,
    )
    if not feature_matrix.empty:
        new_cols = [col for col in feature_matrix.columns if col not in frame.columns]
        if new_cols:
            frame = pd.concat(
                [
                    frame.reset_index(drop=True),
                    feature_matrix.loc[:, new_cols].astype(np.float32, copy=False).reset_index(drop=True),
                ],
                axis=1,
            ).copy()

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
            new_event_cols = [col for col in event_features.columns if col not in frame.columns]
            if new_event_cols:
                frame = pd.concat(
                    [frame, event_features.loc[:, new_event_cols].astype(np.float32, copy=False)],
                    axis=1,
                ).copy()

    features = _feature_columns(frame)
    targets = _label_targets(frame, metrics)
    if not label_arms:
        label_arms = list(LABEL_ARMS)
    missing_labels = sorted(set(label_arms) - set(targets))
    missing_weights = sorted(set(weight_arms) - set(WEIGHT_ARMS))
    if model_feature_selector not in {"all", *PROXY_METHODS}:
        raise ValueError(f"Unknown model feature selector: {model_feature_selector}")
    if missing_labels:
        raise ValueError(f"Unknown label arms: {missing_labels}")
    if missing_weights:
        raise ValueError(f"Unknown weight arms: {missing_weights}")

    monthly_rows: list[dict[str, Any]] = []
    weekly_rows: list[dict[str, Any]] = []
    selected_ledger_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    months = sorted(frame["__ts__"].dt.to_period("M").dropna().astype(str).unique())
    for month in months[1:]:
        rows, weekly, selected_ledger, diagnostics = _run_month(
            frame=frame,
            metrics=metrics,
            targets=targets,
            features=features,
            month=month,
            label_arms=label_arms,
            weight_arms=weight_arms,
            seeds=seeds,
            model_feature_selector=model_feature_selector,
            model_feature_tail_frac=model_feature_tail_frac,
            mae_penalties=mae_penalties,
            wide_penalties=wide_penalties,
            timeout_penalties=timeout_penalties,
            mae_keep_fracs=mae_keep_fracs,
            wide_keep_fracs=wide_keep_fracs,
            timeout_keep_fracs=timeout_keep_fracs,
            top_fracs=top_fracs,
            emit_selected_ledger=emit_selected_ledger,
        )
        monthly_rows.extend(rows)
        weekly_rows.extend(weekly)
        selected_ledger_rows.extend(selected_ledger)
        diagnostic_rows.extend(diagnostics)

    monthly = pd.DataFrame(monthly_rows)
    weekly = pd.DataFrame(weekly_rows)
    selected_ledger = pd.DataFrame(selected_ledger_rows)
    diagnostics = pd.DataFrame(diagnostic_rows)
    aggregate = _aggregate(monthly)
    paths = {
        "monthly": output_dir / "label_dual_target_execution_smoke_monthly.csv",
        "weekly": output_dir / "label_dual_target_execution_smoke_weekly.csv",
        "selected_ledger": output_dir / "label_dual_target_execution_smoke_selected_ledger.csv",
        "aggregate": output_dir / "label_dual_target_execution_smoke_aggregate.csv",
        "diagnostics": output_dir / "label_dual_target_execution_smoke_diagnostics.csv",
        "manifest": output_dir / "manifest.json",
    }
    monthly.to_csv(paths["monthly"], index=False)
    weekly.to_csv(paths["weekly"], index=False)
    selected_ledger.to_csv(paths["selected_ledger"], index=False)
    aggregate.to_csv(paths["aggregate"], index=False)
    diagnostics.to_csv(paths["diagnostics"], index=False)
    manifest = {
        "scope": "dual_target_execution_risk_smoke_not_full_policy_training",
        "labels_path": str(labels_path),
        "output_dir": str(output_dir),
        "rows": int(len(frame)),
        "timestamp_min": frame["__ts__"].min(),
        "timestamp_max": frame["__ts__"].max(),
        "symbols": int(frame["__symbol__"].nunique(dropna=True)),
        "feature_count": int(len(features)),
        "label_arms": label_arms,
        "weight_arms": weight_arms,
        "top_fracs": [float(v) for v in top_fracs],
        "emit_selected_ledger": bool(emit_selected_ledger),
        "selected_ledger_rows": int(len(selected_ledger)),
        "feature_store": feature_store_report,
        "include_causal_outcome_priors": bool(include_causal_outcome_priors),
        "include_causal_state_path_priors": bool(include_causal_state_path_priors),
        "include_event_confirmation_features": bool(include_event_confirmation_features),
        "prior_windows_days": [float(v) for v in prior_windows_days],
        "prior_embargo_hours": float(prior_embargo_hours),
        "causal_outcome_priors": causal_outcome_report,
        "causal_state_path_priors": causal_state_report,
        "event_confirmation_features": event_confirmation_report,
        "feature_dir": str(feature_dir),
        "feature_list_csv": str(feature_list_csv),
        "max_feature_store_features": max_feature_store_features,
        "model": {
            "type": "ExtraTreesRegressor",
            "n_estimators": 96,
            "max_depth": 8,
            "min_samples_leaf": 40,
            "max_features": "sqrt",
            "seeds": [int(seed) for seed in seeds],
            "feature_selector": model_feature_selector,
            "feature_selector_tail_frac": float(model_feature_tail_frac),
        },
        "penalties": {
            "mae": [float(v) for v in mae_penalties],
            "wide": [float(v) for v in wide_penalties],
            "timeout": [float(v) for v in timeout_penalties],
        },
        "risk_gates": {
            "mae_keep_frac": [float(v) for v in mae_keep_fracs],
            "wide_keep_frac": [float(v) for v in wide_keep_fracs],
            "timeout_keep_frac": [float(v) for v in timeout_keep_fracs],
        },
        "outputs": {key: str(value) for key, value in paths.items()},
    }
    paths["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    markdown = _write_markdown(output_dir, aggregate, manifest)
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
    parser.add_argument("--label-arms", default=",".join(DEFAULT_LABEL_ARMS))
    parser.add_argument("--weight-arms", default=",".join(DEFAULT_WEIGHT_ARMS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--model-feature-selector", choices=("all", *PROXY_METHODS), default="stable_tail_recovery")
    parser.add_argument("--model-feature-tail-frac", type=float, default=0.01)
    parser.add_argument("--mae-penalties", default=",".join(str(v) for v in DEFAULT_MAE_PENALTIES))
    parser.add_argument("--wide-penalties", default=",".join(str(v) for v in DEFAULT_WIDE_PENALTIES))
    parser.add_argument("--timeout-penalties", default=",".join(str(v) for v in DEFAULT_TIMEOUT_PENALTIES))
    parser.add_argument("--mae-keep-fracs", default=",".join(str(v) for v in DEFAULT_MAE_KEEP_FRACS))
    parser.add_argument("--wide-keep-fracs", default=",".join(str(v) for v in DEFAULT_WIDE_KEEP_FRACS))
    parser.add_argument("--timeout-keep-fracs", default=",".join(str(v) for v in DEFAULT_TIMEOUT_KEEP_FRACS))
    parser.add_argument(
        "--top-fracs",
        default=",".join(str(v) for v in DEFAULT_TOP_FRACS),
        help="Comma-separated selection fractions to evaluate.",
    )
    parser.add_argument(
        "--emit-selected-ledger",
        action="store_true",
        help="Write a row-level ledger for selected candidates. Useful for focused diagnostics; can be large on broad grids.",
    )
    parser.add_argument("--include-causal-outcome-priors", action="store_true")
    parser.add_argument("--include-causal-state-path-priors", action="store_true")
    parser.add_argument("--include-event-confirmation-features", action="store_true")
    parser.add_argument(
        "--prior-windows-days",
        default=",".join(str(v) for v in DEFAULT_PRIOR_WINDOWS_DAYS),
    )
    parser.add_argument("--prior-embargo-hours", type=float, default=24.0)
    parser.add_argument(
        "--state-path-prior-features",
        default=",".join(DEFAULT_STATE_PATH_PRIOR_FEATURES),
    )
    parser.add_argument(
        "--event-feature-store-features",
        default=",".join(DEFAULT_EVENT_FEATURE_STORE_FEATURES),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = run_smoke(
        labels_path=args.labels_path,
        output_dir=args.output_dir,
        feature_dir=args.feature_dir,
        feature_list_csv=args.feature_list_csv,
        max_feature_store_features=args.max_feature_store_features,
        label_arms=_parse_csv(args.label_arms, DEFAULT_LABEL_ARMS),
        weight_arms=_parse_csv(args.weight_arms, DEFAULT_WEIGHT_ARMS),
        seeds=_parse_int_csv(args.seeds, (args.seed,)),
        model_feature_selector=str(args.model_feature_selector),
        model_feature_tail_frac=float(args.model_feature_tail_frac),
        mae_penalties=_parse_float_csv(args.mae_penalties, DEFAULT_MAE_PENALTIES),
        wide_penalties=_parse_float_csv(args.wide_penalties, DEFAULT_WIDE_PENALTIES),
        timeout_penalties=_parse_float_csv(args.timeout_penalties, DEFAULT_TIMEOUT_PENALTIES),
        mae_keep_fracs=_parse_float_csv(args.mae_keep_fracs, DEFAULT_MAE_KEEP_FRACS),
        wide_keep_fracs=_parse_float_csv(args.wide_keep_fracs, DEFAULT_WIDE_KEEP_FRACS),
        timeout_keep_fracs=_parse_float_csv(args.timeout_keep_fracs, DEFAULT_TIMEOUT_KEEP_FRACS),
        top_fracs=_parse_float_csv(args.top_fracs, DEFAULT_TOP_FRACS),
        emit_selected_ledger=bool(args.emit_selected_ledger),
        include_causal_outcome_priors=bool(args.include_causal_outcome_priors),
        include_causal_state_path_priors=bool(args.include_causal_state_path_priors),
        include_event_confirmation_features=bool(args.include_event_confirmation_features),
        prior_windows_days=_parse_float_csv(args.prior_windows_days, DEFAULT_PRIOR_WINDOWS_DAYS),
        prior_embargo_hours=float(args.prior_embargo_hours),
        state_path_prior_features=_parse_csv(
            args.state_path_prior_features,
            DEFAULT_STATE_PATH_PRIOR_FEATURES,
        ),
        event_feature_store_features=_parse_csv(
            args.event_feature_store_features,
            DEFAULT_EVENT_FEATURE_STORE_FEATURES,
        ),
    )
    print(json.dumps(_json_safe(manifest), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
