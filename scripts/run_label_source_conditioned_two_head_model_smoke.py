#!/usr/bin/env python3
"""Frozen source-conditioned two-head model smoke before full training.

This is a cheap month-forward model screen for the Stage 11 proxy candidate.
It trains two small fixed-parameter regressors on prior months only:

1. bad first-touch execution;
2. margin utility.

The raw-score recipe is deliberately separated from portable coverage gates:
raw thresholds are useful diagnostics, while coverage gates are calibrated from
prior-month model-score distributions.
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
from sklearn.ensemble import ExtraTreesRegressor


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
    _read_feature_list,
    _safe_mean,
    _safe_numeric,
    _safe_quantile,
    _spearman,
)
from scripts.run_label_first_touch_soft_recipe_proxy_ablation import DEFAULT_TOP_KS  # noqa: E402
from scripts.run_label_weighted_proxy_ablation import (  # noqa: E402
    WEIGHT_ARMS,
    _effective_sample_size,
    _weight_series,
)
from scripts.run_label_source_conditioned_two_head_proxy import (  # noqa: E402
    _select_by_source_fit,
    _source_fit_holdout_summary,
)
from scripts.run_label_two_head_abstention_utility_proxy import (  # noqa: E402
    TwoHeadSpec,
    _global_bad_soft,
    _monthly_weekly_rows,
    _target_for_selection,
    _utility_targets as _base_utility_targets,
)
from scripts.run_soft_label_candidate_source_ablation import (  # noqa: E402
    _build_sources,
    _source_context,
    _source_summary,
)
from scripts.run_soft_label_economic_proxy_ablation import (  # noqa: E402
    DEFAULT_EVENT_FEATURE_STORE_FEATURES,
    _event_confirmation_features,
)


DEFAULT_OUTPUT_DIR = Path("data_perp/reports/label_source_conditioned_two_head_model_smoke_v1")
DEFAULT_SOURCE = "quiet_mid"
DEFAULT_UTILITY_TARGET = "margin_utility"
DEFAULT_SCORE_RULES = ("utility_minus_bad025",)
DEFAULT_BAD_THRESHOLDS = (0.20,)
DEFAULT_BAD_COVERAGES = (0.02, 0.03, 0.05, 0.08, 0.10)
DEFAULT_TOP_KS = (20,)
DEFAULT_SEEDS = (17, 29)
DEFAULT_MODEL_KIND = "extratrees"
DEFAULT_SELECTION_POLICY = "legacy"
DEFAULT_WEIGHT_ARMS = ("none",)


def _fit_predict_extra_trees(
    *,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_valid: pd.DataFrame,
    sample_weight: pd.Series | None,
    seeds: list[int],
    n_estimators: int,
    max_depth: int | None,
    min_samples_leaf: int,
) -> tuple[pd.Series, pd.Series, dict[str, Any]]:
    preds: list[np.ndarray] = []
    train_preds: list[np.ndarray] = []
    weight_values = (
        _safe_numeric(sample_weight).reindex(x_train.index).fillna(1.0).to_numpy(dtype=np.float32)
        if sample_weight is not None
        else None
    )
    for seed in seeds:
        model = ExtraTreesRegressor(
            n_estimators=int(n_estimators),
            max_depth=max_depth,
            min_samples_leaf=int(min_samples_leaf),
            max_features="sqrt",
            random_state=int(seed),
            n_jobs=2,
        )
        model.fit(
            x_train,
            _safe_numeric(y_train).fillna(0.0).to_numpy(dtype=np.float32),
            sample_weight=weight_values,
        )
        train_preds.append(model.predict(x_train).astype(np.float32))
        preds.append(model.predict(x_valid).astype(np.float32))
    pred_matrix = np.vstack(preds) if preds else np.empty((0, len(x_valid)), dtype=np.float32)
    train_pred_matrix = np.vstack(train_preds) if train_preds else np.empty((0, len(x_train)), dtype=np.float32)
    pred = np.mean(pred_matrix, axis=0).astype(np.float32) if len(pred_matrix) else np.full(len(x_valid), np.nan)
    train_pred = (
        np.mean(train_pred_matrix, axis=0).astype(np.float32)
        if len(train_pred_matrix)
        else np.full(len(x_train), np.nan)
    )
    return pd.Series(pred).clip(0.0, 1.0), pd.Series(train_pred).clip(0.0, 1.0), {
        "seed_count": int(len(seeds)),
        "seed_std_mean": float(np.mean(np.std(pred_matrix, axis=0))) if len(pred_matrix) > 1 else 0.0,
        "pred_mean": _safe_mean(pred),
        "pred_p10": _safe_quantile(pred, 0.10),
        "pred_p90": _safe_quantile(pred, 0.90),
    }


def _fit_predict_lgbm(
    *,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_valid: pd.DataFrame,
    sample_weight: pd.Series | None,
    seeds: list[int],
    n_estimators: int,
    max_depth: int | None,
    learning_rate: float,
    num_leaves: int,
    min_child_samples: int,
) -> tuple[pd.Series, pd.Series, dict[str, Any]]:
    try:
        import lightgbm as lgb
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise RuntimeError("lightgbm is required for --model-kind lgbm") from exc

    preds: list[np.ndarray] = []
    train_preds: list[np.ndarray] = []
    weight_values = (
        _safe_numeric(sample_weight).reindex(x_train.index).fillna(1.0).to_numpy(dtype=np.float32)
        if sample_weight is not None
        else None
    )
    for seed in seeds:
        model = lgb.LGBMRegressor(
            objective="regression",
            metric="rmse",
            n_estimators=int(n_estimators),
            learning_rate=float(learning_rate),
            num_leaves=int(num_leaves),
            max_depth=-1 if max_depth is None else int(max_depth),
            min_child_samples=int(min_child_samples),
            subsample=0.80,
            subsample_freq=1,
            colsample_bytree=0.80,
            reg_alpha=0.10,
            reg_lambda=1.00,
            random_state=int(seed),
            n_jobs=2,
            verbosity=-1,
            force_col_wise=True,
        )
        model.fit(
            x_train,
            _safe_numeric(y_train).fillna(0.0).to_numpy(dtype=np.float32),
            sample_weight=weight_values,
        )
        train_preds.append(np.asarray(model.predict(x_train), dtype=np.float32))
        preds.append(np.asarray(model.predict(x_valid), dtype=np.float32))
    pred_matrix = np.vstack(preds) if preds else np.empty((0, len(x_valid)), dtype=np.float32)
    train_pred_matrix = np.vstack(train_preds) if train_preds else np.empty((0, len(x_train)), dtype=np.float32)
    pred = np.mean(pred_matrix, axis=0).astype(np.float32) if len(pred_matrix) else np.full(len(x_valid), np.nan)
    train_pred = (
        np.mean(train_pred_matrix, axis=0).astype(np.float32)
        if len(train_pred_matrix)
        else np.full(len(x_train), np.nan)
    )
    return pd.Series(pred).clip(0.0, 1.0), pd.Series(train_pred).clip(0.0, 1.0), {
        "seed_count": int(len(seeds)),
        "seed_std_mean": float(np.mean(np.std(pred_matrix, axis=0))) if len(pred_matrix) > 1 else 0.0,
        "pred_mean": _safe_mean(pred),
        "pred_p10": _safe_quantile(pred, 0.10),
        "pred_p90": _safe_quantile(pred, 0.90),
    }


def _fit_predict_model(
    *,
    model_kind: str,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_valid: pd.DataFrame,
    sample_weight: pd.Series | None,
    seeds: list[int],
    n_estimators: int,
    max_depth: int | None,
    min_samples_leaf: int,
    learning_rate: float,
    num_leaves: int,
    min_child_samples: int,
) -> tuple[pd.Series, pd.Series, dict[str, Any]]:
    if model_kind == "extratrees":
        pred, train_pred, diag = _fit_predict_extra_trees(
            x_train=x_train,
            y_train=y_train,
            x_valid=x_valid,
            sample_weight=sample_weight,
            seeds=seeds,
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
        )
        diag["model"] = "ExtraTreesRegressor"
        return pred, train_pred, diag
    if model_kind == "lgbm":
        pred, train_pred, diag = _fit_predict_lgbm(
            x_train=x_train,
            y_train=y_train,
            x_valid=x_valid,
            sample_weight=sample_weight,
            seeds=seeds,
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            min_child_samples=min_child_samples,
        )
        diag["model"] = "LGBMRegressor"
        return pred, train_pred, diag
    raise ValueError(f"Unknown model kind: {model_kind}")


def _bad_gate_threshold(
    *,
    mode: str,
    gate_value: float,
    bad_pred: pd.Series,
    train_bad_pred: pd.Series,
) -> float:
    if mode == "raw":
        return float(gate_value)
    if mode == "train_coverage":
        return _safe_quantile(train_bad_pred, float(gate_value))
    if mode == "valid_coverage":
        return _safe_quantile(bad_pred, float(gate_value))
    raise ValueError(f"Unknown bad gate mode: {mode}")


def _score_from_rule(
    *,
    score_rule: str,
    utility_pred: pd.Series,
    bad_pred: pd.Series,
    applied_bad_threshold: float,
) -> pd.Series:
    utility = _safe_numeric(utility_pred)
    bad = _safe_numeric(bad_pred)
    if score_rule == "utility":
        score = utility
    elif score_rule == "utility_minus_bad025":
        score = utility - 0.25 * bad
    elif score_rule == "utility_minus_bad050":
        score = utility - 0.50 * bad
    else:
        raise ValueError(f"unknown score rule: {score_rule}")
    return score.where(bad <= float(applied_bad_threshold))


def _month_design(
    *,
    frame: pd.DataFrame,
    features: list[str],
    train_mask: pd.Series,
    valid_mask: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    x_train = frame.loc[train_mask, features].copy()
    x_valid = frame.loc[valid_mask, features].copy()
    x_train = x_train.replace([np.inf, -np.inf], np.nan)
    x_valid = x_valid.replace([np.inf, -np.inf], np.nan)
    med = x_train.median(numeric_only=True)
    x_train = x_train.fillna(med).fillna(0.0).astype(np.float32, copy=False)
    x_valid = x_valid.fillna(med).fillna(0.0).astype(np.float32, copy=False)
    return x_train, x_valid


def _format_table(frame: pd.DataFrame, cols: list[str], limit: int = 80) -> str:
    if frame.empty:
        return "No rows."
    view = frame[[col for col in cols if col in frame.columns]].head(limit).copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda value: f"{float(value):.4f}" if pd.notna(value) else "")
    return view.to_markdown(index=False)


def _optional_float(value: str | None) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "" or text.lower() in {"none", "nan", "null"}:
        return None
    return float(text)


def _optional_int(value: str | None) -> int | None:
    parsed = _optional_float(value)
    if parsed is None:
        return None
    return int(parsed)


def _weights_for_arm(
    weight_arm: str,
    *,
    train_frame: pd.DataFrame,
    train_metrics: pd.DataFrame,
    train_target: pd.DataFrame,
) -> tuple[pd.Series | None, dict[str, Any]]:
    if weight_arm == "none":
        return None, {
            "weight_arm": "none",
            "weight_mean": 1.0,
            "weight_p90": 1.0,
            "weight_p99": 1.0,
            "weight_effective_n": float(len(train_frame)),
            "weight_effective_frac": 1.0 if len(train_frame) else float("nan"),
        }
    weights = _weight_series(
        frame=train_frame,
        metrics=train_metrics,
        target=train_target,
        arm=weight_arm,
    )
    effective_n = _effective_sample_size(weights)
    return weights, {
        "weight_arm": str(weight_arm),
        "weight_mean": _safe_mean(weights),
        "weight_p90": _safe_quantile(weights, 0.90),
        "weight_p99": _safe_quantile(weights, 0.99),
        "weight_effective_n": float(effective_n),
        "weight_effective_frac": float(effective_n / len(weights)) if len(weights) else float("nan"),
    }


def _sigmoid_series(raw: pd.Series | np.ndarray) -> pd.Series:
    index = raw.index if isinstance(raw, pd.Series) else None
    arr = np.asarray(raw, dtype=np.float64)
    arr = np.clip(arr, -60.0, 60.0)
    return pd.Series(1.0 / (1.0 + np.exp(-arr)), index=index)


def _utility_targets(frame: pd.DataFrame, ft: pd.DataFrame) -> dict[str, pd.Series]:
    targets = dict(_base_utility_targets(frame, ft))
    ret_net = _safe_numeric(ft.get("ret_net", pd.Series(np.nan, index=ft.index))).fillna(-0.02)
    clean = _safe_numeric(ft.get("clean_exec_actual", pd.Series(0.0, index=ft.index))).fillna(0.0).clip(0.0, 1.0)
    hit = _safe_numeric(ft.get("first_touch_hit", pd.Series(0.0, index=ft.index))).fillna(0.0).clip(0.0, 1.0)
    bad_mae = _safe_numeric(ft.get("first_touch_mae_to_sl", pd.Series(10.0, index=ft.index))).fillna(10.0)
    clean_soft = (
        0.40
        + 0.35 * clean
        + 0.15 * hit
        + 0.10 * _sigmoid_series((1.25 - bad_mae) / 0.30).reindex(ft.index, fill_value=0.0)
    ).clip(0.0, 1.0)
    ret0 = _sigmoid_series((ret_net - 0.0) / 0.006).reindex(ft.index, fill_value=0.0).clip(0.0, 1.0)
    ret10 = _sigmoid_series((ret_net - 0.0010) / 0.006).reindex(ft.index, fill_value=0.0).clip(0.0, 1.0)
    ret25 = _sigmoid_series((ret_net - 0.0025) / 0.006).reindex(ft.index, fill_value=0.0).clip(0.0, 1.0)
    targets.update(
        {
            "ret_net": ret0,
            "ret_net_margin10": ret10,
            "ret_net_margin25": ret25,
            "ret_net_clean": (ret0 * clean_soft).clip(0.0, 1.0),
            "ret_net_margin10_clean": (ret10 * clean_soft).clip(0.0, 1.0),
            "ret_net_margin25_clean": (ret25 * clean_soft).clip(0.0, 1.0),
        }
    )
    return targets


def _selection_floor_mask(
    frame: pd.DataFrame,
    *,
    require_fit_bounded: bool,
    fit_min_mean_month_u: float | None,
    fit_min_worst_month_u: float | None,
    fit_min_q25_week_u: float | None,
    fit_min_mean_month_return_net: float | None,
    fit_min_worst_month_return_net: float | None,
    fit_min_q25_week_return_net: float | None,
    fit_min_sum_return_net: float | None,
    fit_min_material_positive_return_week_rate: float | None,
    fit_min_selected_rows: int | None,
    fit_min_candidate_timestamp_coverage: float | None,
    fit_min_material_positive_week_rate: float | None,
    fit_min_clean_exec: float | None,
    fit_max_bad_mae: float | None,
    fit_max_p90_mae: float | None,
) -> pd.Series:
    mask = pd.Series(True, index=frame.index)
    if require_fit_bounded and "fit_bounded_pass" in frame.columns:
        mask &= frame["fit_bounded_pass"].astype(bool)
    checks = [
        ("fit_mean_month_u", fit_min_mean_month_u, "min"),
        ("fit_worst_month_u", fit_min_worst_month_u, "min"),
        ("fit_q25_week_u", fit_min_q25_week_u, "min"),
        ("fit_mean_month_return_net", fit_min_mean_month_return_net, "min"),
        ("fit_worst_month_return_net", fit_min_worst_month_return_net, "min"),
        ("fit_q25_week_return_net", fit_min_q25_week_return_net, "min"),
        ("fit_sum_return_net", fit_min_sum_return_net, "min"),
        (
            "fit_material_positive_return_week_rate",
            fit_min_material_positive_return_week_rate,
            "min",
        ),
        ("fit_selected_rows", fit_min_selected_rows, "min"),
        ("fit_candidate_timestamp_coverage", fit_min_candidate_timestamp_coverage, "min"),
        ("fit_material_positive_week_rate", fit_min_material_positive_week_rate, "min"),
        ("fit_clean_exec_actual_rate", fit_min_clean_exec, "min"),
        ("fit_first_touch_bad_mae_to_sl_rate", fit_max_bad_mae, "max"),
        ("fit_p90_first_touch_mae_to_sl", fit_max_p90_mae, "max"),
    ]
    for col, threshold, direction in checks:
        if threshold is None or col not in frame.columns:
            continue
        values = _safe_numeric(frame[col])
        if direction == "min":
            mask &= values.ge(float(threshold))
        else:
            mask &= values.le(float(threshold))
    return mask.fillna(False)


def _profit_selection_objective(frame: pd.DataFrame, *, metric: str) -> pd.Series:
    mean_u = _safe_numeric(frame.get("fit_mean_month_u", pd.Series(np.nan, index=frame.index))).fillna(-1.0)
    q25_u = _safe_numeric(frame.get("fit_q25_week_u", pd.Series(np.nan, index=frame.index))).fillna(-1.0)
    worst_u = _safe_numeric(frame.get("fit_worst_month_u", pd.Series(np.nan, index=frame.index))).fillna(-1.0)
    mean_ret = _safe_numeric(frame.get("fit_mean_month_return_net", pd.Series(np.nan, index=frame.index))).fillna(-1.0)
    q25_ret = _safe_numeric(frame.get("fit_q25_week_return_net", pd.Series(np.nan, index=frame.index))).fillna(-1.0)
    worst_ret = _safe_numeric(frame.get("fit_worst_month_return_net", pd.Series(np.nan, index=frame.index))).fillna(-1.0)
    sum_ret = _safe_numeric(frame.get("fit_sum_return_net", pd.Series(0.0, index=frame.index))).fillna(0.0)
    pos_ret_week = _safe_numeric(
        frame.get("fit_material_positive_return_week_rate", pd.Series(0.0, index=frame.index))
    ).fillna(0.0)
    rows = _safe_numeric(frame.get("fit_selected_rows", pd.Series(0.0, index=frame.index))).fillna(0.0)
    coverage = _safe_numeric(
        frame.get("fit_candidate_timestamp_coverage", pd.Series(0.0, index=frame.index))
    ).fillna(0.0)
    bad_mae = _safe_numeric(
        frame.get("fit_first_touch_bad_mae_to_sl_rate", pd.Series(1.0, index=frame.index))
    ).fillna(1.0)
    p90_mae = _safe_numeric(frame.get("fit_p90_first_touch_mae_to_sl", pd.Series(10.0, index=frame.index))).fillna(10.0)
    clean = _safe_numeric(frame.get("fit_clean_exec_actual_rate", pd.Series(0.0, index=frame.index))).fillna(0.0)
    if metric == "return_net":
        base = (
            mean_ret
            + 0.75 * q25_ret
            + 0.25 * worst_ret
            + 0.0001 * sum_ret.clip(lower=-10.0, upper=10.0)
            + 0.0010 * pos_ret_week.clip(lower=0.0)
        )
    elif metric == "u":
        base = mean_u + 0.75 * q25_u + 0.25 * worst_u
    else:
        raise ValueError(f"Unknown selection objective metric: {metric}")
    return (
        base
        + 0.0002 * np.log1p(rows.clip(lower=0.0))
        + 0.0010 * coverage.clip(lower=0.0)
        + 0.0005 * clean.clip(lower=0.0)
        - 0.0010 * bad_mae.clip(lower=0.0)
        - 0.0005 * (p90_mae.clip(lower=1.0) - 1.0)
    )


def _select_by_fit_policy(
    fit_holdout: pd.DataFrame,
    *,
    selection_policy: str,
    require_fit_bounded: bool,
    fit_min_mean_month_u: float | None,
    fit_min_worst_month_u: float | None,
    fit_min_q25_week_u: float | None,
    fit_min_mean_month_return_net: float | None,
    fit_min_worst_month_return_net: float | None,
    fit_min_q25_week_return_net: float | None,
    fit_min_sum_return_net: float | None,
    fit_min_material_positive_return_week_rate: float | None,
    fit_min_selected_rows: int | None,
    fit_min_candidate_timestamp_coverage: float | None,
    fit_min_material_positive_week_rate: float | None,
    fit_min_clean_exec: float | None,
    fit_max_bad_mae: float | None,
    fit_max_p90_mae: float | None,
    selection_objective_metric: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if fit_holdout.empty:
        return fit_holdout, fit_holdout, fit_holdout
    annotated = fit_holdout.copy()
    annotated["fit_profit_floor_pass"] = _selection_floor_mask(
        annotated,
        require_fit_bounded=require_fit_bounded,
        fit_min_mean_month_u=fit_min_mean_month_u,
        fit_min_worst_month_u=fit_min_worst_month_u,
        fit_min_q25_week_u=fit_min_q25_week_u,
        fit_min_mean_month_return_net=fit_min_mean_month_return_net,
        fit_min_worst_month_return_net=fit_min_worst_month_return_net,
        fit_min_q25_week_return_net=fit_min_q25_week_return_net,
        fit_min_sum_return_net=fit_min_sum_return_net,
        fit_min_material_positive_return_week_rate=fit_min_material_positive_return_week_rate,
        fit_min_selected_rows=fit_min_selected_rows,
        fit_min_candidate_timestamp_coverage=fit_min_candidate_timestamp_coverage,
        fit_min_material_positive_week_rate=fit_min_material_positive_week_rate,
        fit_min_clean_exec=fit_min_clean_exec,
        fit_max_bad_mae=fit_max_bad_mae,
        fit_max_p90_mae=fit_max_p90_mae,
    )
    annotated["fit_profit_selection_objective"] = _profit_selection_objective(
        annotated,
        metric=selection_objective_metric,
    )
    if selection_policy == "legacy":
        selected = _select_by_source_fit(annotated)
    elif selection_policy == "fit_profit_floors":
        rows: list[pd.Series] = []
        eligible = annotated[annotated["fit_profit_floor_pass"]].copy()
        group_cols = ["source", "utility_target", "score_rule", "top_k"]
        if "weight_arm" in eligible.columns:
            group_cols.insert(1, "weight_arm")
        for _, group in eligible.groupby(group_cols, observed=True, dropna=False):
            chosen = group.sort_values(
                [
                    "fit_profit_selection_objective",
                    "fit_mean_month_u",
                    "fit_q25_week_u",
                    "fit_worst_month_u",
                    "fit_selected_rows",
                    "fit_clean_exec_actual_rate",
                    "fit_p90_first_touch_mae_to_sl",
                    "fit_first_touch_bad_mae_to_sl_rate",
                ],
                ascending=[False, False, False, False, False, False, True, True],
            ).iloc[0]
            rows.append(chosen)
        selected = pd.DataFrame(rows)
        if not selected.empty:
            selected = selected.sort_values(
                ["fit_profit_selection_objective", "fit_mean_month_u", "fit_q25_week_u"],
                ascending=[False, False, False],
            )
    else:
        raise ValueError(f"Unknown selection policy: {selection_policy}")
    best = selected.head(1).copy() if not selected.empty else selected.copy()
    return annotated, selected, best


def _write_markdown(
    *,
    output_dir: Path,
    source_summary: pd.DataFrame,
    fit_holdout: pd.DataFrame,
    selected_by_fit: pd.DataFrame,
    best_by_fit: pd.DataFrame,
    diagnostics: pd.DataFrame,
    manifest: dict[str, Any],
) -> Path:
    path = output_dir / "source_conditioned_two_head_model_smoke.md"
    fit_cols = [
        "source",
        "weight_arm",
        "utility_target",
        "selection_policy",
        "bad_gate_mode",
        "score_rule",
        "bad_threshold",
        "top_k",
        "fit_sign_pass",
        "fit_bounded_pass",
        "holdout_sign_pass",
        "holdout_bounded_standalone_pass",
        "holdout_bounded_pass",
        "fit_mean_month_u",
        "fit_material_positive_week_rate",
        "fit_q25_week_u",
        "fit_return_sign_pass",
        "fit_mean_month_return_net",
        "fit_worst_month_return_net",
        "fit_q25_week_return_net",
        "fit_sum_return_net",
        "fit_material_positive_return_week_rate",
        "fit_p90_first_touch_mae_to_sl",
        "fit_first_touch_bad_mae_to_sl_rate",
        "fit_clean_exec_actual_rate",
        "holdout_mean_month_u",
        "holdout_material_positive_week_rate",
        "holdout_q25_week_u",
        "holdout_return_sign_pass",
        "holdout_mean_month_return_net",
        "holdout_worst_month_return_net",
        "holdout_q25_week_return_net",
        "holdout_sum_return_net",
        "holdout_material_positive_return_week_rate",
        "holdout_p90_first_touch_mae_to_sl",
        "holdout_first_touch_bad_mae_to_sl_rate",
        "holdout_clean_exec_actual_rate",
        "fit_selection_objective",
        "fit_profit_floor_pass",
        "fit_profit_selection_objective",
        "holdout_objective",
    ]
    diag_cols = [
        "source",
        "weight_arm",
        "period",
        "train_rows",
        "valid_rows",
        "weight_effective_frac",
        "utility_ic_target",
        "utility_ic_u",
        "utility_ic_clean_exec",
        "utility_ic_bad",
        "bad_ic_bad",
        "bad_ic_u",
        "bad_ic_clean_exec",
        "utility_seed_std_mean",
        "bad_seed_std_mean",
    ]
    lines = [
        "# Source-Conditioned Two-Head Model Smoke",
        "",
        "Scope: cheap month-forward fixed-model smoke. No LightGBM pipeline, Optuna, policy geometry, or base/meta production training is run.",
        "",
        f"Labels: `{manifest['labels_path']}`",
        f"Model: `{manifest['model']}`",
        f"Source: `{manifest['source']}`",
        f"Weight arms: `{','.join(manifest['weight_arms'])}`",
        f"Utility target: `{manifest['utility_target']}`",
        f"Score rules: `{','.join(manifest['score_rules'])}`",
        f"Bad gate mode: `{manifest['bad_gate_mode']}`",
        f"Bad gate values: `{','.join(str(v) for v in manifest['bad_gate_values'])}`",
        f"Top K: `{','.join(str(v) for v in manifest['top_ks'])}`",
        f"Fit months: `{','.join(manifest['fit_months'])}`",
        f"Holdout month: `{manifest['holdout_month']}`",
        f"Selection policy: `{manifest['selection_policy']}`",
        "",
        "The model is trained month-forward on prior source rows only. Reported fit selection uses Apr-May only, then evaluates June.",
        "",
        "## Source Summary",
        "",
        _format_table(
            source_summary,
            ["source", "rows", "row_frac", "mean_u", "hit_u", "bad_mae_1r_rate", "timeout_rate", "rows_2026_04", "rows_2026_05", "rows_2026_06"],
            limit=20,
        ),
        "",
        "## Selected By Fit",
        "",
        _format_table(selected_by_fit, fit_cols, limit=80),
        "",
        "## Best By Fit",
        "",
        _format_table(best_by_fit, fit_cols, limit=20),
        "",
        "## Fit/Holdout Grid",
        "",
        _format_table(fit_holdout, fit_cols, limit=80),
        "",
        "## Model Diagnostics",
        "",
        _format_table(diagnostics, diag_cols, limit=80),
        "",
        "## Outputs",
        "",
        f"- Monthly: `{manifest['outputs']['monthly']}`",
        f"- Weekly: `{manifest['outputs']['weekly']}`",
        f"- Fit/Holdout: `{manifest['outputs']['fit_holdout']}`",
        f"- Selected by fit: `{manifest['outputs']['selected_by_fit']}`",
        f"- Best by fit: `{manifest['outputs']['best_by_fit']}`",
        f"- Diagnostics: `{manifest['outputs']['diagnostics']}`",
        f"- Manifest: `{manifest['outputs']['manifest']}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_smoke(
    *,
    labels_path: Path,
    output_dir: Path,
    feature_dir: Path,
    feature_list_csv: Path,
    max_feature_store_features: int | None,
    source: str,
    utility_target: str,
    score_rules: list[str],
    bad_gate_mode: str,
    bad_gate_values: list[float],
    top_ks: list[int],
    fit_months: list[str],
    holdout_month: str,
    min_week_rows: int,
    run_gap_hours: float,
    event_feature_store_features: list[str],
    weight_arms: list[str],
    model_kind: str,
    seeds: list[int],
    n_estimators: int,
    max_depth: int | None,
    min_samples_leaf: int,
    learning_rate: float,
    num_leaves: int,
    min_child_samples: int,
    selection_policy: str,
    selection_objective_metric: str,
    fit_floor_require_bounded: bool,
    fit_min_mean_month_u: float | None,
    fit_min_worst_month_u: float | None,
    fit_min_q25_week_u: float | None,
    fit_min_mean_month_return_net: float | None,
    fit_min_worst_month_return_net: float | None,
    fit_min_q25_week_return_net: float | None,
    fit_min_sum_return_net: float | None,
    fit_min_material_positive_return_week_rate: float | None,
    fit_min_selected_rows: int | None,
    fit_min_candidate_timestamp_coverage: float | None,
    fit_min_material_positive_week_rate: float | None,
    fit_min_clean_exec: float | None,
    fit_max_bad_mae: float | None,
    fit_max_p90_mae: float | None,
    min_train_source_rows: int,
    min_valid_source_rows: int,
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
        frame = pd.concat([frame.reset_index(drop=True), feature_matrix.loc[:, new_cols].reset_index(drop=True)], axis=1)
    event_features, event_report = _event_confirmation_features(
        frame,
        event_features=event_feature_store_features,
    )
    if not event_features.empty:
        new_event_cols = [col for col in event_features.columns if col not in frame.columns]
        frame = pd.concat([frame.reset_index(drop=True), event_features.loc[:, new_event_cols].reset_index(drop=True)], axis=1)
    context = _source_context(frame)
    frame = pd.concat([frame.reset_index(drop=True), context.reset_index(drop=True)], axis=1)
    metrics = _path_metrics(frame)
    ft = _first_touch_metrics(frame, metrics)
    source_masks = _build_sources(frame, context, run_gap_hours=run_gap_hours)
    if source not in source_masks:
        raise ValueError(f"Unknown source: {source}")
    source_mask = source_masks[source].reindex(frame.index, fill_value=False).astype(bool)
    source_summary = _source_summary(frame=frame, metrics=ft, context=context, sources={source: source_mask})

    utility_map = _utility_targets(frame, ft)
    if utility_target not in utility_map:
        raise ValueError(f"Unknown utility target: {utility_target}")
    utility_soft = utility_map[utility_target]
    bad_soft = _global_bad_soft(ft)
    full_target = _target_for_selection(ft, utility_soft, bad_soft)
    features = _feature_columns(frame)
    month_series = frame["__ts__"].dt.to_period("M").astype(str)
    months = sorted(month_series.dropna().unique())

    monthly_rows: list[dict[str, Any]] = []
    weekly_rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []

    for month in months[1:]:
        train_mask = month_series.lt(str(month)) & source_mask
        valid_mask = month_series.eq(str(month)) & source_mask
        train_rows = int(train_mask.sum())
        valid_rows = int(valid_mask.sum())
        if train_rows < int(min_train_source_rows) or valid_rows < int(min_valid_source_rows):
            diagnostics.append(
                {
                    "source": source,
                    "period": str(month),
                    "skipped": True,
                    "train_rows": train_rows,
                    "valid_rows": valid_rows,
                }
            )
            continue
        x_train, x_valid = _month_design(
            frame=frame,
            features=features,
            train_mask=train_mask,
            valid_mask=valid_mask,
        )
        valid = frame.loc[valid_mask].copy()
        valid_metrics = ft.loc[valid_mask].copy()
        valid_bad = bad_soft.loc[valid_mask].reset_index(drop=True)
        valid_target = _target_for_selection(
            valid_metrics,
            utility_soft.loc[valid_mask].reset_index(drop=True),
            valid_bad,
        )
        valid_metrics_reset = valid_metrics.reset_index(drop=True)
        valid_target_reset = valid_target.reset_index(drop=True)
        train_frame = frame.loc[train_mask].copy()
        train_metrics = ft.loc[train_mask].copy()
        train_target = full_target.loc[train_mask].copy()

        for weight_arm in weight_arms:
            sample_weight, weight_diag = _weights_for_arm(
                weight_arm,
                train_frame=train_frame,
                train_metrics=train_metrics,
                train_target=train_target,
            )
            bad_pred, train_bad_pred, bad_diag = _fit_predict_model(
                model_kind=model_kind,
                x_train=x_train,
                y_train=bad_soft.loc[train_mask],
                x_valid=x_valid,
                sample_weight=sample_weight,
                seeds=seeds,
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                learning_rate=learning_rate,
                num_leaves=num_leaves,
                min_child_samples=min_child_samples,
            )
            utility_pred, _, utility_diag = _fit_predict_model(
                model_kind=model_kind,
                x_train=x_train,
                y_train=utility_soft.loc[train_mask],
                x_valid=x_valid,
                sample_weight=sample_weight,
                seeds=seeds,
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                learning_rate=learning_rate,
                num_leaves=num_leaves,
                min_child_samples=min_child_samples,
            )
            diag = {
                "source": source,
                "weight_arm": str(weight_arm),
                "period": str(month),
                "train_rows": train_rows,
                "valid_rows": valid_rows,
                "feature_count": int(len(features)),
                **weight_diag,
                "utility_ic_target": _spearman(utility_pred, valid_target_reset["target_soft"]),
                "utility_ic_u": _spearman(utility_pred, valid_metrics_reset["u_policy_net"]),
                "utility_ic_clean_exec": _spearman(utility_pred, valid_metrics_reset["clean_exec_actual"]),
                "utility_ic_bad": _spearman(utility_pred, valid_bad),
                "bad_ic_bad": _spearman(bad_pred, valid_bad),
                "bad_ic_u": _spearman(bad_pred, valid_metrics_reset["u_policy_net"]),
                "bad_ic_clean_exec": _spearman(bad_pred, valid_metrics_reset["clean_exec_actual"]),
                "utility_seed_std_mean": utility_diag.get("seed_std_mean"),
                "bad_seed_std_mean": bad_diag.get("seed_std_mean"),
                "utility_pred_mean": utility_diag.get("pred_mean"),
                "bad_pred_mean": bad_diag.get("pred_mean"),
                "skipped": False,
            }
            diagnostics.append(diag)
            for score_rule in score_rules:
                for bad_gate_value in bad_gate_values:
                    applied_bad_threshold = _bad_gate_threshold(
                        mode=bad_gate_mode,
                        gate_value=float(bad_gate_value),
                        bad_pred=bad_pred,
                        train_bad_pred=train_bad_pred,
                    )
                    spec = TwoHeadSpec(
                        name=(
                            f"{source}::{weight_arm}::{utility_target}_{score_rule}_"
                            f"{bad_gate_mode}{int(round(float(bad_gate_value) * 10000)):04d}"
                        ),
                        utility_target=utility_target,
                        score_rule=score_rule,
                        bad_threshold=float(bad_gate_value),
                    )
                    score = _score_from_rule(
                        score_rule=score_rule,
                        utility_pred=utility_pred,
                        bad_pred=bad_pred,
                        applied_bad_threshold=applied_bad_threshold,
                    )
                    row_diag = {
                        "source": source,
                        "weight_arm": str(weight_arm),
                        "model": bad_diag.get("model", model_kind),
                        "bad_gate_mode": bad_gate_mode,
                        "bad_gate_value": float(bad_gate_value),
                        "applied_bad_threshold": float(applied_bad_threshold),
                        "train_rows": train_rows,
                        "valid_rows": valid_rows,
                        **diag,
                    }
                    m_rows, w_rows = _monthly_weekly_rows(
                        valid_frame=valid,
                        valid_metrics=valid_metrics,
                        valid_target=valid_target,
                        score=score,
                        spec=spec,
                        month=str(month),
                        top_ks=top_ks,
                        diag=row_diag,
                    )
                    for row in m_rows:
                        row["source"] = source
                        row["weight_arm"] = str(weight_arm)
                        row["source_train_rows"] = train_rows
                        row["source_valid_rows"] = valid_rows
                    for row in w_rows:
                        row["source"] = source
                        row["weight_arm"] = str(weight_arm)
                        row["source_train_rows"] = train_rows
                        row["source_valid_rows"] = valid_rows
                    monthly_rows.extend(m_rows)
                    weekly_rows.extend(w_rows)

    monthly = pd.DataFrame(monthly_rows)
    weekly = pd.DataFrame(weekly_rows)
    diagnostics_frame = pd.DataFrame(diagnostics)
    fit_holdout = _source_fit_holdout_summary(
        monthly=monthly,
        weekly=weekly,
        fit_months=fit_months,
        holdout_month=holdout_month,
        min_week_rows=min_week_rows,
    )
    if not fit_holdout.empty:
        fit_holdout.insert(5, "bad_gate_mode", bad_gate_mode)
        fit_holdout.insert(6, "bad_gate_value", fit_holdout["bad_threshold"])
        if "weight_arm" not in fit_holdout.columns and "arm" in fit_holdout.columns:
            arm_parts = fit_holdout["arm"].astype(str).str.split("::")
            fit_holdout.insert(1, "weight_arm", arm_parts.map(lambda parts: parts[1] if len(parts) >= 3 else "none"))
    fit_holdout, selected_by_fit, best_by_fit = _select_by_fit_policy(
        fit_holdout,
        selection_policy=selection_policy,
        require_fit_bounded=fit_floor_require_bounded,
        fit_min_mean_month_u=fit_min_mean_month_u,
        fit_min_worst_month_u=fit_min_worst_month_u,
        fit_min_q25_week_u=fit_min_q25_week_u,
        fit_min_mean_month_return_net=fit_min_mean_month_return_net,
        fit_min_worst_month_return_net=fit_min_worst_month_return_net,
        fit_min_q25_week_return_net=fit_min_q25_week_return_net,
        fit_min_sum_return_net=fit_min_sum_return_net,
        fit_min_material_positive_return_week_rate=fit_min_material_positive_return_week_rate,
        fit_min_selected_rows=fit_min_selected_rows,
        fit_min_candidate_timestamp_coverage=fit_min_candidate_timestamp_coverage,
        fit_min_material_positive_week_rate=fit_min_material_positive_week_rate,
        fit_min_clean_exec=fit_min_clean_exec,
        fit_max_bad_mae=fit_max_bad_mae,
        fit_max_p90_mae=fit_max_p90_mae,
        selection_objective_metric=selection_objective_metric,
    )

    paths = {
        "source_summary": output_dir / "source_summary.csv",
        "monthly": output_dir / "source_conditioned_two_head_model_monthly.csv",
        "weekly": output_dir / "source_conditioned_two_head_model_weekly.csv",
        "fit_holdout": output_dir / "source_conditioned_two_head_model_fit_holdout.csv",
        "selected_by_fit": output_dir / "source_conditioned_two_head_model_selected_by_fit.csv",
        "best_by_fit": output_dir / "source_conditioned_two_head_model_best_by_fit.csv",
        "diagnostics": output_dir / "source_conditioned_two_head_model_diagnostics.csv",
        "manifest": output_dir / "manifest.json",
    }
    source_summary.to_csv(paths["source_summary"], index=False)
    monthly.to_csv(paths["monthly"], index=False)
    weekly.to_csv(paths["weekly"], index=False)
    fit_holdout.to_csv(paths["fit_holdout"], index=False)
    selected_by_fit.to_csv(paths["selected_by_fit"], index=False)
    best_by_fit.to_csv(paths["best_by_fit"], index=False)
    diagnostics_frame.to_csv(paths["diagnostics"], index=False)

    floor_config = {
        "require_fit_bounded": bool(fit_floor_require_bounded),
        "selection_objective_metric": str(selection_objective_metric),
        "fit_min_mean_month_u": fit_min_mean_month_u,
        "fit_min_worst_month_u": fit_min_worst_month_u,
        "fit_min_q25_week_u": fit_min_q25_week_u,
        "fit_min_mean_month_return_net": fit_min_mean_month_return_net,
        "fit_min_worst_month_return_net": fit_min_worst_month_return_net,
        "fit_min_q25_week_return_net": fit_min_q25_week_return_net,
        "fit_min_sum_return_net": fit_min_sum_return_net,
        "fit_min_material_positive_return_week_rate": fit_min_material_positive_return_week_rate,
        "fit_min_selected_rows": fit_min_selected_rows,
        "fit_min_candidate_timestamp_coverage": fit_min_candidate_timestamp_coverage,
        "fit_min_material_positive_week_rate": fit_min_material_positive_week_rate,
        "fit_min_clean_exec": fit_min_clean_exec,
        "fit_max_bad_mae": fit_max_bad_mae,
        "fit_max_p90_mae": fit_max_p90_mae,
    }
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
        "event_feature_report": event_report,
        "feature_count": int(len(features)),
        "model": "LGBMRegressor" if model_kind == "lgbm" else "ExtraTreesRegressor",
        "model_kind": str(model_kind),
        "n_estimators": int(n_estimators),
        "max_depth": max_depth,
        "min_samples_leaf": int(min_samples_leaf),
        "learning_rate": float(learning_rate),
        "num_leaves": int(num_leaves),
        "min_child_samples": int(min_child_samples),
        "selection_policy": str(selection_policy),
        "selection_objective_metric": str(selection_objective_metric),
        "selection_floor_config": floor_config,
        "seeds": [int(seed) for seed in seeds],
        "source": source,
        "weight_arms": [str(v) for v in weight_arms],
        "utility_target": utility_target,
        "bad_gate_mode": bad_gate_mode,
        "score_rules": list(score_rules),
        "bad_gate_values": [float(v) for v in bad_gate_values],
        "bad_thresholds": [float(v) for v in bad_gate_values],
        "top_ks": [int(v) for v in top_ks],
        "fit_months": [str(v) for v in fit_months],
        "holdout_month": str(holdout_month),
        "min_train_source_rows": int(min_train_source_rows),
        "min_valid_source_rows": int(min_valid_source_rows),
        "rows_monthly": int(len(monthly)),
        "rows_weekly": int(len(weekly)),
        "fit_sign_pass_rows": int(fit_holdout["fit_sign_pass"].sum()) if not fit_holdout.empty else 0,
        "holdout_sign_pass_rows": int(fit_holdout["holdout_sign_pass"].sum()) if not fit_holdout.empty else 0,
        "fit_bounded_pass_rows": int(fit_holdout["fit_bounded_pass"].sum()) if not fit_holdout.empty else 0,
        "holdout_bounded_pass_rows": int(fit_holdout["holdout_bounded_pass"].sum()) if not fit_holdout.empty else 0,
        "selected_by_fit_rows": int(len(selected_by_fit)),
        "selected_by_fit_holdout_bounded_rows": int(selected_by_fit["holdout_bounded_pass"].sum())
        if not selected_by_fit.empty
        else 0,
        "best_by_fit_rows": int(len(best_by_fit)),
        "best_by_fit_holdout_bounded_rows": int(best_by_fit["holdout_bounded_pass"].sum())
        if not best_by_fit.empty
        else 0,
        "outputs": {key: str(value) for key, value in paths.items()},
    }
    markdown = _write_markdown(
        output_dir=output_dir,
        source_summary=source_summary,
        fit_holdout=fit_holdout,
        selected_by_fit=selected_by_fit,
        best_by_fit=best_by_fit,
        diagnostics=diagnostics_frame,
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
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--utility-target", default=DEFAULT_UTILITY_TARGET)
    parser.add_argument("--score-rules", default=",".join(DEFAULT_SCORE_RULES))
    parser.add_argument("--bad-thresholds", default=",".join(str(v) for v in DEFAULT_BAD_THRESHOLDS))
    parser.add_argument("--bad-gate-mode", choices=["raw", "train_coverage", "valid_coverage"], default="raw")
    parser.add_argument("--bad-coverages", default=",".join(str(v) for v in DEFAULT_BAD_COVERAGES))
    parser.add_argument("--top-ks", default=",".join(str(v) for v in DEFAULT_TOP_KS))
    parser.add_argument("--fit-months", default=",".join(DEFAULT_FIT_MONTHS))
    parser.add_argument("--holdout-month", default=DEFAULT_HOLDOUT_MONTH)
    parser.add_argument("--min-week-rows", type=int, default=3)
    parser.add_argument("--run-gap-hours", type=float, default=24.0)
    parser.add_argument("--event-feature-store-features", default=",".join(DEFAULT_EVENT_FEATURE_STORE_FEATURES))
    parser.add_argument("--weight-arms", default=",".join(DEFAULT_WEIGHT_ARMS))
    parser.add_argument("--model-kind", choices=["extratrees", "lgbm"], default=DEFAULT_MODEL_KIND)
    parser.add_argument("--seeds", default=",".join(str(v) for v in DEFAULT_SEEDS))
    parser.add_argument("--n-estimators", type=int, default=220)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--min-samples-leaf", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--min-child-samples", type=int, default=50)
    parser.add_argument("--selection-policy", choices=["legacy", "fit_profit_floors"], default=DEFAULT_SELECTION_POLICY)
    parser.add_argument("--selection-objective-metric", choices=["u", "return_net"], default="u")
    parser.add_argument("--fit-floor-allow-unbounded", action="store_true")
    parser.add_argument("--fit-min-mean-month-u", default=None)
    parser.add_argument("--fit-min-worst-month-u", default=None)
    parser.add_argument("--fit-min-q25-week-u", default=None)
    parser.add_argument("--fit-min-mean-month-return-net", default=None)
    parser.add_argument("--fit-min-worst-month-return-net", default=None)
    parser.add_argument("--fit-min-q25-week-return-net", default=None)
    parser.add_argument("--fit-min-sum-return-net", default=None)
    parser.add_argument("--fit-min-material-positive-return-week-rate", default=None)
    parser.add_argument("--fit-min-selected-rows", default=None)
    parser.add_argument("--fit-min-candidate-timestamp-coverage", default=None)
    parser.add_argument("--fit-min-material-positive-week-rate", default=None)
    parser.add_argument("--fit-min-clean-exec", default=None)
    parser.add_argument("--fit-max-bad-mae", default=None)
    parser.add_argument("--fit-max-p90-mae", default=None)
    parser.add_argument("--min-train-source-rows", type=int, default=500)
    parser.add_argument("--min-valid-source-rows", type=int, default=100)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    weight_arms = _parse_csv(args.weight_arms)
    missing_weight_arms = sorted(set(weight_arms) - ({"none"} | set(WEIGHT_ARMS)))
    if missing_weight_arms:
        raise ValueError(f"Unknown weight arms: {missing_weight_arms}")
    manifest = run_smoke(
        labels_path=args.labels_path,
        output_dir=args.output_dir,
        feature_dir=args.feature_dir,
        feature_list_csv=args.feature_list_csv,
        max_feature_store_features=args.max_feature_store_features,
        source=str(args.source),
        utility_target=str(args.utility_target),
        score_rules=_parse_csv(args.score_rules),
        bad_gate_mode=str(args.bad_gate_mode),
        bad_gate_values=(
            _parse_float_csv(args.bad_thresholds)
            if str(args.bad_gate_mode) == "raw"
            else _parse_float_csv(args.bad_coverages)
        ),
        top_ks=[int(v) for v in _parse_csv(args.top_ks)],
        fit_months=_parse_csv(args.fit_months),
        holdout_month=str(args.holdout_month),
        min_week_rows=int(args.min_week_rows),
        run_gap_hours=float(args.run_gap_hours),
        event_feature_store_features=_parse_csv(args.event_feature_store_features),
        weight_arms=weight_arms,
        model_kind=str(args.model_kind),
        seeds=[int(v) for v in _parse_csv(args.seeds)],
        n_estimators=int(args.n_estimators),
        max_depth=int(args.max_depth) if args.max_depth is not None and int(args.max_depth) > 0 else None,
        min_samples_leaf=int(args.min_samples_leaf),
        learning_rate=float(args.learning_rate),
        num_leaves=int(args.num_leaves),
        min_child_samples=int(args.min_child_samples),
        selection_policy=str(args.selection_policy),
        selection_objective_metric=str(args.selection_objective_metric),
        fit_floor_require_bounded=not bool(args.fit_floor_allow_unbounded),
        fit_min_mean_month_u=_optional_float(args.fit_min_mean_month_u),
        fit_min_worst_month_u=_optional_float(args.fit_min_worst_month_u),
        fit_min_q25_week_u=_optional_float(args.fit_min_q25_week_u),
        fit_min_mean_month_return_net=_optional_float(args.fit_min_mean_month_return_net),
        fit_min_worst_month_return_net=_optional_float(args.fit_min_worst_month_return_net),
        fit_min_q25_week_return_net=_optional_float(args.fit_min_q25_week_return_net),
        fit_min_sum_return_net=_optional_float(args.fit_min_sum_return_net),
        fit_min_material_positive_return_week_rate=_optional_float(
            args.fit_min_material_positive_return_week_rate
        ),
        fit_min_selected_rows=_optional_int(args.fit_min_selected_rows),
        fit_min_candidate_timestamp_coverage=_optional_float(args.fit_min_candidate_timestamp_coverage),
        fit_min_material_positive_week_rate=_optional_float(args.fit_min_material_positive_week_rate),
        fit_min_clean_exec=_optional_float(args.fit_min_clean_exec),
        fit_max_bad_mae=_optional_float(args.fit_max_bad_mae),
        fit_max_p90_mae=_optional_float(args.fit_max_p90_mae),
        min_train_source_rows=int(args.min_train_source_rows),
        min_valid_source_rows=int(args.min_valid_source_rows),
    )
    summary_keys = [
        "output_dir",
        "rows",
        "feature_count",
        "model",
        "source",
        "weight_arms",
        "utility_target",
        "bad_gate_mode",
        "score_rules",
        "bad_gate_values",
        "top_ks",
        "fit_sign_pass_rows",
        "holdout_sign_pass_rows",
        "fit_bounded_pass_rows",
        "holdout_bounded_pass_rows",
        "selected_by_fit_rows",
        "selected_by_fit_holdout_bounded_rows",
        "best_by_fit_rows",
        "best_by_fit_holdout_bounded_rows",
        "outputs",
    ]
    print(json.dumps(_json_safe({key: manifest.get(key) for key in summary_keys}), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
