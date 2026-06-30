"""First-stage per-strategy bad/good market-state models."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Any, Literal, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from extreme_price_movements.performance_regimes.labels import (
    StrategyPerformanceLabelBundle,
)


@dataclass(frozen=True)
class TimeSeriesSplitSpec:
    n_splits: int = 3
    purge: int = 0
    embargo: int = 0
    min_train_size: int | None = None
    purge_hours: float | None = None
    embargo_hours: float | None = None


@dataclass(frozen=True)
class FirstStageLGBMConfig:
    objective: str = "binary"
    max_depth: int = 4
    num_leaves: int = 16
    min_child_samples_fraction: float = 0.01
    learning_rate: float = 0.03
    n_estimators: int = 2000
    subsample: float = 0.85
    subsample_freq: int = 1
    colsample_bytree: float = 0.85
    min_gain_to_split: float = 1e-3
    lambda_l1: float = 0.0
    lambda_l2: float = 1e-3
    early_stopping_rounds: int = 50
    random_state: int = 42


@dataclass(frozen=True)
class FirstStageFoldModel:
    strategy: str
    direction: Literal["bad", "good"]
    fold_id: int
    model: Any
    feature_columns: tuple[str, ...]
    fill_values: pd.Series
    train_idx: np.ndarray
    valid_idx: np.ndarray
    min_child_samples: int
    baseline_prediction: float
    model_type: str


@dataclass(frozen=True)
class FirstStageModelResult:
    strategy: str
    direction: Literal["bad", "good"]
    fold_models: tuple[FirstStageFoldModel, ...]
    oof_predictions: pd.Series
    baseline_oof_predictions: pd.Series
    diagnostics: pd.DataFrame


@dataclass(frozen=True)
class FirstStageModelBundle:
    by_strategy_direction: dict[tuple[str, Literal["bad", "good"]], FirstStageModelResult]
    diagnostics: pd.DataFrame
    config: FirstStageLGBMConfig
    cv: TimeSeriesSplitSpec


def walk_forward_splits(
    index: pd.Index,
    spec: TimeSeriesSplitSpec,
) -> list[tuple[np.ndarray, np.ndarray]]:
    n = len(index)
    if n < 2:
        return []
    n_splits = max(2, int(spec.n_splits))
    base_train = max(1, n // (n_splits + 1))
    if spec.min_train_size is not None:
        base_train = max(base_train, int(spec.min_train_size))
    if base_train >= n:
        return []
    remaining = n - base_train
    fold_sizes = np.full(n_splits, remaining // n_splits, dtype=int)
    fold_sizes[: remaining % n_splits] += 1
    positions = np.arange(n, dtype=int)
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    start = base_train
    ts = pd.to_datetime(index, utc=True, errors="coerce")
    time_based = bool(
        spec.purge_hours is not None
        and pd.notna(ts).all()
    )
    for size in fold_sizes:
        valid_start = start
        valid_end = min(n, start + int(size))
        start = valid_end
        if valid_end <= valid_start:
            continue
        valid_idx = positions[valid_start:valid_end]
        if time_based:
            first_valid = ts[valid_start]
            purge_td = pd.Timedelta(hours=float(spec.purge_hours or 0.0))
            train_mask = ts < first_valid - purge_td
            train_mask_arr = np.asarray(train_mask, dtype=bool)
            train_idx = positions[:valid_start][train_mask_arr[:valid_start]]
        else:
            train_end = max(0, valid_start - int(spec.purge))
            train_idx = positions[:train_end]
        if train_idx.size == 0:
            continue
        if spec.min_train_size is not None and train_idx.size < int(spec.min_train_size):
            continue
        splits.append((train_idx.astype(int), valid_idx.astype(int)))
    return splits


def _weighted_mse(y: np.ndarray, pred: np.ndarray, weight: np.ndarray) -> float:
    ok = np.isfinite(y) & np.isfinite(pred) & np.isfinite(weight) & (weight >= 0.0)
    if not ok.any():
        return np.nan
    return float(np.average((y[ok] - pred[ok]) ** 2, weights=np.maximum(weight[ok], 1e-12)))


def _weighted_r2(y: np.ndarray, pred: np.ndarray, weight: np.ndarray) -> float:
    ok = np.isfinite(y) & np.isfinite(pred) & np.isfinite(weight) & (weight >= 0.0)
    if not ok.any():
        return np.nan
    y_ok = y[ok]
    w = np.maximum(weight[ok], 1e-12)
    mean = float(np.average(y_ok, weights=w))
    denom = float(np.sum(w * (y_ok - mean) ** 2))
    if denom <= 1e-12:
        return 0.0
    return float(1.0 - np.sum(w * (y_ok - pred[ok]) ** 2) / denom)


def _fit_lgbm_or_fallback(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    w_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    w_valid: pd.Series,
    *,
    config: FirstStageLGBMConfig,
    min_child_samples: int,
) -> tuple[Any, str, int | None]:
    try:
        from lightgbm import LGBMRegressor, early_stopping, log_evaluation

        model = LGBMRegressor(
            objective="regression",
            max_depth=int(config.max_depth),
            num_leaves=int(config.num_leaves),
            min_child_samples=int(min_child_samples),
            learning_rate=float(config.learning_rate),
            n_estimators=int(config.n_estimators),
            subsample=float(config.subsample),
            subsample_freq=int(config.subsample_freq),
            colsample_bytree=float(config.colsample_bytree),
            min_gain_to_split=float(config.min_gain_to_split),
            reg_alpha=float(config.lambda_l1),
            reg_lambda=float(config.lambda_l2),
            random_state=int(config.random_state),
            verbosity=-1,
        )
        callbacks = [
            early_stopping(int(config.early_stopping_rounds), verbose=False),
            log_evaluation(period=0),
        ]
        model.fit(
            X_train,
            y_train,
            sample_weight=w_train,
            eval_set=[(X_valid, y_valid)],
            eval_sample_weight=[w_valid],
            eval_metric="l2",
            callbacks=callbacks,
        )
        best_iteration = getattr(model, "best_iteration_", None)
        return model, "lightgbm_regressor_soft_label", best_iteration
    except Exception:
        model = HistGradientBoostingRegressor(
            max_iter=min(int(config.n_estimators), 300),
            learning_rate=float(config.learning_rate),
            max_leaf_nodes=int(config.num_leaves),
            max_depth=int(config.max_depth),
            l2_regularization=float(config.lambda_l2),
            random_state=int(config.random_state),
        )
        model.fit(X_train, y_train, sample_weight=w_train)
        return model, "hist_gradient_boosting_fallback", None


def _predict(model: Any, X: pd.DataFrame) -> np.ndarray:
    pred = np.asarray(model.predict(X), dtype=np.float32)
    return np.clip(pred, 0.0, 1.0)


def _leaf_diagnostics(model: Any, X_valid: pd.DataFrame) -> dict[str, float]:
    if not hasattr(model, "predict"):
        return {
            "effective_leaves_used": np.nan,
            "median_leaf_coverage": np.nan,
            "p05_leaf_coverage": np.nan,
            "p95_leaf_coverage": np.nan,
            "trees_with_no_splits": np.nan,
        }
    try:
        leaves = np.asarray(model.predict(X_valid, pred_leaf=True))
        if leaves.ndim == 1:
            leaves = leaves.reshape(-1, 1)
        coverages: list[float] = []
        effective = 0
        for tree_id in range(leaves.shape[1]):
            _, counts = np.unique(leaves[:, tree_id], return_counts=True)
            effective += int(len(counts))
            coverages.extend((counts / max(len(leaves), 1)).astype(float).tolist())
        return {
            "effective_leaves_used": float(effective),
            "median_leaf_coverage": float(np.nanmedian(coverages)) if coverages else np.nan,
            "p05_leaf_coverage": float(np.nanpercentile(coverages, 5)) if coverages else np.nan,
            "p95_leaf_coverage": float(np.nanpercentile(coverages, 95)) if coverages else np.nan,
            "trees_with_no_splits": float(sum(1 for c in coverages if c >= 0.999)),
        }
    except Exception:
        return {
            "effective_leaves_used": np.nan,
            "median_leaf_coverage": np.nan,
            "p05_leaf_coverage": np.nan,
            "p95_leaf_coverage": np.nan,
            "trees_with_no_splits": np.nan,
        }


def _top_features(model: Any, feature_columns: Sequence[str]) -> str:
    values = getattr(model, "feature_importances_", None)
    if values is None:
        return ""
    arr = np.asarray(values, dtype=float)
    if arr.size != len(feature_columns):
        return ""
    order = np.argsort(arr)[::-1][:10]
    return ",".join(str(feature_columns[i]) for i in order if arr[i] > 0)


def train_first_stage_bad_good_models(
    X_t: pd.DataFrame,
    labels: StrategyPerformanceLabelBundle,
    *,
    strategies: Sequence[str],
    cv: TimeSeriesSplitSpec,
    lgbm_config: FirstStageLGBMConfig,
) -> FirstStageModelBundle:
    """Train fold-local bad/good soft-label models with timestamp OOF outputs."""

    X = (
        X_t.sort_index()
        .replace([np.inf, -np.inf], np.nan)
        .apply(pd.to_numeric, errors="coerce")
        .astype(np.float32, copy=False)
    )
    feature_columns = tuple(str(c) for c in X.columns)
    splits = walk_forward_splits(X.index, cv)
    if not splits:
        raise ValueError("No valid walk-forward splits produced")
    by_key: dict[tuple[str, Literal["bad", "good"]], FirstStageModelResult] = {}
    all_diag: list[pd.DataFrame] = []
    for strategy in [str(s) for s in strategies]:
        if strategy not in labels.by_strategy:
            raise KeyError(f"Missing labels for strategy {strategy}")
        label_set = labels.by_strategy[strategy]
        for direction in ("bad", "good"):
            y = (label_set.bad_label if direction == "bad" else label_set.good_label).reindex(X.index)
            sw = (
                label_set.bad_sample_weight
                if direction == "bad"
                else label_set.good_sample_weight
            ).reindex(X.index)
            oof = pd.Series(np.nan, index=X.index, dtype=float)
            baseline = pd.Series(np.nan, index=X.index, dtype=float)
            fold_models: list[FirstStageFoldModel] = []
            rows: list[dict[str, object]] = []
            for fold_id, (train_idx, valid_idx) in enumerate(splits, start=1):
                X_train_raw = X.iloc[train_idx]
                X_valid_raw = X.iloc[valid_idx]
                fill_values = (
                    X_train_raw.median(numeric_only=True)
                    .reindex(feature_columns)
                    .fillna(0.0)
                    .astype(np.float32)
                )
                X_train = (
                    X_train_raw.reindex(columns=feature_columns)
                    .fillna(fill_values)
                    .astype(np.float32, copy=False)
                )
                X_valid = (
                    X_valid_raw.reindex(columns=feature_columns)
                    .fillna(fill_values)
                    .astype(np.float32, copy=False)
                )
                y_train = pd.to_numeric(y.iloc[train_idx], errors="coerce").fillna(0.5).astype(np.float32)
                y_valid = pd.to_numeric(y.iloc[valid_idx], errors="coerce").fillna(0.5).astype(np.float32)
                w_train = pd.to_numeric(sw.iloc[train_idx], errors="coerce").fillna(1.0).astype(np.float32)
                w_valid = pd.to_numeric(sw.iloc[valid_idx], errors="coerce").fillna(1.0).astype(np.float32)
                min_child_samples = int(
                    ceil(float(lgbm_config.min_child_samples_fraction) * len(train_idx))
                )
                min_child_samples = max(1, min_child_samples)
                base_value = float(np.average(y_train, weights=np.maximum(w_train, 1e-12)))
                model, model_type, best_iteration = _fit_lgbm_or_fallback(
                    X_train,
                    y_train,
                    w_train,
                    X_valid,
                    y_valid,
                    w_valid,
                    config=lgbm_config,
                    min_child_samples=min_child_samples,
                )
                pred_valid = _predict(model, X_valid)
                pred_train = _predict(model, X_train)
                oof.iloc[valid_idx] = pred_valid
                baseline.iloc[valid_idx] = base_value
                leaf_diag = _leaf_diagnostics(model, X_valid)
                valid_loss = _weighted_mse(
                    y_valid.to_numpy(dtype=float),
                    pred_valid,
                    w_valid.to_numpy(dtype=float),
                )
                train_loss = _weighted_mse(
                    y_train.to_numpy(dtype=float),
                    pred_train,
                    w_train.to_numpy(dtype=float),
                )
                pred_std = float(np.nanstd(pred_valid))
                near_mean = float(np.mean(np.abs(pred_valid - base_value) <= 1e-4))
                rows.append(
                    {
                        "strategy": strategy,
                        "direction": direction,
                        "fold": int(fold_id),
                        "n_train": int(len(train_idx)),
                        "n_valid": int(len(valid_idx)),
                        "objective": "regression_soft_label",
                        "max_depth": int(lgbm_config.max_depth),
                        "num_leaves": int(lgbm_config.num_leaves),
                        "min_child_samples": int(min_child_samples),
                        "min_gain_to_split": float(lgbm_config.min_gain_to_split),
                        "best_iteration": best_iteration,
                        "oof_weighted_brier": valid_loss,
                        "oof_weighted_r2": _weighted_r2(
                            y_valid.to_numpy(dtype=float),
                            pred_valid,
                            w_valid.to_numpy(dtype=float),
                        ),
                        "train_loss": train_loss,
                        "valid_loss": valid_loss,
                        "train_valid_gap": float(valid_loss - train_loss),
                        "prediction_std": pred_std,
                        "share_predictions_near_global_mean": near_mean,
                        "top_features": _top_features(model, feature_columns),
                        "over_regularised_flag": bool(
                            pred_std <= 1e-6
                            or near_mean >= 0.95
                            or leaf_diag.get("effective_leaves_used", np.inf) <= 2
                        ),
                        **leaf_diag,
                    }
                )
                fold_models.append(
                    FirstStageFoldModel(
                        strategy=strategy,
                        direction=direction,  # type: ignore[arg-type]
                        fold_id=int(fold_id),
                        model=model,
                        feature_columns=feature_columns,
                        fill_values=fill_values,
                        train_idx=train_idx,
                        valid_idx=valid_idx,
                        min_child_samples=int(min_child_samples),
                        baseline_prediction=base_value,
                        model_type=model_type,
                    )
                )
            diagnostics = pd.DataFrame(rows)
            result = FirstStageModelResult(
                strategy=strategy,
                direction=direction,  # type: ignore[arg-type]
                fold_models=tuple(fold_models),
                oof_predictions=oof,
                baseline_oof_predictions=baseline,
                diagnostics=diagnostics,
            )
            by_key[(strategy, direction)] = result  # type: ignore[index]
            all_diag.append(diagnostics)
    return FirstStageModelBundle(
        by_strategy_direction=by_key,
        diagnostics=pd.concat(all_diag, ignore_index=True) if all_diag else pd.DataFrame(),
        config=lgbm_config,
        cv=cv,
    )
