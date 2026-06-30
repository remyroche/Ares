"""One regularised market-state expert per archetype."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Literal, Mapping

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from extreme_price_movements.performance_regimes.first_stage_models import (
    TimeSeriesSplitSpec,
    walk_forward_splits,
)
from extreme_price_movements.performance_regimes.leaf_scoring import weighted_brier


@dataclass(frozen=True)
class ArchetypeExpertConfig:
    model_type: Literal["lgbm", "xgb"] = "lgbm"
    objective: str = "binary"
    max_depth_values: tuple[int, ...] = (2, 3)
    min_child_samples_fraction: float = 0.02
    learning_rate: float = 0.03
    n_estimators: int = 2000
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_gain_to_split_range: tuple[float, float] = (1e-4, 1e-1)
    lambda_l2_range: tuple[float, float] = (1e-4, 1e1)
    early_stopping_rounds: int = 30
    hpo_trials: int = 20
    random_state: int = 42


@dataclass(frozen=True)
class ArchetypeExpertResult:
    archetype_id: str
    models: tuple[object, ...]
    feature_columns: tuple[str, ...]
    excluded_identity_columns: tuple[str, ...]
    oof_prediction: pd.Series
    activity_score: pd.Series
    diagnostics: pd.DataFrame


@dataclass(frozen=True)
class ArchetypeExpertBundle:
    by_archetype: dict[str, ArchetypeExpertResult]
    scores: pd.DataFrame
    activity_scores: pd.DataFrame
    diagnostics: pd.DataFrame


@dataclass(frozen=True)
class FrozenArchetypeExpertScores:
    p_active: pd.DataFrame
    activity_scores: pd.DataFrame
    diagnostics: pd.DataFrame


def _identity_leakage_columns(columns: list[str], archetype_id: str) -> list[str]:
    needles = {
        archetype_id,
        f"A_{archetype_id}",
        f"activity_{archetype_id}",
        f"{archetype_id}__activity",
    }
    out = []
    for col in columns:
        text = str(col)
        if text in needles or text.endswith(f"__{archetype_id}") or text.startswith(f"{archetype_id}__"):
            out.append(text)
    return out


def _fit_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    w_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    w_valid: pd.Series,
    *,
    config: ArchetypeExpertConfig,
    min_child_samples: int,
):
    try:
        from lightgbm import LGBMRegressor, early_stopping, log_evaluation

        model = LGBMRegressor(
            objective="regression",
            max_depth=int(config.max_depth_values[0]),
            num_leaves=2 ** int(config.max_depth_values[0]),
            min_child_samples=int(min_child_samples),
            learning_rate=float(config.learning_rate),
            n_estimators=int(config.n_estimators),
            subsample=float(config.subsample),
            colsample_bytree=float(config.colsample_bytree),
            min_gain_to_split=float(config.min_gain_to_split_range[0]),
            reg_lambda=float(config.lambda_l2_range[0]),
            random_state=int(config.random_state),
            verbosity=-1,
        )
        model.fit(
            X_train,
            y_train,
            sample_weight=w_train,
            eval_set=[(X_valid, y_valid)],
            eval_sample_weight=[w_valid],
            eval_metric="l2",
            callbacks=[
                early_stopping(int(config.early_stopping_rounds), verbose=False),
                log_evaluation(period=0),
            ],
        )
        return model, "lightgbm"
    except Exception:
        model = HistGradientBoostingRegressor(
            max_iter=min(int(config.n_estimators), 300),
            learning_rate=float(config.learning_rate),
            max_depth=int(config.max_depth_values[0]),
            random_state=int(config.random_state),
        )
        try:
            model.fit(X_train, y_train, sample_weight=w_train)
            return model, "hist_gradient_boosting"
        except Exception:
            ridge = Ridge(alpha=1.0)
            ridge.fit(X_train, y_train, sample_weight=w_train)
            return ridge, "ridge_fallback"


def _predict(model, X: pd.DataFrame) -> np.ndarray:
    return np.clip(np.asarray(model.predict(X), dtype=np.float32), 0.0, 1.0)


def _filled_fold_matrix(
    X_values: np.ndarray,
    columns: pd.Index,
    train_idx: np.ndarray,
    valid_idx: np.ndarray,
    feature_columns: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray, pd.Series]:
    """Return train/valid float32 arrays with train-fitted median imputation.

    Archetype experts often train many one-vs-activity models against the same
    market-state matrix. Building pandas slices and downcasting them for every
    archetype is expensive; this keeps the imputation causal while doing the
    repeated work in NumPy.
    """

    col_positions = columns.get_indexer(feature_columns)
    if np.any(col_positions < 0):
        missing = [feature_columns[i] for i, pos in enumerate(col_positions) if pos < 0]
        raise KeyError(f"Missing expert feature columns: {missing[:5]}")
    train_raw = np.asarray(X_values[np.asarray(train_idx)[:, None], col_positions], dtype=np.float32)
    valid_raw = np.asarray(X_values[np.asarray(valid_idx)[:, None], col_positions], dtype=np.float32)
    with np.errstate(all="ignore"):
        fill_values = np.nanmedian(train_raw, axis=0).astype(np.float32, copy=False)
    fill_values = np.where(np.isfinite(fill_values), fill_values, np.float32(0.0)).astype(np.float32, copy=False)
    train = np.where(np.isfinite(train_raw), train_raw, fill_values).astype(np.float32, copy=False)
    valid = np.where(np.isfinite(valid_raw), valid_raw, fill_values).astype(np.float32, copy=False)
    fill = pd.Series(fill_values, index=pd.Index(feature_columns), dtype=np.float32)
    return train, valid, fill


def train_archetype_experts(
    X_t: pd.DataFrame,
    archetype_targets: Mapping[str, pd.Series],
    archetype_sample_weights: Mapping[str, pd.Series],
    *,
    cv: TimeSeriesSplitSpec,
    config: ArchetypeExpertConfig,
) -> ArchetypeExpertBundle:
    """Train one activity expert per archetype with identity columns removed."""

    X_all = (
        X_t.sort_index()
        .replace([np.inf, -np.inf], np.nan)
        .apply(pd.to_numeric, errors="coerce")
        .astype(np.float32, copy=False)
    )
    X_columns = pd.Index(X_all.columns)
    X_values = X_all.to_numpy(dtype=np.float32, copy=False)
    splits = walk_forward_splits(X_all.index, cv)
    if not splits:
        raise ValueError("No valid walk-forward splits for archetype experts")
    results: dict[str, ArchetypeExpertResult] = {}
    scores: dict[str, pd.Series] = {}
    activity_scores: dict[str, pd.Series] = {}
    all_diag: list[pd.DataFrame] = []
    matrix_cache: dict[tuple[tuple[str, ...], int], tuple[np.ndarray, np.ndarray, pd.Series]] = {}
    for archetype_id, target in archetype_targets.items():
        y = pd.to_numeric(target.reindex(X_all.index), errors="coerce").fillna(0.0).clip(0.0, 1.0)
        sw = pd.to_numeric(
            archetype_sample_weights.get(archetype_id, pd.Series(1.0, index=X_all.index)).reindex(X_all.index),
            errors="coerce",
        ).fillna(1.0)
        excluded = _identity_leakage_columns(list(X_all.columns), str(archetype_id))
        feature_columns = tuple(col for col in X_all.columns if col not in set(excluded))
        oof = pd.Series(np.nan, index=X_all.index, dtype=float)
        models: list[object] = []
        rows: list[dict[str, object]] = []
        for fold_id, (train_idx, valid_idx) in enumerate(splits, start=1):
            cache_key = (feature_columns, int(fold_id))
            cached = matrix_cache.get(cache_key)
            if cached is None:
                cached = _filled_fold_matrix(X_values, X_columns, train_idx, valid_idx, feature_columns)
                matrix_cache[cache_key] = cached
            X_train, X_valid, fill = cached
            min_child_samples = max(1, int(ceil(float(config.min_child_samples_fraction) * len(train_idx))))
            model, model_type = _fit_model(
                X_train,
                y.iloc[train_idx],
                sw.iloc[train_idx],
                X_valid,
                y.iloc[valid_idx],
                sw.iloc[valid_idx],
                config=config,
                min_child_samples=min_child_samples,
            )
            pred = _predict(model, X_valid)
            oof.iloc[valid_idx] = pred
            models.append({"model": model, "fill_values": fill, "model_type": model_type})
            rows.append(
                {
                    "archetype_id": archetype_id,
                    "fold": int(fold_id),
                    "n_train": int(len(train_idx)),
                    "n_valid": int(len(valid_idx)),
                    "model_type": model_type,
                    "min_child_samples": int(min_child_samples),
                    "oof_weighted_brier": weighted_brier(
                        y.iloc[valid_idx],
                        pd.Series(pred, index=y.index[valid_idx]),
                        sw.iloc[valid_idx],
                    ),
                    "prediction_std": float(np.nanstd(pred)),
                    "target_mean": float(y.iloc[train_idx].mean()),
                    "excluded_identity_column_count": int(len(excluded)),
                }
            )
        activity = (2.0 * oof - 1.0).clip(-1.0, 1.0)
        diag = pd.DataFrame(rows)
        result = ArchetypeExpertResult(
            archetype_id=str(archetype_id),
            models=tuple(models),
            feature_columns=feature_columns,
            excluded_identity_columns=tuple(excluded),
            oof_prediction=oof,
            activity_score=activity,
            diagnostics=diag,
        )
        results[str(archetype_id)] = result
        scores[str(archetype_id)] = oof
        activity_scores[str(archetype_id)] = activity
        all_diag.append(diag)
    return ArchetypeExpertBundle(
        by_archetype=results,
        scores=pd.DataFrame(scores, index=X_all.index),
        activity_scores=pd.DataFrame(activity_scores, index=X_all.index),
        diagnostics=pd.concat(all_diag, ignore_index=True) if all_diag else pd.DataFrame(),
    )


def score_frozen_archetype_experts(
    X_t: pd.DataFrame,
    expert_bundle: ArchetypeExpertBundle,
) -> FrozenArchetypeExpertScores:
    """Apply trained archetype experts to validation/inference timestamps."""

    X_numeric = (
        X_t.replace([np.inf, -np.inf], np.nan)
        .apply(pd.to_numeric, errors="coerce")
        .astype(np.float32, copy=False)
    )
    raw_cache: dict[tuple[str, ...], np.ndarray] = {}
    p_active: dict[str, pd.Series] = {}
    rows: list[dict[str, object]] = []
    for archetype_id, result in expert_bundle.by_archetype.items():
        raw = raw_cache.get(result.feature_columns)
        if raw is None:
            raw = X_numeric.reindex(columns=result.feature_columns).to_numpy(dtype=np.float32, copy=False)
            raw_cache[result.feature_columns] = raw
        fold_preds = []
        for payload in result.models:
            if isinstance(payload, dict):
                model = payload.get("model")
                fill_values = payload.get("fill_values")
                model_type = str(payload.get("model_type", "unknown"))
            else:
                model = payload
                fill_values = pd.Series(0.0, index=result.feature_columns)
                model_type = type(model).__name__
            if isinstance(fill_values, pd.Series):
                fill = (
                    fill_values.reindex(result.feature_columns)
                    .fillna(0.0)
                    .to_numpy(dtype=np.float32, copy=False)
                )
            else:
                fill = np.zeros(len(result.feature_columns), dtype=np.float32)
            X = np.where(np.isfinite(raw), raw, fill).astype(np.float32, copy=False)
            try:
                pred = np.clip(np.asarray(model.predict(X), dtype=np.float32), 0.0, 1.0)
            except Exception:
                pred = np.full(len(X_t), np.nan, dtype=float)
            fold_preds.append(pred)
            rows.append(
                {
                    "archetype_id": archetype_id,
                    "model_type": model_type,
                    "scored_rows": int(len(X_t)),
                    "finite_prediction_share": float(np.isfinite(pred).mean()) if len(pred) else 0.0,
                }
            )
        if fold_preds:
            p = np.nanmean(np.vstack(fold_preds), axis=0)
        else:
            p = np.full(len(X_t), np.nan, dtype=float)
        p_active[archetype_id] = pd.Series(np.clip(p, 0.0, 1.0), index=X_t.index)
    p_frame = pd.DataFrame(p_active, index=X_t.index)
    activity = (2.0 * p_frame - 1.0).clip(-1.0, 1.0)
    return FrozenArchetypeExpertScores(
        p_active=p_frame,
        activity_scores=activity,
        diagnostics=pd.DataFrame(rows),
    )
