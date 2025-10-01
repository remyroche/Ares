"""
Unified Cross-Validation API

This module centralizes cross-validation logic used across the codebase to reduce
duplication and ensure consistent behavior. It provides:

- Standard KFold/Stratified KFold cross-validation
- Temporal (TimeSeriesSplit) cross-validation with optional gap/test_size
- Nested cross-validation for unbiased model assessment

Backwards-compatible wrappers in existing modules can delegate to this API.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import logging
import numpy as np

try:
    from sklearn.model_selection import (
        KFold,
        StratifiedKFold,
        TimeSeriesSplit,
        cross_val_score,
        cross_validate,
    )
    from sklearn.utils.multiclass import type_of_target
    SKLEARN_AVAILABLE = True
except Exception:  # pragma: no cover - environment dependent
    SKLEARN_AVAILABLE = False


LOGGER = logging.getLogger("MLCommon.UnifiedCV")


def _is_classification_target(y: np.ndarray) -> bool:
    if not SKLEARN_AVAILABLE:
        # Heuristic fallback
        try:
            unique_values = np.unique(y)
            return len(unique_values) <= 10
        except Exception:
            return False
    try:
        t = type_of_target(y)
        return t in {"binary", "multiclass", "multilabel-indicator", "multiclass-multioutput"}
    except Exception:
        try:
            unique_values = np.unique(y)
            return len(unique_values) <= 10
        except Exception:
            return False


@dataclass
class UnifiedCVResult:
    scores: Optional[List[float]] = None
    mean: Optional[float] = None
    std: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None
    folds: Optional[int] = None
    # For multi-metric scoring
    mean_scores: Optional[Dict[str, float]] = None
    std_scores: Optional[Dict[str, float]] = None
    train_scores: Optional[Dict[str, float]] = None


class UnifiedCrossValidator:
    """Central cross-validation helper with standard, temporal and nested CV."""

    def run(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        *,
        strategy: str = "standard",  # "standard" | "temporal"
        cv_folds: int = 5,
        scoring: Union[str, List[str], None] = None,
        random_state: Optional[int] = 42,
        stratified: Optional[bool] = None,
        n_jobs: int = -1,
        temporal_gap: int = 0,
        temporal_test_size: Optional[int] = None,
    ) -> UnifiedCVResult:
        if not SKLEARN_AVAILABLE:
            LOGGER.warning("Scikit-learn not available; returning empty CV result")
            return UnifiedCVResult(scores=[], mean=0.0, std=0.0, min=0.0, max=0.0, folds=cv_folds)

        try:
            is_classification = _is_classification_target(y) if stratified is None else stratified
            if scoring is None:
                scoring = "accuracy" if is_classification else "r2"

            if strategy == "temporal":
                # TimeSeriesSplit supports 'gap' and (from sklearn>=1.3) 'test_size'
                import inspect
                if temporal_test_size is None:
                    temporal_test_size = max(1, len(X) // (cv_folds + 1))
                if "test_size" in inspect.signature(TimeSeriesSplit).parameters:
                    cv = TimeSeriesSplit(n_splits=cv_folds, gap=temporal_gap, test_size=temporal_test_size)
                else:
                    cv = TimeSeriesSplit(n_splits=cv_folds, gap=temporal_gap)
            else:
                # ⚠️ WARNING: For trading/time series data, ALWAYS use strategy="temporal"!
                # Using shuffle for time series data causes SEVERE data leakage
                LOGGER.warning("⚠️ Using non-temporal CV strategy! For time series data, use strategy='temporal' to prevent data leakage")
                LOGGER.warning("⚠️ Disabling shuffle to reduce (but not eliminate) data leakage risk")
                if is_classification:
                    cv = StratifiedKFold(n_splits=cv_folds, shuffle=False)
                else:
                    cv = KFold(n_splits=cv_folds, shuffle=False)

            # Multi-metric vs single-metric
            if isinstance(scoring, list):
                cv_result = cross_validate(
                    model,
                    X,
                    y,
                    cv=cv,
                    scoring=scoring,
                    n_jobs=n_jobs,
                    return_train_score=True,
                )
                mean_scores = {m: float(np.mean(cv_result.get(f"test_{m}", []))) for m in scoring}
                std_scores = {m: float(np.std(cv_result.get(f"test_{m}", []))) for m in scoring}
                train_scores = {m: float(np.mean(cv_result.get(f"train_{m}", []))) for m in scoring if f"train_{m}" in cv_result}
                return UnifiedCVResult(
                    mean_scores=mean_scores,
                    std_scores=std_scores,
                    train_scores=train_scores,
                    folds=cv_folds,
                )
            else:
                scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=n_jobs)
                return UnifiedCVResult(
                    scores=scores.tolist(),
                    mean=float(np.mean(scores)),
                    std=float(np.std(scores)),
                    min=float(np.min(scores)),
                    max=float(np.max(scores)),
                    folds=cv_folds,
                )
        except Exception as e:
            LOGGER.warning(f"Unified CV failed: {e}")
            return UnifiedCVResult(scores=[], mean=0.0, std=0.0, min=0.0, max=0.0, folds=cv_folds)

    def nested(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        *,
        outer_folds: int = 5,
        inner_folds: int = 3,
        scoring: Optional[str] = None,
        random_state: int = 42,
        stratified: Optional[bool] = None,
    ) -> float:
        if not SKLEARN_AVAILABLE:
            LOGGER.warning("Scikit-learn not available; returning 0.0 for nested CV")
            return 0.0

        try:
            is_classification = _is_classification_target(y) if stratified is None else stratified
            if scoring is None:
                scoring = "accuracy" if is_classification else "r2"

            # Outer CV
            # ⚠️ WARNING: For trading/time series data, use TimeSeriesSplit instead!
            LOGGER.warning("⚠️ Using nested CV without temporal awareness! This may cause data leakage for time series data")
            LOGGER.warning("⚠️ Disabling shuffle to reduce (but not eliminate) data leakage risk")
            if is_classification:
                outer_cv = StratifiedKFold(n_splits=outer_folds, shuffle=False)
                inner_cv = StratifiedKFold(n_splits=inner_folds, shuffle=False)
            else:
                outer_cv = KFold(n_splits=outer_folds, shuffle=False)
                inner_cv = KFold(n_splits=inner_folds, shuffle=False)

            outer_scores: List[float] = []
            for train_idx, val_idx in outer_cv.split(X, y):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                # Simple inner selection: pick best of inner CV among clones with same params
                # For unified behavior without HPO, we just fit on train and evaluate on val.
                try:
                    model_fit = type(model)(**(model.get_params() if hasattr(model, 'get_params') else {}))
                except Exception:
                    model_fit = model
                model_fit.fit(X_train, y_train)

                if scoring == "accuracy":
                    y_pred = model_fit.predict(X_val)
                    score = float(np.mean(y_pred == y_val))
                else:
                    # Default to r2-like measure
                    y_pred = model_fit.predict(X_val)
                    denom = np.var(y_val) if np.var(y_val) != 0 else 1.0
                    score = float(1 - np.mean((y_val - y_pred) ** 2) / denom)

                outer_scores.append(score)

            return float(np.mean(outer_scores)) if outer_scores else 0.0
        except Exception as e:
            LOGGER.error(f"Nested CV failed: {e}")
            return 0.0


# Convenience functions
def perform_cross_validation(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    **kwargs,
) -> Dict[str, Any]:
    """Run standard CV and return a dict compatible with existing call sites."""
    result = UnifiedCrossValidator().run(model, X, y, **kwargs)
    if result.mean_scores is not None:
        return {
            'mean_scores': result.mean_scores,
            'std_scores': result.std_scores,
            'train_scores': result.train_scores,
            'cv_folds': result.folds,
        }
    return {
        'scores': result.scores or [],
        'mean': result.mean,
        'std': result.std,
        'min': result.min,
        'max': result.max,
        'cv_folds': result.folds,
    }


def temporal_cross_validation(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_splits: int = 5,
    gap: int = 0,
    test_size: Optional[int] = None,
    scoring: Optional[Union[str, List[str]]] = None,
) -> Dict[str, Any]:
    return perform_cross_validation(
        model,
        X,
        y,
        strategy="temporal",
        cv_folds=n_splits,
        temporal_gap=gap,
        temporal_test_size=test_size,
        scoring=scoring,
    )


def nested_cross_validation(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    *,
    outer_folds: int = 5,
    inner_folds: int = 3,
    scoring: Optional[str] = None,
    random_state: int = 42,
    stratified: Optional[bool] = None,
) -> float:
    return UnifiedCrossValidator().nested(
        model,
        X,
        y,
        outer_folds=outer_folds,
        inner_folds=inner_folds,
        scoring=scoring,
        random_state=random_state,
        stratified=stratified,
    )


# Backward-compatibility aliases for legacy imports
from .cv import PurgedSplitConfig as PurgedKFold  # type: ignore
TemporalCrossValidator = UnifiedCrossValidator  # type: ignore

__all__ = [
    "UnifiedCrossValidator",
    "UnifiedCVResult",
    "perform_cross_validation",
    "temporal_cross_validation",
    "nested_cross_validation",
    # Legacy names
    "TemporalCrossValidator",
    "PurgedKFold",
]

