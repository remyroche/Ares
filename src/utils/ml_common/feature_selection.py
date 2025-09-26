"""Modular entry points for ML-common feature selection tooling.

This module used to house an 8k-line monolith that re-implemented the
training feature-selection stack.  The audit highlighted that the file was
unmaintainable and duplicated logic that already lives in the dedicated
`src.utils.feature_selection` package.  The implementation below keeps the
public surface area stable while delegating the heavy lifting to the shared
framework.  Doing so eliminates the circular import pressure and makes the
module fast to import.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from src.utils.feature_selection import (
    cross_validated_feature_selection as _training_cross_validated_selection,
    get_feature_selection_framework as _get_training_framework,
    hierarchical_feature_selection as _training_hierarchical_selection,
    lasso_feature_selection as _training_lasso_selection,
    run_comprehensive_feature_selection as _training_run_comprehensive,
    select_features as _training_select_features,
)
from .feature_selection_backwards_compat import (
    FeatureSelectionConfig,
    FeatureSelector as LegacyFeatureSelector,
    create_feature_selector,
    select_features as legacy_select_features,
)

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class FeatureSelectionResult:
    """Lightweight container mirroring the legacy dictionary contract."""

    selected_features: List[str]
    feature_scores: Dict[str, float]
    diagnostics: Dict[str, Any]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "selected_features": list(self.selected_features),
            "feature_scores": dict(self.feature_scores),
            "diagnostics": dict(self.diagnostics),
        }


class FeatureSelectionFramework:
    """Thin adapter around the shared training FeatureSelectionFramework."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self._config = config or {}
        self._framework = _get_training_framework(self._config)
        self.logger = _LOGGER.getChild("FeatureSelectionFramework")
        self.logger.debug("Initialised FeatureSelectionFramework adapter", extra={"config": self._config})

    # ------------------------------------------------------------------
    # Delegated comprehensive routines
    # ------------------------------------------------------------------
    def run_comprehensive_feature_selection(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray, Sequence[float], Sequence[int]],
        feature_names: Optional[Sequence[str]] = None,
        target_features: Optional[int] = None,
        model_type: str = "default",
        enable_stability_analysis: bool = True,
        enable_temporal_analysis: bool = False,
        enable_causal_analysis: bool = False,
        enable_pid_analysis: bool = False,
    ) -> Dict[str, Any]:
        self.logger.debug(
            "Delegating comprehensive feature selection",
            extra={
                "target_features": target_features,
                "model_type": model_type,
                "stability": enable_stability_analysis,
                "temporal": enable_temporal_analysis,
                "causal": enable_causal_analysis,
                "pid": enable_pid_analysis,
            },
        )
        return _training_run_comprehensive(
            X,
            y,
            feature_names=list(feature_names) if feature_names is not None else None,
            target_features=target_features,
            model_type=model_type,
            enable_stability_analysis=enable_stability_analysis,
            enable_temporal_analysis=enable_temporal_analysis,
            enable_causal_analysis=enable_causal_analysis,
            enable_pid_analysis=enable_pid_analysis,
            framework_config=self._config,
        )

    def select_features(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray, Sequence[float], Sequence[int]],
        method: str = "auto",
        task_type: str = "regression",
        max_features: Optional[int] = None,
        feature_names: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        is_classification = task_type.lower().startswith("class")
        X_array, inferred_names = _normalize_features(X, feature_names)
        result = _training_select_features(
            X_array,
            np.asarray(y),
            method=method,
            max_features=max_features,
            is_classification=is_classification,
            feature_names=inferred_names,
            framework_config=self._config,
        )

        selected = result.get("selected_features", [])
        scores = result.get("final_scores") or result.get("feature_scores", {})
        diagnostics = {
            k: v
            for k, v in result.items()
            if k not in {"selected_features", "final_scores", "feature_scores"}
        }

        payload = FeatureSelectionResult(selected, scores, diagnostics).as_dict()
        payload.update({
            "method": result.get("method", method),
            "task_type": task_type,
            "total_features": len(inferred_names),
        })
        return payload

    def lasso_feature_selection(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray, Sequence[float], Sequence[int]],
        n_features: int,
        feature_names: Optional[Sequence[str]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return _training_lasso_selection(
            X,
            y,
            n_features,
            feature_names=list(feature_names) if feature_names is not None else None,
            config=config,
        )

    def cross_validated_feature_selection(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray, Sequence[float], Sequence[int]],
        n_features: int,
        feature_names: Optional[Sequence[str]] = None,
        cv_folds: int = 5,
        scoring: str = "accuracy",
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return _training_cross_validated_selection(
            X,
            y,
            n_features,
            feature_names=list(feature_names) if feature_names is not None else None,
            cv_folds=cv_folds,
            scoring=scoring,
            config=config,
        )

    def hierarchical_feature_selection(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray, Sequence[float], Sequence[int]],
        n_features: int,
        feature_names: Optional[Sequence[str]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return _training_hierarchical_selection(
            X,
            y,
            n_features,
            feature_names=list(feature_names) if feature_names is not None else None,
            config=config,
        )


# ----------------------------------------------------------------------
# Module-level convenience wrappers mirroring the previous dictionary API
# ----------------------------------------------------------------------

def _normalize_features(
    X: Union[pd.DataFrame, np.ndarray],
    feature_names: Optional[Sequence[str]] = None,
) -> Tuple[np.ndarray, List[str]]:
    if isinstance(X, pd.DataFrame):
        names = list(feature_names or X.columns)
        return X.values, names  # type: ignore[attr-defined]

    X_arr = np.asarray(X)
    names = list(feature_names or [f"feature_{i}" for i in range(X_arr.shape[1])])
    return X_arr, names


def select_features(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray, Sequence[float], Sequence[int]],
    method: str = "auto",
    task_type: str = "regression",
    max_features: Optional[int] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    framework = FeatureSelectionFramework(config)
    return framework.select_features(
        X,
        y,
        method=method,
        task_type=task_type,
        max_features=max_features,
    )


def run_comprehensive_feature_selection(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray, Sequence[float], Sequence[int]],
    feature_names: Optional[Sequence[str]] = None,
    target_features: Optional[int] = None,
    model_type: str = "default",
    enable_stability_analysis: bool = True,
    enable_temporal_analysis: bool = False,
    enable_causal_analysis: bool = False,
    enable_pid_analysis: bool = False,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    framework = FeatureSelectionFramework(config)
    return framework.run_comprehensive_feature_selection(
        X,
        y,
        feature_names=feature_names,
        target_features=target_features,
        model_type=model_type,
        enable_stability_analysis=enable_stability_analysis,
        enable_temporal_analysis=enable_temporal_analysis,
        enable_causal_analysis=enable_causal_analysis,
        enable_pid_analysis=enable_pid_analysis,
    )


def get_feature_importance(model: Any) -> Dict[str, float]:
    """Best-effort extraction of feature importance values."""

    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_)
    elif hasattr(model, "coef_"):
        values = np.abs(np.asarray(model.coef_).reshape(-1))
    else:
        _LOGGER.debug("Model exposes no native importance attributes", extra={"model": type(model).__name__})
        return {}

    names = getattr(model, "feature_names_in_", [f"feature_{i}" for i in range(len(values))])
    return {name: float(val) for name, val in zip(names, values)}


def get_feature_selection_framework(config: Optional[Dict[str, Any]] = None) -> FeatureSelectionFramework:
    return FeatureSelectionFramework(config)


# Backwards-compatible exports -------------------------------------------------
FeatureSelector = LegacyFeatureSelector

__all__ = [
    "FeatureSelectionConfig",
    "FeatureSelectionFramework",
    "FeatureSelectionResult",
    "FeatureSelector",
    "create_feature_selector",
    "get_feature_importance",
    "get_feature_selection_framework",
    "legacy_select_features",
    "run_comprehensive_feature_selection",
    "select_features",
]
