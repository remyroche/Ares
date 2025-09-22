"""
Central Feature Selection Bank

This module centralizes feature selection tools and pipelines under
`src/utils/feature_selection/` and delegates heavy-lifting to the
training feature selection framework. It provides a stable API that
other modules can import without depending on disparate implementations.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union, Tuple

import numpy as np
import pandas as pd

# Import the training framework components
from src.training.utils.feature_selection.main_framework import (
    FeatureSelectionFramework as _TrainingFeatureSelectionFramework,
)
from src.training.utils.feature_selection.selection_methods import (
    MRMRSelector,
    ElasticNetStabilitySelector,
    RecursiveFeatureEliminator,
    FeatureImportanceRanker,
)
from src.training.utils.feature_selection.stability_analysis import (
    StabilityAnalyzer,
)


_GLOBAL_FS_FRAMEWORK: Optional[_TrainingFeatureSelectionFramework] = None


def get_feature_selection_framework(config: Optional[Dict[str, Any]] = None) -> _TrainingFeatureSelectionFramework:
    """
    Get a global instance of the training FeatureSelectionFramework to be used as the core engine.
    """
    global _GLOBAL_FS_FRAMEWORK
    if _GLOBAL_FS_FRAMEWORK is None:
        _GLOBAL_FS_FRAMEWORK = _TrainingFeatureSelectionFramework(config)
    return _GLOBAL_FS_FRAMEWORK


def _ensure_feature_names(X: Union[np.ndarray, pd.DataFrame], feature_names: Optional[List[str]]) -> Tuple[np.ndarray, List[str]]:
    if hasattr(X, "values"):
        X_np = X.values  # type: ignore[attr-defined]
        names = feature_names or list(getattr(X, "columns"))  # type: ignore[attr-defined]
    else:
        X_np = np.asarray(X)
        names = feature_names or [f"feature_{i}" for i in range(X_np.shape[1])]
    return X_np, names


def select_features(
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series, List[float], List[int]],
    method: str = "comprehensive",
    max_features: Optional[int] = None,
    is_classification: Optional[bool] = None,
    feature_names: Optional[List[str]] = None,
    framework_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Unified select_features API. Delegates to the training framework.

    Args:
        X: Feature matrix (np.ndarray or pandas DataFrame)
        y: Target vector
        method: Selection method preference (mapped to comprehensive pipeline)
        max_features: Maximum features to aim for
        is_classification: Optional toggle; inferred if not provided
        feature_names: Optional list of feature names
        framework_config: Optional configuration for the underlying framework
    """
    framework = get_feature_selection_framework(framework_config)

    # Normalize inputs
    if isinstance(X, pd.DataFrame):
        names = feature_names or list(X.columns)
    else:
        X = np.asarray(X)
        names = feature_names or [f"feature_{i}" for i in range(np.asarray(X).shape[1])]

    y_arr = np.asarray(y)

    # Map legacy methods to comprehensive pipeline where applicable
    legacy_to_framework = {
        "auto": "comprehensive",
        "filter": "comprehensive",
        "wrapper": "comprehensive",
        "embedded": "comprehensive",
        "hybrid": "comprehensive",
        "comprehensive": "comprehensive",
        "basic": "basic",
        "fast": "fast",
    }
    method_mapped = legacy_to_framework.get(method.lower() if isinstance(method, str) else "comprehensive", "comprehensive")

    # The training framework works directly with DataFrame too
    sel_result = framework.select_features(
        X if isinstance(X, pd.DataFrame) else pd.DataFrame(np.asarray(X), columns=names),
        y_arr,
        method=method_mapped,
        max_features=max_features,
        is_classification=True if is_classification else False,
    )

    # Ensure consistent keys
    if "selected_features" not in sel_result and "final_selected_features" in sel_result:
        sel_result["selected_features"] = sel_result["final_selected_features"]

    sel_result.setdefault("method", method_mapped)
    sel_result.setdefault("total_features", len(names))
    sel_result.setdefault("selected_count", len(sel_result.get("selected_features", [])))
    return sel_result


def run_comprehensive_feature_selection(
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series, List[float], List[int]],
    feature_names: Optional[List[str]] = None,
    target_features: Optional[int] = None,
    model_type: str = "default",
    enable_stability_analysis: bool = True,
    enable_temporal_analysis: bool = False,
    enable_causal_analysis: bool = False,
    enable_pid_analysis: bool = False,
    framework_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Run the comprehensive feature selection pipeline provided by the training framework.
    """
    framework = get_feature_selection_framework(framework_config)
    X_np, names = _ensure_feature_names(X, feature_names)
    y_arr = np.asarray(y)

    return framework.run_comprehensive_feature_selection(
        X_np,
        y_arr,
        names,
        target_features=target_features,
        model_type=model_type,
        enable_stability_analysis=enable_stability_analysis,
        enable_temporal_analysis=enable_temporal_analysis,
        enable_causal_analysis=enable_causal_analysis,
        enable_pid_analysis=enable_pid_analysis,
    )


def lasso_feature_selection(
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series, List[float], List[int]],
    n_features: int,
    feature_names: Optional[List[str]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    LASSO-based feature selection via ElasticNet with l1_ratio=1.0.
    """
    X_np, names = _ensure_feature_names(X, feature_names)
    y_arr = np.asarray(y)

    en_config = dict(config or {})
    en_config.setdefault("l1_ratio_range", (1.0, 1.0))
    selector = ElasticNetStabilitySelector(en_config)
    result = selector.select_features(X_np, y_arr, names)

    # Align response to requested n_features if needed
    if n_features and result.get("selected_features"):
        sel = result["selected_features"][:n_features]
        result["selected_features"] = sel
        result["selected_indices"] = [names.index(f) for f in sel]
    result["method"] = "lasso"
    return result


def cross_validated_feature_selection(
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series, List[float], List[int]],
    n_features: int,
    feature_names: Optional[List[str]] = None,
    cv_folds: int = 5,
    scoring: str = "accuracy",
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Cross-validated feature selection using RFE with configurable CV.
    """
    X_np, names = _ensure_feature_names(X, feature_names)
    y_arr = np.asarray(y)
    rfe = RecursiveFeatureEliminator({"cv": cv_folds, "scoring": scoring, **(config or {})})
    return rfe.select_features(X_np, y_arr, names, n_features)


def hierarchical_feature_selection(
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series, List[float], List[int]],
    n_features: int,
    feature_names: Optional[List[str]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Hierarchical selection: mRMR pre-filter followed by ElasticNet stability refinement.
    """
    X_np, names = _ensure_feature_names(X, feature_names)
    y_arr = np.asarray(y)

    # Step 1: mRMR pre-filter (2x target)
    prefilter_count = min(max(n_features * 2, n_features), X_np.shape[1])
    mrmr = MRMRSelector((config or {}).get("mrmr", {}))
    mrmr_res = mrmr.select_features(X_np, y_arr, names, prefilter_count)

    if not mrmr_res.get("success", False) or not mrmr_res.get("selected_features"):
        # Fallback: return top n_features by feature importance
        fir = FeatureImportanceRanker((config or {}).get("importance", {}))
        return fir.select_features(X_np, y_arr, names, n_features)

    kept_names = mrmr_res["selected_features"]
    kept_indices = [names.index(f) for f in kept_names]
    X_pref = X_np[:, kept_indices]

    # Step 2: ElasticNet stability on prefiltered set
    en_conf = dict((config or {}).get("elastic_net_stability", {}))
    selector = ElasticNetStabilitySelector(en_conf)
    en_res = selector.select_features(X_pref, y_arr, kept_names)

    # Map back indices to original space
    selected = en_res.get("selected_features", [])[:n_features]
    selected_indices = [names.index(f) for f in selected if f in names]

    return {
        "selected_features": selected,
        "selected_indices": selected_indices,
        "method": "hierarchical_mrmr_elasticnet",
        "prefilter": {
            "mrmr_selected": kept_names,
            "mrmr_scores": mrmr_res.get("scores", {}),
        },
        "stability_scores": en_res.get("stability_scores", {}),
        "success": True,
    }


def comprehensive_feature_selection(
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series, List[float], List[int]],
    feature_names: Optional[List[str]] = None,
    target_features: Optional[int] = None,
    framework_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Alias to the comprehensive pipeline.
    """
    return run_comprehensive_feature_selection(
        X,
        y,
        feature_names=feature_names,
        target_features=target_features,
        framework_config=framework_config,
    )


__all__ = [
    "get_feature_selection_framework",
    "select_features",
    "run_comprehensive_feature_selection",
    "lasso_feature_selection",
    "cross_validated_feature_selection",
    "hierarchical_feature_selection",
    "comprehensive_feature_selection",
    # Expose key selectors/analyzers for advanced users
    "MRMRSelector",
    "ElasticNetStabilitySelector",
    "RecursiveFeatureEliminator",
    "FeatureImportanceRanker",
    "StabilityAnalyzer",
]

