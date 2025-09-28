"""PatchTST integration helpers for downstream tree models."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple, Union

import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)

from src.utils.ml_common.models.tree_clvsa_wrapper import PatchTSTTreeConfig, TreePatchTSTWrapper

logger = logging.getLogger(__name__)

TreeModelSpec = Union[str, BaseEstimator, None]


def _resolve_tree_model(
    spec: TreeModelSpec,
    task: str,
    random_state: int,
) -> BaseEstimator:
    if spec is not None and hasattr(spec, "fit"):
        return clone(spec)
    name = (spec or "random_forest").lower()
    if task == "classification":
        if name == "random_forest":
            return RandomForestClassifier(
                n_estimators=400,
                max_depth=8,
                min_samples_leaf=3,
                max_features="sqrt",
                n_jobs=-1,
                random_state=random_state,
            )
        if name == "extra_trees":
            return ExtraTreesClassifier(
                n_estimators=400,
                max_depth=8,
                min_samples_leaf=3,
                max_features="sqrt",
                n_jobs=-1,
                random_state=random_state,
            )
        if name == "xgboost":
            try:
                import xgboost as xgb

                return xgb.XGBClassifier(
                    n_estimators=400,
                    max_depth=5,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.7,
                    random_state=random_state,
                    n_jobs=-1,
                )
            except ImportError:  # pragma: no cover - optional dependency
                logger.warning("XGBoost not available; falling back to RandomForestClassifier")
                return _resolve_tree_model("random_forest", task="classification", random_state=random_state)
        if name == "lightgbm":
            try:
                import lightgbm as lgb

                return lgb.LGBMClassifier(
                    n_estimators=400,
                    max_depth=-1,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.7,
                    random_state=random_state,
                )
            except ImportError:  # pragma: no cover
                logger.warning("LightGBM not available; falling back to RandomForestClassifier")
                return _resolve_tree_model("random_forest", task="classification", random_state=random_state)
        if name == "catboost":
            try:
                from catboost import CatBoostClassifier

                return CatBoostClassifier(
                    iterations=400,
                    depth=6,
                    learning_rate=0.05,
                    random_seed=random_state,
                    verbose=False,
                )
            except ImportError:  # pragma: no cover
                logger.warning("CatBoost not available; falling back to RandomForestClassifier")
                return _resolve_tree_model("random_forest", task="classification", random_state=random_state)
        raise ValueError(f"Unsupported classification tree model: {spec}")
    # Regression defaults
    if name == "random_forest":
        return RandomForestRegressor(
            n_estimators=400,
            max_depth=8,
            min_samples_leaf=3,
            max_features="sqrt",
            n_jobs=-1,
            random_state=random_state,
        )
    if name == "extra_trees":
        return ExtraTreesRegressor(
            n_estimators=400,
            max_depth=8,
            min_samples_leaf=3,
            max_features="sqrt",
            n_jobs=-1,
            random_state=random_state,
        )
    if name == "xgboost":
        try:
            import xgboost as xgb

            return xgb.XGBRegressor(
                n_estimators=400,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.7,
                random_state=random_state,
                n_jobs=-1,
            )
        except ImportError:  # pragma: no cover
            logger.warning("XGBoost not available; falling back to RandomForestRegressor")
            return _resolve_tree_model("random_forest", task="regression", random_state=random_state)
    if name == "lightgbm":
        try:
            import lightgbm as lgb

            return lgb.LGBMRegressor(
                n_estimators=400,
                max_depth=-1,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.7,
                random_state=random_state,
            )
        except ImportError:  # pragma: no cover
            logger.warning("LightGBM not available; falling back to RandomForestRegressor")
            return _resolve_tree_model("random_forest", task="regression", random_state=random_state)
    if name == "catboost":
        try:
            from catboost import CatBoostRegressor

            return CatBoostRegressor(
                iterations=400,
                depth=6,
                learning_rate=0.05,
                random_seed=random_state,
                verbose=False,
            )
        except ImportError:  # pragma: no cover
            logger.warning("CatBoost not available; falling back to RandomForestRegressor")
            return _resolve_tree_model("random_forest", task="regression", random_state=random_state)
    raise ValueError(f"Unsupported regression tree model: {spec}")


class PatchTSTTreeModel(BaseEstimator, RegressorMixin):
    """Convenience estimator wrapping :class:`TreePatchTSTWrapper`."""

    def __init__(
        self,
        tree_model: TreeModelSpec = None,
        classification_model: TreeModelSpec = None,
        patch_config: Optional[PatchTSTTreeConfig] = None,
        task_type: str = "both",
    ) -> None:
        self.tree_model = tree_model
        self.classification_model = classification_model
        self.task_type = task_type
        if patch_config is None:
            self.patch_config = PatchTSTTreeConfig(task_type=task_type)
        else:
            patch_config.task_type = task_type
            self.patch_config = patch_config
        self.wrapper: Optional[TreePatchTSTWrapper] = None
        self.training_summary_: Dict[str, Any] = {}
        self.fitted_: bool = False

    def _build_wrapper(self) -> TreePatchTSTWrapper:
        base_model = _resolve_tree_model(self.tree_model, task="regression", random_state=self.patch_config.random_state)
        cls_model = None
        if self.patch_config.task_type in {"classification", "both"}:
            cls_model = _resolve_tree_model(
                self.classification_model if self.classification_model is not None else self.tree_model,
                task="classification",
                random_state=self.patch_config.random_state,
            )
        return TreePatchTSTWrapper(base_model=base_model, config=self.patch_config, classification_model=cls_model)

    # ------------------------------------------------------------------
    # scikit-learn style API
    # ------------------------------------------------------------------
    def fit(self, X: Union[pd.DataFrame, Any], y: Optional[Any] = None) -> "PatchTSTTreeModel":
        wrapper = self._build_wrapper()
        wrapper.fit(X, y=y)
        self.wrapper = wrapper
        self.training_summary_ = wrapper.training_metadata
        self.fitted_ = True
        return self

    def predict(self, X: Union[pd.DataFrame, Any]) -> pd.DataFrame:
        if not self.fitted_ or self.wrapper is None:
            raise RuntimeError("PatchTSTTreeModel must be fitted before calling predict")
        return self.wrapper.predict(X)

    def predict_direction_proba(self, X: Union[pd.DataFrame, Any]) -> pd.DataFrame:
        if not self.fitted_ or self.wrapper is None:
            raise RuntimeError("PatchTSTTreeModel must be fitted before calling predict_direction_proba")
        return self.wrapper.predict_direction_proba(X)

    def transform(self, X: Union[pd.DataFrame, Any]) -> pd.DataFrame:
        if not self.fitted_ or self.wrapper is None:
            raise RuntimeError("PatchTSTTreeModel must be fitted before calling transform")
        return self.wrapper.transform(X)

    def get_patch_embeddings(self) -> Optional[pd.DataFrame]:
        if not self.fitted_ or self.wrapper is None:
            return None
        return self.wrapper.get_patch_embeddings()

    @property
    def patch_outputs_(self) -> Dict[str, pd.DataFrame]:
        if not self.fitted_ or self.wrapper is None:
            return {}
        return self.wrapper.last_patch_outputs_


def create_default_patchtst_tree_model(
    task_type: str = "both",
    tree_model: TreeModelSpec = None,
    classification_model: TreeModelSpec = None,
    patch_kwargs: Optional[Dict[str, Any]] = None,
) -> PatchTSTTreeModel:
    """Utility helper that constructs a :class:`PatchTSTTreeModel` with sane defaults."""

    patch_kwargs = patch_kwargs or {}
    patch_config = PatchTSTTreeConfig(task_type=task_type, **patch_kwargs)
    return PatchTSTTreeModel(
        tree_model=tree_model,
        classification_model=classification_model,
        patch_config=patch_config,
        task_type=task_type,
    )


# Backwards compatibility alias for older code paths that imported the CVLSA model
HybridCVLSATreeModel = PatchTSTTreeModel

# Additional PatchTST components for compatibility
class PatchTSTFeatureExtractor:
    """PatchTST feature extractor for compatibility."""
    def __init__(self, config=None):
        self.config = config or {}
    
    def extract_features(self, X):
        """Extract features using PatchTST."""
        # This would be implemented with actual PatchTST feature extraction
        return X

def create_patchtst_feature_extractor(config=None):
    """Create PatchTST feature extractor."""
    return PatchTSTFeatureExtractor(config)
