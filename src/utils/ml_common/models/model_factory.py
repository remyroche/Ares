"""Modular model factory implementation.

The previous implementation bundled every model builder, dependency check and
hardware optimisation into a single 2.3k line module.  The audit flagged the
file as unmaintainable and a major contributor to import bloat.  This version
introduces a registry-driven design where individual builders can live in
separate modules while the public API remains backward compatible.
"""

from __future__ import annotations

import importlib
import inspect
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Optional

import numpy as np

_LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public configuration objects
# ---------------------------------------------------------------------------


class ModelType(str, Enum):
    """Enumeration of supported model identifiers.

    The values mirror the legacy implementation so downstream configuration
    files keep working.  Builders can choose to support a subset of the enum
    and surface helpful errors for unsupported variants.
    """

    # Tree-based models
    RANDOM_FOREST = "RandomForestRegressor"
    RANDOM_FOREST_CLASSIFIER = "RandomForestClassifier"
    EXTRA_TREES = "ExtraTreesRegressor"
    EXTRA_TREES_CLASSIFIER = "ExtraTreesClassifier"
    LIGHTGBM = "LGBMRegressor"
    LIGHTGBM_CLASSIFIER = "LGBMClassifier"
    HIST_GRADIENT_BOOSTING = "HistGradientBoostingRegressor"
    HIST_GRADIENT_BOOSTING_CLASSIFIER = "HistGradientBoostingClassifier"
    CATBOOST = "CatBoostRegressor"
    CATBOOST_CLASSIFIER = "CatBoostClassifier"
    XGBOOST = "XGBRegressor"
    XGBOOST_CLASSIFIER = "XGBClassifier"
    XGBOOST_CUSTOM = "XGBoostCustom"
    XGBOOST_META = "XGBoostMeta"

    # Neural network models
    TABNET = "TabNetRegressor"
    TABNET_CLASSIFIER = "TabNetClassifier"
    TABNET_ATTENTION = "TabNetAttention"
    NODE = "NODE"
    NODE_CLASSIFIER = "NODEClassifier"
    TIME_SERIES_TRANSFORMER = "TimeSeriesTransformer"
    TEMPORAL_FUSION_TRANSFORMER = "TemporalFusionTransformer"
    WAVENET = "WaveNet"
    TCN = "TCN"
    LSTM = "LSTM"
    DEEPSCALER = "DeepScaler"
    DEEPSCALER_CLASSIFIER = "DeepScalerClassifier"
    NBEATS = "NBEATS"
    FINANCIAL_RESNET = "FinancialResNet"
    ADVANCED_MAMBA_HYBRID = "AdvancedMambaHybrid"
    DEEPSCALER_1M = "DeepScaler1m"
    CLVSA = "CLVSA"
    MULTISCALE_NBEATS = "MultiScaleNBEATS"
    NAS = "NAS"
    NAS_CLASSIFIER = "NASClassifier"

    # Linear models
    RIDGE = "Ridge"
    RIDGE_CLASSIFIER = "RidgeClassifier"
    ELASTIC_NET = "ElasticNet"
    ELASTIC_NET_CLASSIFIER = "ElasticNetClassifier"
    ELASTIC_NET_CV = "ElasticNetCV"
    ELASTIC_NET_CV_CLASSIFIER = "ElasticNetCVClassifier"
    ELASTIC_NET_QUANTILE = "ElasticNetQuantile"
    QUANTILE_REGRESSION = "QuantileRegression"
    LOGISTIC_REGRESSION = "LogisticRegression"
    LINEAR_REGRESSION = "LinearRegression"
    HUBER_REGRESSION = "HuberRegression"

    # Ensemble models
    VOTING_CLASSIFIER = "VotingClassifier"
    VOTING_REGRESSOR = "VotingRegressor"
    STACKING_CLASSIFIER = "StackingClassifier"
    STACKING_REGRESSOR = "StackingRegressor"
    BAGGING_CLASSIFIER = "BaggingClassifier"
    BAGGING_REGRESSOR = "BaggingRegressor"
    ADABOOST_CLASSIFIER = "AdaBoostClassifier"
    ADABOOST_REGRESSOR = "AdaBoostRegressor"
    GRADIENT_BOOSTING_CLASSIFIER = "GradientBoostingClassifier"
    GRADIENT_BOOSTING_REGRESSOR = "GradientBoostingRegressor"


@dataclass
class ModelConfig:
    model_type: ModelType
    model_name: str
    model_params: Dict[str, Any] = field(default_factory=dict)
    random_state: int = 42
    n_jobs: int = -1
    enable_gpu_acceleration: bool = True
    tags: Dict[str, Any] = field(default_factory=dict)

    def copy_with_overrides(self, **overrides: Any) -> "ModelConfig":
        params = {**self.__dict__, **overrides}
        return ModelConfig(**params)


ModelBuilder = Callable[[ModelConfig], Any]


class ModelBuilderRegistry:
    """Registry that maps ``ModelType`` entries to concrete builders."""

    def __init__(self) -> None:
        self._builders: Dict[ModelType, ModelBuilder] = {}

    def register(self, model_type: ModelType, builder: ModelBuilder) -> None:
        _LOGGER.debug("Registering model builder", extra={"model_type": model_type.value})
        self._builders[model_type] = builder

    def get(self, model_type: ModelType) -> ModelBuilder:
        if model_type in self._builders:
            return self._builders[model_type]

        plugin = _load_plugin(model_type)
        if plugin is not None:
            self._builders[model_type] = plugin
            return plugin

        raise NotImplementedError(
            f"No builder registered for model type '{model_type.value}'. "
            "Register a builder via ModelBuilderRegistry.register or "
            "create a plugin module under 'src.utils.ml_common.models.plugins'."
        )

    def items(self) -> Iterable[tuple[ModelType, ModelBuilder]]:
        return self._builders.items()

    def __contains__(self, model_type: ModelType) -> bool:
        return model_type in self._builders


def _load_plugin(model_type: ModelType) -> Optional[ModelBuilder]:
    module_name = f"src.utils.ml_common.models.plugins.{model_type.name.lower()}"
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError:
        return None

    builder = getattr(module, "build_model", None)
    if callable(builder):
        _LOGGER.debug("Loaded plugin builder", extra={"model_type": model_type.value, "module": module_name})
        return builder  # type: ignore[return-value]
    return None


# ---------------------------------------------------------------------------
# Built-in builders
# ---------------------------------------------------------------------------


def _guard_import(module: str, package_hint: str) -> Any:
    try:
        return importlib.import_module(module)
    except ModuleNotFoundError as exc:  # pragma: no cover - defensive logging
        raise ImportError(
            f"Optional dependency '{package_hint}' is required for this model."
        ) from exc


def _configure_sklearn_model(cls_name: str, default_params: Dict[str, Any], config: ModelConfig) -> Any:
    _guard_import("sklearn", "scikit-learn")
    # Delayed attribute lookup keeps import times down for callers that only
    # need lightweight models.
    target_cls = None
    for submodule in ("ensemble", "linear_model", "tree", "svm"):
        try:
            target_cls = getattr(importlib.import_module(f"sklearn.{submodule}"), cls_name)
            break
        except (AttributeError, ModuleNotFoundError):
            continue
    if target_cls is None:
        raise AttributeError(f"sklearn does not provide class '{cls_name}'")

    params = {**default_params, **config.model_params}
    init_signature = inspect.signature(target_cls.__init__)
    if "random_state" not in params and "random_state" in init_signature.parameters:
        params["random_state"] = config.random_state
    if "n_jobs" not in params and "n_jobs" in init_signature.parameters:
        params["n_jobs"] = config.n_jobs
    return target_cls(**params)


def _build_random_forest(config: ModelConfig) -> Any:
    defaults = {"n_estimators": 500, "n_jobs": config.n_jobs, "random_state": config.random_state}
    cls_name = "RandomForestClassifier" if config.model_type.name.endswith("CLASSIFIER") else "RandomForestRegressor"
    return _configure_sklearn_model(cls_name, defaults, config)


def _build_extra_trees(config: ModelConfig) -> Any:
    defaults = {"n_estimators": 500, "n_jobs": config.n_jobs, "random_state": config.random_state}
    cls_name = "ExtraTreesClassifier" if config.model_type.name.endswith("CLASSIFIER") else "ExtraTreesRegressor"
    return _configure_sklearn_model(cls_name, defaults, config)


def _build_hist_gradient_boosting(config: ModelConfig) -> Any:
    cls_name = (
        "HistGradientBoostingClassifier"
        if config.model_type.name.endswith("CLASSIFIER")
        else "HistGradientBoostingRegressor"
    )
    defaults: Dict[str, Any] = {"random_state": config.random_state}
    return _configure_sklearn_model(cls_name, defaults, config)


def _build_logistic_regression(config: ModelConfig) -> Any:
    defaults = {"max_iter": 1000, "n_jobs": config.n_jobs}
    return _configure_sklearn_model("LogisticRegression", defaults, config)


def _build_linear_regression(config: ModelConfig) -> Any:
    return _configure_sklearn_model("LinearRegression", {}, config)


def _build_ridge(config: ModelConfig) -> Any:
    return _configure_sklearn_model("Ridge", {"alpha": 1.0}, config)


def _build_elastic_net(config: ModelConfig) -> Any:
    defaults = {"l1_ratio": 0.5, "alpha": 0.1}
    cls_name = "ElasticNetCV" if "CV" in config.model_type.name else "ElasticNet"
    return _configure_sklearn_model(cls_name, defaults, config)


def _build_bagging(config: ModelConfig) -> Any:
    cls_name = "BaggingClassifier" if config.model_type.name.endswith("CLASSIFIER") else "BaggingRegressor"
    defaults = {"n_estimators": 50, "random_state": config.random_state}
    return _configure_sklearn_model(cls_name, defaults, config)


def _build_adaboost(config: ModelConfig) -> Any:
    cls_name = "AdaBoostClassifier" if config.model_type.name.endswith("CLASSIFIER") else "AdaBoostRegressor"
    defaults = {"n_estimators": 200, "random_state": config.random_state}
    return _configure_sklearn_model(cls_name, defaults, config)


def _build_gradient_boosting(config: ModelConfig) -> Any:
    cls_name = (
        "GradientBoostingClassifier"
        if config.model_type.name.endswith("CLASSIFIER")
        else "GradientBoostingRegressor"
    )
    defaults = {"random_state": config.random_state}
    return _configure_sklearn_model(cls_name, defaults, config)


def _build_lightgbm(config: ModelConfig) -> Any:
    lgb = _guard_import("lightgbm", "lightgbm")
    cls_name = "LGBMClassifier" if config.model_type.name.endswith("CLASSIFIER") else "LGBMRegressor"
    defaults = {"n_estimators": 500, "random_state": config.random_state}
    params = {**defaults, **config.model_params}
    return getattr(lgb, cls_name)(**params)


def _build_catboost(config: ModelConfig) -> Any:
    catboost = _guard_import("catboost", "catboost")
    cls_name = "CatBoostClassifier" if config.model_type.name.endswith("CLASSIFIER") else "CatBoostRegressor"
    defaults = {
        "iterations": 500,
        "learning_rate": 0.05,
        "depth": 6,
        "random_seed": config.random_state,
        "verbose": False,
    }
    params = {**defaults, **config.model_params}
    return getattr(catboost, cls_name)(**params)


def _build_xgboost(config: ModelConfig) -> Any:
    xgb = _guard_import("xgboost", "xgboost")
    cls_name = "XGBClassifier" if config.model_type.name.endswith("CLASSIFIER") else "XGBRegressor"
    defaults = {
        "n_estimators": 400,
        "learning_rate": 0.05,
        "max_depth": 6,
        "n_jobs": config.n_jobs,
        "random_state": config.random_state,
        "verbosity": 0,
    }
    params = {**defaults, **config.model_params}
    return getattr(xgb, cls_name)(**params)


_BUILTIN_BUILDERS: Dict[ModelType, ModelBuilder] = {
    ModelType.RANDOM_FOREST: _build_random_forest,
    ModelType.RANDOM_FOREST_CLASSIFIER: _build_random_forest,
    ModelType.EXTRA_TREES: _build_extra_trees,
    ModelType.EXTRA_TREES_CLASSIFIER: _build_extra_trees,
    ModelType.HIST_GRADIENT_BOOSTING: _build_hist_gradient_boosting,
    ModelType.HIST_GRADIENT_BOOSTING_CLASSIFIER: _build_hist_gradient_boosting,
    ModelType.LOGISTIC_REGRESSION: _build_logistic_regression,
    ModelType.LINEAR_REGRESSION: _build_linear_regression,
    ModelType.RIDGE: _build_ridge,
    ModelType.RIDGE_CLASSIFIER: _build_ridge,
    ModelType.ELASTIC_NET: _build_elastic_net,
    ModelType.ELASTIC_NET_CLASSIFIER: _build_elastic_net,
    ModelType.ELASTIC_NET_CV: _build_elastic_net,
    ModelType.ELASTIC_NET_CV_CLASSIFIER: _build_elastic_net,
    ModelType.BAGGING_CLASSIFIER: _build_bagging,
    ModelType.BAGGING_REGRESSOR: _build_bagging,
    ModelType.ADABOOST_CLASSIFIER: _build_adaboost,
    ModelType.ADABOOST_REGRESSOR: _build_adaboost,
    ModelType.GRADIENT_BOOSTING_CLASSIFIER: _build_gradient_boosting,
    ModelType.GRADIENT_BOOSTING_REGRESSOR: _build_gradient_boosting,
    ModelType.LIGHTGBM: _build_lightgbm,
    ModelType.LIGHTGBM_CLASSIFIER: _build_lightgbm,
    ModelType.CATBOOST: _build_catboost,
    ModelType.CATBOOST_CLASSIFIER: _build_catboost,
    ModelType.XGBOOST: _build_xgboost,
    ModelType.XGBOOST_CLASSIFIER: _build_xgboost,
}


# ---------------------------------------------------------------------------
# Factory façade
# ---------------------------------------------------------------------------


class EnhancedModelFactory:
    """User-facing factory that delegates to registered builders."""

    def __init__(self, registry: Optional[ModelBuilderRegistry] = None):
        self.registry = registry or ModelBuilderRegistry()
        self._initialise_builtin_builders()

    def _initialise_builtin_builders(self) -> None:
        for model_type, builder in _BUILTIN_BUILDERS.items():
            if model_type not in self.registry:
                self.registry.register(model_type, builder)

    def register_model(self, model_type: ModelType, builder: ModelBuilder) -> None:
        self.registry.register(model_type, builder)

    def available_models(self) -> List[str]:
        return [model_type.value for model_type, _ in self.registry.items()]

    def create_model(self, model_config: ModelConfig) -> Any:
        builder = self.registry.get(model_config.model_type)
        _LOGGER.debug(
            "Creating model",
            extra={"model_type": model_config.model_type.value, "model_name": model_config.model_name},
        )
        model = builder(model_config)
        _attach_feature_name_metadata(model, model_config)
        return model


def _attach_feature_name_metadata(model: Any, config: ModelConfig) -> None:
    feature_names = config.model_params.get("feature_names")
    if feature_names is not None and not hasattr(model, "feature_names_in_"):
        try:
            setattr(model, "feature_names_in_", np.asarray(feature_names))
        except Exception:  # pragma: no cover - best effort metadata attachment
            _LOGGER.debug(
                "Failed to attach feature metadata", extra={"model": type(model).__name__, "feature_count": len(feature_names)}
            )


def create_model_factory(config: Optional[Dict[str, Any]] = None) -> EnhancedModelFactory:
    factory = EnhancedModelFactory()

    if config and config.get("custom_builders"):
        registry = factory.registry
        for model_type_value, dotted_path in config["custom_builders"].items():
            try:
                model_type = ModelType(model_type_value)
            except ValueError:
                _LOGGER.warning("Unknown model type in custom builder config", extra={"model_type": model_type_value})
                continue

            module_name, _, func_name = dotted_path.rpartition(".")
            try:
                module = importlib.import_module(module_name)
                builder = getattr(module, func_name)
            except Exception as exc:  # pragma: no cover - config error handling
                _LOGGER.error(
                    "Failed to load custom model builder", extra={"model_type": model_type_value, "path": dotted_path, "error": exc}
                )
                continue

            if callable(builder):
                registry.register(model_type, builder)  # type: ignore[arg-type]
            else:
                _LOGGER.error(
                    "Configured builder is not callable",
                    extra={"model_type": model_type_value, "path": dotted_path},
                )

    return factory


__all__ = [
    "EnhancedModelFactory",
    "ModelBuilder",
    "ModelBuilderRegistry",
    "ModelConfig",
    "ModelType",
    "create_model_factory",
]
