"""
ML Common - Model Registry for Explainability

This module provides a registry for explainable models and their configurations.
"""

from typing import Any, Dict, List, Optional, Union, Type
from dataclasses import dataclass
from enum import Enum
import logging


class ModelType(Enum):
    """Supported model types for explainability."""
    LINEAR = "linear"
    TREE = "tree"
    NEURAL_NETWORK = "neural_network"
    ENSEMBLE = "ensemble"
    OTHER = "other"


@dataclass
class ExplainabilityConfig:
    """Configuration for model explainability."""
    model_type: ModelType
    explainer_type: str = "auto"
    max_evals: int = 100
    background_samples: int = 100
    feature_names: Optional[List[str]] = None
    categorical_features: Optional[List[str]] = None
    target_names: Optional[List[str]] = None
    random_state: int = 42


class ExplainabilityRegistry:
    """Registry for explainable models and their configurations."""

    def __init__(self):
        self._logger = logging.getLogger(__name__)
        self._model_configs: Dict[str, ExplainabilityConfig] = {}
        self._explainers: Dict[str, Any] = {}
        self._supported_models: Dict[ModelType, List[str]] = {
            ModelType.LINEAR: ["linear", "logistic"],
            ModelType.TREE: ["decision_tree", "random_forest", "xgboost", "lightgbm"],
            ModelType.NEURAL_NETWORK: ["tensorflow", "pytorch", "sklearn_mlp"],
            ModelType.ENSEMBLE: ["voting", "stacking", "bagging"],
            ModelType.OTHER: []
        }

    def register_model(self, model_id: str, config: ExplainabilityConfig) -> None:
        """Register a model with its explainability configuration."""
        self._model_configs[model_id] = config
        self._logger.info(f"Registered model '{model_id}' with type {config.model_type.value}")

    def get_config(self, model_id: str) -> Optional[ExplainabilityConfig]:
        """Get the explainability configuration for a model."""
        return self._model_configs.get(model_id)

    def is_supported(self, model_type: ModelType, model_name: str) -> bool:
        """Check if a model type and name combination is supported."""
        return model_name in self._supported_models.get(model_type, [])

    def get_supported_explainers(self, model_type: ModelType) -> List[str]:
        """Get list of supported explainers for a model type."""
        if model_type == ModelType.LINEAR:
            return ["shap", "lime", "coefficients"]
        elif model_type == ModelType.TREE:
            return ["shap", "tree_explainer", "feature_importance"]
        elif model_type == ModelType.NEURAL_NETWORK:
            return ["shap", "lime", "integrated_gradients"]
        elif model_type == ModelType.ENSEMBLE:
            return ["shap", "lime", "individual_explainer"]
        else:
            return ["lime"]  # Default fallback

    def validate_model_for_explainability(self, model: Any, model_type: ModelType) -> Dict[str, Any]:
        """Validate if a model is suitable for explainability analysis."""
        validation_result = {
            "is_valid": True,
            "warnings": [],
            "recommendations": []
        }

        # Basic validation
        if model is None:
            validation_result["is_valid"] = False
            validation_result["warnings"].append("Model is None")
            return validation_result

        # Model type specific validation
        if model_type == ModelType.LINEAR:
            # Check for sklearn linear models
            if not hasattr(model, 'coef_'):
                validation_result["warnings"].append("Model may not have coefficients for linear explanation")
        elif model_type == ModelType.TREE:
            # Check for tree-based models
            if not hasattr(model, 'feature_importances_'):
                validation_result["warnings"].append("Model may not have feature importances")

        # Check for prediction method
        if not hasattr(model, 'predict'):
            validation_result["warnings"].append("Model does not have predict method")

        # Recommendations
        if len(validation_result["warnings"]) > 0:
            validation_result["recommendations"].append("Consider using LIME as a fallback explainer")

        return validation_result

    def create_default_config(self, model_type: ModelType, model_name: str = "") -> ExplainabilityConfig:
        """Create a default explainability configuration for a model type."""
        return ExplainabilityConfig(
            model_type=model_type,
            explainer_type=self.get_supported_explainers(model_type)[0] if self.get_supported_explainers(model_type) else "lime"
        )


# Global registry instance
_explainability_registry = ExplainabilityRegistry()


def get_explainability_registry() -> ExplainabilityRegistry:
    """Get the global explainability registry instance."""
    return _explainability_registry


def register_model_for_explainability(model_id: str, config: ExplainabilityConfig) -> None:
    """Register a model for explainability analysis."""
    _explainability_registry.register_model(model_id, config)


def get_model_explainability_config(model_id: str) -> Optional[ExplainabilityConfig]:
    """Get the explainability configuration for a model."""
    return _explainability_registry.get_config(model_id)


__all__ = [
    'ExplainabilityRegistry',
    'ExplainabilityConfig',
    'ModelType',
    'get_explainability_registry',
    'register_model_for_explainability',
    'get_model_explainability_config'
]
