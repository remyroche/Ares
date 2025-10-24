"""
SHAP and LIME Integration for Multi-Output Stacking Ensemble Models

This module provides comprehensive explainability integration for the multi-output
stacking ensemble system, including SHAP and LIME explanations at each training step.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import logging
import time
from datetime import datetime

# SHAP imports
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    shap = None

# LIME imports
try:
    import lime
    import lime.tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    lime = None

from src.utils.logger import system_logger
from src.utils.tprint import tprint_data_format, LogLevel

logger = system_logger.getChild('SHAPLIMEIntegration')

@dataclass
class ExplanationConfig:
    """Configuration for model explanations."""
    # SHAP configuration
    enable_shap: bool = True
    shap_explainer_type: str = "tree"  # auto, tree, linear, kernel, deep
    shap_sample_size: int = 100
    shap_max_features: int = 50

    # LIME configuration
    enable_lime: bool = True
    lime_sample_size: int = 1000
    lime_num_features: int = 10
    lime_discretize_continuous: bool = True

    # Multi-output configuration
    explain_all_outputs: bool = True
    output_names: Optional[List[str]] = None

    # Performance configuration
    enable_caching: bool = True
    cache_size_mb: int = 100
    parallel_explanations: bool = True
    max_workers: int = 4

@dataclass
class ExplanationResult:
    """Result of model explanation."""
    # SHAP results
    shap_values: Optional[Dict[str, np.ndarray]] = None
    shap_base_values: Optional[Dict[str, float]] = None
    shap_feature_names: Optional[List[str]] = None
    shap_explainer: Optional[Any] = None

    # LIME results
    lime_explanations: Optional[Dict[str, Any]] = None
    lime_feature_names: Optional[List[str]] = None

    # Metadata
    explanation_time: float = 0.0
    model_name: str = ""
    output_names: List[str] = None
    feature_names: List[str] = None
    sample_size: int = 0

class SHAPLIMEExplainer:
    """
    Comprehensive explainability integration for multi-output stacking ensemble models.

    This class provides SHAP and LIME explanations for both base models and meta-models
    in the stacking ensemble, with support for multi-output predictions.
    """

    def __init__(self, config: Optional[ExplanationConfig] = None):
        """Initialize the SHAP/LIME explainer."""
        self.config = config or ExplanationConfig()
        self.logger = logger.getChild('SHAPLIMEExplainer')

        # Validate dependencies
        self._validate_dependencies()

        # Initialize explainers
        self.shap_explainers = {}
        self.lime_explainers = {}

        # Cache for explanations
        self.explanation_cache = {}

        self.logger.info("✅ SHAP/LIME Explainer initialized")

    def _validate_dependencies(self):
        """Validate that required dependencies are available."""
        if self.config.enable_shap and not SHAP_AVAILABLE:
            self.logger.warning("⚠️ SHAP not available, disabling SHAP explanations")
            self.config.enable_shap = False

        if self.config.enable_lime and not LIME_AVAILABLE:
            self.logger.warning("⚠️ LIME not available, disabling LIME explanations")
            self.config.enable_lime = False

        if not self.config.enable_shap and not self.config.enable_lime:
            self.logger.warning("⚠️ No explainability methods available")

    def explain_model(
        self,
        model: Any,
        X: np.ndarray,
        model_name: str,
        output_names: Optional[List[str]] = None,
        feature_names: Optional[List[str]] = None
    ) -> ExplanationResult:
        """
        Generate explanations for a single model.

        Args:
            model: The model to explain
            X: Input features
            model_name: Name of the model
            output_names: Names of outputs (for multi-output models)
            feature_names: Names of input features

        Returns:
            ExplanationResult object
        """
        self.logger.info(f"🔄 Generating explanations for {model_name}")
        start_time = time.time()

        # Check cache first
        cache_key = self._get_cache_key(model_name, X)
        if self.config.enable_caching and cache_key in self.explanation_cache:
            self.logger.debug("📋 Using cached explanation")
            return self.explanation_cache[cache_key]

        # Prepare data
        X_sample = self._prepare_sample(X)
        feature_names = feature_names or [f"feature_{i}" for i in range(X.shape[1])]
        output_names = output_names or ["output"]

        result = ExplanationResult(
            model_name=model_name,
            output_names=output_names,
            feature_names=feature_names,
            sample_size=len(X_sample)
        )

        # Generate SHAP explanations
        if self.config.enable_shap:
            try:
                shap_result = self._generate_shap_explanations(
                    model, X_sample, model_name, output_names, feature_names
                )
                result.shap_values = shap_result['values']
                result.shap_base_values = shap_result['base_values']
                result.shap_feature_names = shap_result['feature_names']
                result.shap_explainer = shap_result['explainer']
                self.logger.debug("✅ SHAP explanations generated")
            except Exception as e:
                self.logger.warning(f"⚠️ SHAP explanation failed: {e}")

        # Generate LIME explanations
        if self.config.enable_lime:
            try:
                lime_result = self._generate_lime_explanations(
                    model, X_sample, model_name, output_names, feature_names
                )
                result.lime_explanations = lime_result['explanations']
                result.lime_feature_names = lime_result['feature_names']
                self.logger.debug("✅ LIME explanations generated")
            except Exception as e:
                self.logger.warning(f"⚠️ LIME explanation failed: {e}")

        # Calculate explanation time
        result.explanation_time = time.time() - start_time

        # Cache result
        if self.config.enable_caching:
            self.explanation_cache[cache_key] = result

        self.logger.info(f"✅ Explanations generated for {model_name} in {result.explanation_time:.3f}s")
        return result

    def explain_stacking_ensemble(
        self,
        base_models: Dict[str, Any],
        meta_model: Any,
        X: np.ndarray,
        output_names: List[str],
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, ExplanationResult]:
        """
        Generate explanations for a complete stacking ensemble.

        Args:
            base_models: Dictionary of base models
            meta_model: Meta-model for stacking
            X: Input features
            output_names: Names of outputs
            feature_names: Names of input features

        Returns:
            Dictionary of explanation results for each model
        """
        self.logger.info("🔄 Generating explanations for stacking ensemble")
        start_time = time.time()

        results = {}
        feature_names = feature_names or [f"feature_{i}" for i in range(X.shape[1])]

        # Explain base models
        for model_name, model in base_models.items():
            self.logger.debug(f"🔄 Explaining base model: {model_name}")
            results[model_name] = self.explain_model(
                model, X, model_name, output_names, feature_names
            )

        # Generate base model predictions for meta-model
        base_predictions = self._generate_base_predictions(base_models, X)

        # Create meta-features (original features + base model predictions)
        meta_features = np.hstack([X, base_predictions])
        meta_feature_names = feature_names + [f"{name}_pred" for name in base_models.keys()]

        # Explain meta-model
        self.logger.debug("🔄 Explaining meta-model")
        results['meta_model'] = self.explain_model(
            meta_model, meta_features, 'meta_model', output_names, meta_feature_names
        )

        total_time = time.time() - start_time
        self.logger.info(f"✅ Stacking ensemble explanations generated in {total_time:.3f}s")

        return results

    def _generate_shap_explanations(
        self,
        model: Any,
        X: np.ndarray,
        model_name: str,
        output_names: List[str],
        feature_names: List[str]
    ) -> Dict[str, Any]:
        """Generate SHAP explanations for a model."""

        # Select appropriate explainer
        explainer = self._get_shap_explainer(model, X, model_name)

        # Generate SHAP values
        if hasattr(explainer, 'shap_values'):
            # For models that support shap_values directly
            shap_values = explainer.shap_values(X)
        else:
            # For other models, use the explainer
            shap_values = explainer(X)

        # Handle multi-output case
        if isinstance(shap_values, list):
            # Multi-output model
            shap_dict = {}
            base_values = {}
            for i, output_name in enumerate(output_names):
                if i < len(shap_values):
                    shap_dict[output_name] = shap_values[i]
                    if hasattr(explainer, 'expected_value'):
                        base_values[output_name] = explainer.expected_value[i] if isinstance(explainer.expected_value, list) else explainer.expected_value
        else:
            # Single output model
            shap_dict = {output_names[0]: shap_values}
            base_values = {output_names[0]: explainer.expected_value if hasattr(explainer, 'expected_value') else 0.0}

        return {
            'values': shap_dict,
            'base_values': base_values,
            'feature_names': feature_names,
            'explainer': explainer
        }

    def _generate_lime_explanations(
        self,
        model: Any,
        X: np.ndarray,
        model_name: str,
        output_names: List[str],
        feature_names: List[str]
    ) -> Dict[str, Any]:
        """Generate LIME explanations for a model."""

        # Create LIME explainer
        explainer = lime.tabular.LimeTabularExplainer(
            X,
            feature_names=feature_names,
            discretize_continuous=self.config.lime_discretize_continuous,
            random_state=42
        )

        # Generate explanations for each output
        explanations = {}
        for i, output_name in enumerate(output_names):
            # Create prediction function for this output
            def predict_fn(X_input):
                if hasattr(model, 'predict_proba'):
                    pred = model.predict_proba(X_input)
                    if pred.ndim > 1 and pred.shape[1] > i:
                        return pred[:, i]
                    else:
                        return pred
                else:
                    pred = model.predict(X_input)
                    if pred.ndim > 1 and pred.shape[1] > i:
                        return pred[:, i]
                    else:
                        return pred

            # Generate explanation for a sample
            sample_idx = 0
            explanation = explainer.explain_instance(
                X[sample_idx],
                predict_fn,
                num_features=self.config.lime_num_features
            )

            explanations[output_name] = explanation

        return {
            'explanations': explanations,
            'feature_names': feature_names
        }

    def _get_shap_explainer(self, model: Any, X: np.ndarray, model_name: str) -> Any:
        """Get appropriate SHAP explainer for the model."""

        if model_name in self.shap_explainers:
            return self.shap_explainers[model_name]

        # Determine explainer type
        explainer_type = self.config.shap_explainer_type

        if explainer_type == "auto":
            # Auto-detect based on model type - prefer TreeSHAP when possible
            if hasattr(model, 'predict_proba') or hasattr(model, 'predict'):
                explainer_type = "tree"
            elif hasattr(model, 'coef_'):
                explainer_type = "linear"
            else:
                # Fallback to TreeSHAP instead of KernelSHAP for better performance
                explainer_type = "tree"

        # Create explainer
        if explainer_type == "tree":
            explainer = shap.TreeExplainer(model)
        elif explainer_type == "linear":
            explainer = shap.LinearExplainer(model, X)
        elif explainer_type == "kernel":
            explainer = shap.KernelExplainer(model.predict, X[:self.config.shap_sample_size])
        else:
            explainer = shap.Explainer(model, X[:self.config.shap_sample_size])

        # Cache explainer
        self.shap_explainers[model_name] = explainer

        return explainer

    def _generate_base_predictions(self, base_models: Dict[str, Any], X: np.ndarray) -> np.ndarray:
        """Generate predictions from all base models."""
        predictions = []

        for model_name, model in base_models.items():
            try:
                pred = model.predict(X)
                if pred.ndim == 1:
                    pred = pred.reshape(-1, 1)
                predictions.append(pred)
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to get predictions from {model_name}: {e}")
                # Add zero predictions as fallback
                predictions.append(np.zeros((len(X), 1)))

        return np.hstack(predictions)

    def _prepare_sample(self, X: np.ndarray) -> np.ndarray:
        """Prepare a sample of the data for explanation."""
        sample_size = min(self.config.shap_sample_size, len(X))
        if sample_size < len(X):
            indices = np.random.choice(len(X), sample_size, replace=False)
            return X[indices]
        return X

    def _get_cache_key(self, model_name: str, X: np.ndarray) -> str:
        """Generate cache key for explanations."""
        return f"{model_name}_{hash(X.tobytes())}"

    def clear_cache(self) -> None:
        """Clear explanation cache."""
        self.explanation_cache.clear()
        self.logger.info("🗑️ Explanation cache cleared")

    def get_explanation_summary(self, result: ExplanationResult) -> Dict[str, Any]:
        """Get a summary of explanation results."""
        summary = {
            'model_name': result.model_name,
            'explanation_time': result.explanation_time,
            'sample_size': result.sample_size,
            'output_names': result.output_names,
            'feature_names': result.feature_names,
            'has_shap': result.shap_values is not None,
            'has_lime': result.lime_explanations is not None
        }

        if result.shap_values:
            summary['shap_outputs'] = list(result.shap_values.keys())
            summary['shap_feature_count'] = len(result.shap_feature_names) if result.shap_feature_names else 0

        if result.lime_explanations:
            summary['lime_outputs'] = list(result.lime_explanations.keys())
            summary['lime_feature_count'] = len(result.lime_feature_names) if result.lime_feature_names else 0

        return summary

# Convenience functions
def create_explainer(config: Optional[ExplanationConfig] = None) -> SHAPLIMEExplainer:
    """Create a SHAP/LIME explainer instance."""
    return SHAPLIMEExplainer(config)

def explain_model(
    model: Any,
    X: np.ndarray,
    model_name: str,
    config: Optional[ExplanationConfig] = None,
    output_names: Optional[List[str]] = None,
    feature_names: Optional[List[str]] = None
) -> ExplanationResult:
    """Generate explanations for a single model."""
    explainer = create_explainer(config)
    return explainer.explain_model(model, X, model_name, output_names, feature_names)

def explain_stacking_ensemble(
    base_models: Dict[str, Any],
    meta_model: Any,
    X: np.ndarray,
    output_names: List[str],
    config: Optional[ExplanationConfig] = None,
    feature_names: Optional[List[str]] = None
) -> Dict[str, ExplanationResult]:
    """Generate explanations for a complete stacking ensemble."""
    explainer = create_explainer(config)
    return explainer.explain_stacking_ensemble(
        base_models, meta_model, X, output_names, feature_names
    )
