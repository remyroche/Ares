from src.utils.tprint import tprint, tprint_data_format, LogLevel

"""
Model-Focused Explainability Integration for ML Commons

This module provides comprehensive model explainability that integrates seamlessly
with the ML commons training and model registry system. Instead of component-specific
explainers (tactician/analyst/SR/HMM), this focuses on individual ML models and
automatically provides explanations when models are fetched or trained.

Key Features:
- Automatic explainability integration with model training
- Model-specific explainers that adapt to model types
- Integration with ModelRegistry for persistent explanations
- SHAP and LIME support with fallback mechanisms
- Comprehensive explanation caching and retrieval
- Integration with existing ML commons utilities

Built on existing utilities:
- Uses model_explanations.py for SHAP/LIME integration
- Integrates with model_registry.py for persistence
- Leverages model_training.py for automatic integration
- Builds on existing explanation patterns
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import time
import logging
from datetime import datetime
from pathlib import Path
import json
import pickle
from dataclasses import dataclass, asdict

from .model_explanations import ModelExplainer, explain_model_with_shap_lime
from ..models.model_registry import ModelRegistry
from ..common_operations import create_fallback_logger, safe_json_dump, safe_json_load

# Enhanced dependency management with fast fail
try:
    from ..logger import get_logger
    _LOGGER = get_logger("MLCommon.ModelExplainability")
    tprint("✅ Custom logger available for MLCommon.ModelExplainability")
except Exception as e:
    tprint(f"⚠️ Custom logger not available: {e}. Using standard logging.")
    _LOGGER = logging.getLogger("MLCommon.ModelExplainability")
    _LOGGER.setLevel(logging.INFO)

logger = _LOGGER

@dataclass
class ModelExplanationResult:
    """Comprehensive model explanation result."""
    model_id: str
    model_type: str
    model_name: str
    prediction: Any
    feature_names: List[str]
    feature_values: np.ndarray
    shap_values: Optional[np.ndarray] = None
    lime_explanation: Optional[Dict[str, Any]] = None
    feature_importance: Optional[Dict[str, float]] = None
    confidence: float = 0.0
    explanation_confidence: float = 0.0
    processing_time_ms: float = 0.0
    timestamp: datetime = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = asdict(self)
        # Convert numpy arrays to lists for JSON serialization
        if isinstance(result['feature_values'], np.ndarray):
            result['feature_values'] = result['feature_values'].tolist()
        if isinstance(result['shap_values'], np.ndarray):
            result['shap_values'] = result['shap_values'].tolist()
        if isinstance(result['timestamp'], datetime):
            result['timestamp'] = result['timestamp'].isoformat()
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelExplanationResult':
        """Create from dictionary."""
        # Convert lists back to numpy arrays
        if 'feature_values' in data and isinstance(data['feature_values'], list):
            data['feature_values'] = np.array(data['feature_values'])
        if 'shap_values' in data and isinstance(data['shap_values'], list):
            data['shap_values'] = np.array(data['shap_values'])
        if 'timestamp' in data and isinstance(data['timestamp'], str):
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)

class ModelExplainabilityManager:
    """Manager for model explainability integrated with ML commons."""

    def __init__(self, config: Optional[Dict[str, Any]] = None,
                 model_registry: Optional[ModelRegistry] = None):
        """
        Initialize the model explainability manager.

        Args:
            config: Configuration dictionary for explainability parameters
            model_registry: Optional ModelRegistry instance for persistence
        """
        self.config = config or {}
        self.logger = create_fallback_logger()

        _LOGGER.info("🚀 Initializing ModelExplainabilityManager...")

        # Initialize model explainer
        self.explainer = ModelExplainer(self.config.get('explanations', {}))

        # Initialize model registry integration
        self.model_registry = model_registry
        if self.model_registry is None:
            _LOGGER.info("🔧 Initializing default ModelRegistry...")
            self.model_registry = ModelRegistry(
                registry_path=self.config.get('registry_path', './model_registry'),
                config=self.config.get('registry', {})
            )

        # Configuration
        self.enable_auto_explanations = self.config.get('enable_auto_explanations', True)
        self.enable_explanation_caching = self.config.get('enable_explanation_caching', True)
        self.explanation_cache_size = self.config.get('explanation_cache_size', 1000)
        self.auto_explain_on_training = self.config.get('auto_explain_on_training', True)
        self.auto_explain_on_prediction = self.config.get('auto_explain_on_prediction', False)

        # Explanation cache
        self.explanation_cache: Dict[str, ModelExplanationResult] = {}
        self.cache_hits = 0
        self.cache_misses = 0

        _LOGGER.info(f"⚙️ Configuration - Auto explanations: {self.enable_auto_explanations}")
        _LOGGER.info(f"⚙️ Configuration - Explanation caching: {self.enable_explanation_caching}")
        _LOGGER.info(f"⚙️ Configuration - Auto explain on training: {self.auto_explain_on_training}")
        _LOGGER.info(f"⚙️ Configuration - Auto explain on prediction: {self.auto_explain_on_prediction}")

        _LOGGER.info("✅ ModelExplainabilityManager initialized successfully")

    def explain_model(self, model: Any, X_train: np.ndarray, X_test: np.ndarray,
                     model_id: str, model_type: str = "unknown",
                     feature_names: Optional[List[str]] = None,
                     cache_key: Optional[str] = None) -> ModelExplanationResult:
        """
        Generate comprehensive model explanations.

        Args:
            model: The trained model to explain
            X_train: Training features
            X_test: Test features
            model_id: Unique model identifier
            model_type: Type of model (e.g., 'random_forest', 'neural_network')
            feature_names: List of feature names
            cache_key: Optional cache key for explanation caching

        Returns:
            ModelExplanationResult with comprehensive explanations
        """
        start_time = time.time()
        _LOGGER.info(f"🔍 Generating explanations for model: {model_id}")
        _LOGGER.info(f"📊 Model type: {model_type}")
        _LOGGER.info(f"📊 Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")

        # Check cache first
        if self.enable_explanation_caching and cache_key:
            cached_explanation = self._get_cached_explanation(cache_key)
            if cached_explanation:
                _LOGGER.info(f"📋 Using cached explanation for {model_id}")
                return cached_explanation

        try:
            # Generate explanations using existing ModelExplainer
            explanation_data = self.explainer.explain_model(
                model=model,
                X_train=X_train,
                X_test=X_test,
                feature_names=feature_names,
                model_name=model_id
            )

            # Create comprehensive result
            result = ModelExplanationResult(
                model_id=model_id,
                model_type=model_type,
                model_name=model_id,
                prediction=explanation_data.get('predictions', []),
                feature_names=explanation_data.get('feature_names', []),
                feature_values=explanation_data.get('feature_values', np.array([])),
                shap_values=explanation_data.get('shap_values'),
                lime_explanation=explanation_data.get('lime_explanation'),
                feature_importance=explanation_data.get('feature_importance'),
                confidence=explanation_data.get('confidence', 0.0),
                explanation_confidence=explanation_data.get('explanation_confidence', 0.0),
                processing_time_ms=(time.time() - start_time) * 1000,
                metadata={
                    'model_type': model_type,
                    'training_samples': X_train.shape[0],
                    'test_samples': X_test.shape[0],
                    'feature_count': len(feature_names) if feature_names else X_train.shape[1],
                    'explanation_methods': explanation_data.get('methods_used', []),
                    'cache_key': cache_key
                }
            )

            # Cache the result
            if self.enable_explanation_caching and cache_key:
                self._cache_explanation(cache_key, result)

            # Save to model registry if available
            if self.model_registry:
                self._save_explanation_to_registry(result)

            execution_time = time.time() - start_time
            _LOGGER.info(f"✅ Model explanations generated in {execution_time:.3f}s")
            _LOGGER.info(f"📊 Explanation confidence: {result.explanation_confidence:.3f}")

            return result

        except Exception as e:
            _LOGGER.error(f"❌ Error generating model explanations: {e}")
            # Return minimal result on error
            return ModelExplanationResult(
                model_id=model_id,
                model_type=model_type,
                model_name=model_id,
                prediction=[],
                feature_names=feature_names or [],
                feature_values=np.array([]),
                confidence=0.0,
                explanation_confidence=0.0,
                processing_time_ms=(time.time() - start_time) * 1000,
                metadata={'error': str(e), 'cache_key': cache_key}
            )

    def get_model_explanation(self, model_id: str, version: str = 'latest') -> Optional[ModelExplanationResult]:
        """
        Retrieve stored model explanation from registry.

        Args:
            model_id: Model identifier
            version: Model version

        Returns:
            ModelExplanationResult if found, None otherwise
        """
        _LOGGER.info(f"📂 Retrieving explanation for model: {model_id} (version: {version})")

        try:
            if not self.model_registry:
                _LOGGER.warning("⚠️ No model registry available for explanation retrieval")
                return None

            # Try to load explanation from registry
            explanation_path = self.model_registry.registry_path / model_id / version / "explanation.json"

            if explanation_path.exists():
                _LOGGER.info(f"📂 Loading explanation from: {explanation_path}")
                explanation_data = safe_json_load(explanation_path)
                result = ModelExplanationResult.from_dict(explanation_data)
                _LOGGER.info(f"✅ Explanation loaded successfully")
                return result
            else:
                _LOGGER.info(f"📂 No explanation found for model {model_id} version {version}")
                return None

        except Exception as e:
            _LOGGER.error(f"❌ Error retrieving model explanation: {e}")
            return None

    def explain_prediction(self, model: Any, X: np.ndarray, model_id: str,
                          feature_names: Optional[List[str]] = None) -> ModelExplanationResult:
        """
        Generate explanation for a single prediction.

        Args:
            model: The trained model
            X: Input features for prediction
            model_id: Model identifier
            feature_names: List of feature names

        Returns:
            ModelExplanationResult for the prediction
        """
        _LOGGER.info(f"🔍 Generating prediction explanation for model: {model_id}")
        _LOGGER.info(f"📊 Input shape: {X.shape}")

        # For single prediction, we need some training data for SHAP/LIME
        # This is a limitation - we'll use the input as both train and test
        return self.explain_model(
            model=model,
            X_train=X,
            X_test=X,
            model_id=model_id,
            model_type="prediction",
            feature_names=feature_names,
            cache_key=f"prediction_{model_id}_{hash(X.tobytes())}"
        )

    def _get_cached_explanation(self, cache_key: str) -> Optional[ModelExplanationResult]:
        """Get explanation from cache."""
        if cache_key in self.explanation_cache:
            self.cache_hits += 1
            return self.explanation_cache[cache_key]
        self.cache_misses += 1
        return None

    def _cache_explanation(self, cache_key: str, explanation: ModelExplanationResult) -> None:
        """Cache explanation result."""
        # Simple LRU-style cache management
        if len(self.explanation_cache) >= self.explanation_cache_size:
            # Remove oldest entry (simple implementation)
            oldest_key = next(iter(self.explanation_cache))
            del self.explanation_cache[oldest_key]

        self.explanation_cache[cache_key] = explanation
        _LOGGER.debug(f"📋 Cached explanation with key: {cache_key}")

    def _save_explanation_to_registry(self, explanation: ModelExplanationResult) -> None:
        """Save explanation to model registry."""
        try:
            if not self.model_registry:
                return

            # Save explanation alongside model
            explanation_path = (self.model_registry.registry_path /
                              explanation.model_id / 'latest' / "explanation.json")

            # Ensure directory exists
            explanation_path.parent.mkdir(parents=True, exist_ok=True)

            # Save explanation data
            explanation_data = explanation.to_dict()
            safe_json_dump(explanation_data, explanation_path)

            _LOGGER.debug(f"💾 Saved explanation to: {explanation_path}")

        except Exception as e:
            _LOGGER.warning(f"⚠️ Could not save explanation to registry: {e}")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get explanation cache statistics."""
        return {
            'cache_size': len(self.explanation_cache),
            'max_cache_size': self.explanation_cache_size,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0.0
        }

    def clear_cache(self) -> None:
        """Clear explanation cache."""
        self.explanation_cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
        _LOGGER.info("🧹 Explanation cache cleared")

# Integration decorators for automatic explainability
def with_explainability(config: Optional[Dict[str, Any]] = None):
    """
    Decorator to automatically add explainability to model training functions.

    Args:
        config: Configuration for explainability

    Returns:
        Decorated function with explainability integration
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            # Extract model and data from function arguments
            # This is a simplified implementation - in practice, you'd need
            # to analyze the function signature to extract the right parameters

            result = func(*args, **kwargs)

            # If the function returns a model and training data, add explanations
            if isinstance(result, dict) and 'model' in result:
                try:
                    explainability_manager = ModelExplainabilityManager(config)

                    # Try to extract training data from result or kwargs
                    X_train = result.get('X_train') or kwargs.get('X_train')
                    X_test = result.get('X_test') or kwargs.get('X_test')
                    model_id = result.get('model_id') or kwargs.get('model_id', 'unknown')

                    if X_train is not None and X_test is not None:
                        explanation = explainability_manager.explain_model(
                            model=result['model'],
                            X_train=X_train,
                            X_test=X_test,
                            model_id=model_id,
                            model_type=result.get('model_type', 'unknown')
                        )
                        result['explanation'] = explanation
                        _LOGGER.info(f"✅ Added explainability to model: {model_id}")

                except Exception as e:
                    _LOGGER.warning(f"⚠️ Could not add explainability: {e}")

            return result

        return wrapper
    return decorator

# Convenience function for quick model explanation
def explain_model_quick(model: Any, X_train: np.ndarray, X_test: np.ndarray,
                       model_id: str, config: Optional[Dict[str, Any]] = None) -> ModelExplanationResult:
    """
    Quick function to generate model explanations.

    Args:
        model: The trained model
        X_train: Training features
        X_test: Test features
        model_id: Model identifier
        config: Optional configuration

    Returns:
        ModelExplanationResult
    """
    manager = ModelExplainabilityManager(config)
    return manager.explain_model(model, X_train, X_test, model_id)
