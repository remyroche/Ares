from src.utils.tprint import tprint

"""
Model Explanation Utilities with SHAP and LIME Integration

This module provides comprehensive model explanation utilities using SHAP and LIME
for interpretable machine learning and feature importance analysis.

Key Features:
- SHAP (SHapley Additive exPlanations) integration
- LIME (Local Interpretable Model-agnostic Explanations) integration
- Feature importance analysis
- Model interpretability assessment
- Explanation visualization support

Built on existing utilities:
- Uses confidence_metrics.py for prediction confidence
- Integrates with model_evaluation.py for comprehensive assessment
- Leverages common_operations.py for robust error handling
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
import logging
from datetime import datetime
import time

from ..confidence_metrics import calculate_confidence_metrics
from ...common_operations import create_fallback_logger

# Enhanced dependency management with fast fail
try:
    from ..logger import get_logger
    _LOGGER = get_logger("MLCommon.ModelExplanations")
    tprint("✅ Custom logger available for MLCommon.ModelExplanations")
except Exception as e:
    tprint(f"⚠️ Custom logger not available: {e}. Using standard logging.")
    _LOGGER = logging.getLogger("MLCommon.ModelExplanations")
    _LOGGER.setLevel(logging.INFO)

logger = _LOGGER

# Optional imports for SHAP and LIME
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("SHAP not available - SHAP explanations will be disabled")

try:
    import lime
    import lime.lime_tabular
    LIME_AVAILABLE = True
    logger.info("LIME library available - LIME explanations enabled")
except ImportError:
    LIME_AVAILABLE = False
    logger.debug("LIME not available - using alternative explanation methods")
    logger.info("Model explanations available using SHAP and fallback methods")

class ModelExplainer:
    """Comprehensive model explanation system using SHAP and LIME."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the model explainer.

        Args:
            config: Configuration dictionary for explanation parameters
        """
        self.config = config or {}
        self.logger = create_fallback_logger('ModelExplainer')

        _LOGGER.info("🚀 Initializing ModelExplainer...")

        # Explanation configuration
        self.enable_shap = self.config.get('enable_shap', SHAP_AVAILABLE)
        self.enable_lime = self.config.get('enable_lime', LIME_AVAILABLE)
        self.shap_sample_size = self.config.get('shap_sample_size', 100)
        self.lime_sample_size = self.config.get('lime_sample_size', 10)
        self.explanation_cache = {}

        _LOGGER.info(f"⚙️ Configuration - SHAP enabled: {self.enable_shap}")
        _LOGGER.info(f"⚙️ Configuration - LIME enabled: {self.enable_lime}")
        _LOGGER.info(f"⚙️ Configuration - SHAP sample size: {self.shap_sample_size}")
        _LOGGER.info(f"⚙️ Configuration - LIME sample size: {self.lime_sample_size}")
        _LOGGER.info(f"⚙️ Dependencies - SHAP available: {SHAP_AVAILABLE}, LIME available: {LIME_AVAILABLE}")

        _LOGGER.info("✅ ModelExplainer initialized successfully")

    def explain_model(self, model: Any, X_train: np.ndarray, X_test: np.ndarray,
                     feature_names: Optional[List[str]] = None,
                     model_name: str = "model") -> Dict[str, Any]:
        """
        Generate comprehensive model explanations using SHAP and LIME.

        Args:
            model: The trained model to explain
            X_train: Training features
            X_test: Test features
            feature_names: List of feature names
            model_name: Name of the model

        Returns:
            Comprehensive explanation results
        """
        start_time = time.time()
        _LOGGER.info(f'🔍 Starting model explanations for {model_name}...')
        _LOGGER.info(f'📊 Data shapes - Train: {X_train.shape}, Test: {X_test.shape}')
        _LOGGER.info(f'📊 Features: {len(feature_names) if feature_names else X_train.shape[1]}')

        try:
            explanation_results = {
                'model_name': model_name,
                'shap_explanations': {},
                'lime_explanations': {},
                'feature_importance': {},
                'explanation_metadata': {
                    'shap_available': self.enable_shap and SHAP_AVAILABLE,
                    'lime_available': self.enable_lime and LIME_AVAILABLE,
                    'feature_count': len(feature_names) if feature_names else X_train.shape[1],
                    'sample_count': len(X_test)
                }
            }

            # Generate SHAP explanations
            if self.enable_shap and SHAP_AVAILABLE:
                try:
                    explanation_results['shap_explanations'] = self._generate_shap_explanations(
                        model, X_train, X_test, feature_names
                    )
                except Exception as e:
                    self.logger.warning(f'SHAP explanation failed: {e}')
                    explanation_results['shap_explanations'] = {'error': str(e)}

            # Generate LIME explanations
            if self.enable_lime and LIME_AVAILABLE:
                try:
                    explanation_results['lime_explanations'] = self._generate_lime_explanations(
                        model, X_train, X_test, feature_names
                    )
                except Exception as e:
                    self.logger.warning(f'LIME explanation failed: {e}')
                    explanation_results['lime_explanations'] = {'error': str(e)}

            # Generate feature importance analysis
            explanation_results['feature_importance'] = self._analyze_feature_importance(
                model, X_test, feature_names
            )

            self.logger.info(f'✅ Model explanations generated for {model_name}')
            execution_time = time.time() - start_time
            _LOGGER.info(f'✅ Model explanations completed in {execution_time:.3f}s for {model_name}')
            _LOGGER.info(f'📊 Results - SHAP: {bool(explanation_results["shap_explanations"])}, '
                        f'LIME: {bool(explanation_results["lime_explanations"])}, '
                        f'Feature importance: {bool(explanation_results["feature_importance"])}')
            return explanation_results

        except Exception as e:
            execution_time = time.time() - start_time
            _LOGGER.error(f'❌ Model explanation failed after {execution_time:.3f}s for {model_name}: {e}')
            return {
                'model_name': model_name,
                'error': str(e),
                'shap_explanations': {},
                'lime_explanations': {},
                'feature_importance': {}
            }

    def _generate_shap_explanations(self, model: Any, X_train: np.ndarray, X_test: np.ndarray,
                                   feature_names: Optional[List[str]]) -> Dict[str, Any]:
        """Generate SHAP explanations for the model."""
        _LOGGER.debug("🔍 Generating SHAP explanations...")
        try:
            # Sample data for SHAP (to avoid memory issues)
            sample_size = min(self.shap_sample_size, len(X_test))
            test_indices = np.random.choice(len(X_test), sample_size, replace=False)
            X_test_sample = X_test[test_indices]
            _LOGGER.debug(f"📊 Using sample size {sample_size} for SHAP explanations")

            # Create SHAP explainer based on model type
            if hasattr(model, 'predict_proba'):
                # For models with predict_proba
                if hasattr(model, 'feature_importances_'):
                    # Tree-based models
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(X_test_sample)
                else:
                    # Other models
                    explainer = shap.Explainer(model, X_train[:100])  # Use small sample for background
                    shap_values = explainer(X_test_sample)
            else:
                # For models without predict_proba
                explainer = shap.Explainer(model, X_train[:100])
                shap_values = explainer(X_test_sample)

            # Process SHAP values
            if isinstance(shap_values, list):
                # Multi-class case - use the first class
                shap_values_processed = shap_values[0] if len(shap_values) > 0 else shap_values
            else:
                shap_values_processed = shap_values.values if hasattr(shap_values, 'values') else shap_values

            # Calculate feature importance from SHAP values
            feature_importance = np.abs(shap_values_processed).mean(axis=0)

            # Create feature importance dictionary
            importance_dict = {}
            if feature_names:
                for i, (feature, importance) in enumerate(zip(feature_names, feature_importance)):
                    importance_dict[feature] = float(importance)
            else:
                for i, importance in enumerate(feature_importance):
                    importance_dict[f'feature_{i}'] = float(importance)

            # Sort by importance
            sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)

            return {
                'shap_values': shap_values_processed.tolist() if hasattr(shap_values_processed, 'tolist') else None,
                'feature_importance': importance_dict,
                'top_features': sorted_importance[:10],
                'mean_importance': float(np.mean(feature_importance)),
                'std_importance': float(np.std(feature_importance)),
                'sample_size': sample_size,
                'explainer_type': type(explainer).__name__
            }

        except Exception as e:
            _LOGGER.warning(f'⚠️ SHAP explanation generation failed: {e}')
            return {'error': str(e)}

    def _generate_lime_explanations(self, model: Any, X_train: np.ndarray, X_test: np.ndarray,
                                   feature_names: Optional[List[str]]) -> Dict[str, Any]:
        """Generate LIME explanations for the model."""
        _LOGGER.debug("🔍 Generating LIME explanations...")
        try:
            # Sample data for LIME
            sample_size = min(self.lime_sample_size, len(X_test))
            test_indices = np.random.choice(len(X_test), sample_size, replace=False)
            X_test_sample = X_test[test_indices]
            _LOGGER.debug(f"📊 Using sample size {sample_size} for LIME explanations")

            # Create LIME explainer
            explainer = lime.lime_tabular.LimeTabularExplainer(
                X_train,
                feature_names=feature_names or [f'feature_{i}' for i in range(X_train.shape[1])],
                mode='classification' if hasattr(model, 'predict_proba') else 'regression',
                discretize_continuous=True
            )

            # Generate explanations for sample instances
            explanations = []
            feature_importance_scores = {}

            for i, instance in enumerate(X_test_sample):
                try:
                    # Generate explanation for this instance
                    explanation = explainer.explain_instance(
                        instance,
                        model.predict_proba if hasattr(model, 'predict_proba') else model.predict,
                        num_features=min(10, len(feature_names) if feature_names else X_train.shape[1])
                    )

                    # Extract feature importance
                    exp_list = explanation.as_list()
                    explanations.append(exp_list)

                    # Aggregate feature importance scores
                    for feature, importance in exp_list:
                        if feature not in feature_importance_scores:
                            feature_importance_scores[feature] = []
                        feature_importance_scores[feature].append(abs(importance))

                except Exception as e:
                    self.logger.warning(f'LIME explanation failed for instance {i}: {e}')
                    continue

            # Calculate average feature importance
            avg_importance = {}
            for feature, scores in feature_importance_scores.items():
                avg_importance[feature] = float(np.mean(scores))

            # Sort by importance
            sorted_importance = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)

            return {
                'explanations': explanations,
                'feature_importance': avg_importance,
                'top_features': sorted_importance[:10],
                'mean_importance': float(np.mean(list(avg_importance.values()))) if avg_importance else 0.0,
                'std_importance': float(np.std(list(avg_importance.values()))) if avg_importance else 0.0,
                'sample_size': len(explanations),
                'successful_explanations': len(explanations)
            }

        except Exception as e:
            _LOGGER.warning(f'⚠️ LIME explanation generation failed: {e}')
            return {'error': str(e)}

    def _analyze_feature_importance(self, model: Any, X_test: np.ndarray,
                                   feature_names: Optional[List[str]]) -> Dict[str, Any]:
        """Analyze feature importance using multiple methods."""
        try:
            importance_analysis = {
                'model_based_importance': {},
                'permutation_importance': {},
                'combined_importance': {}
            }

            # Model-based feature importance
            if hasattr(model, 'feature_importances_'):
                model_importance = model.feature_importances_
                if feature_names:
                    model_importance_dict = {feature: float(imp) for feature, imp in zip(feature_names, model_importance)}
                else:
                    model_importance_dict = {f'feature_{i}': float(imp) for i, imp in enumerate(model_importance)}
                importance_analysis['model_based_importance'] = model_importance_dict

            # Permutation importance (simplified version)
            try:
                from sklearn.inspection import permutation_importance
                perm_importance = permutation_importance(model, X_test, model.predict(X_test), n_repeats=5, random_state=42)

                if feature_names:
                    perm_importance_dict = {feature: float(imp) for feature, imp in zip(feature_names, perm_importance.importances_mean)}
                else:
                    perm_importance_dict = {f'feature_{i}': float(imp) for i, imp in enumerate(perm_importance.importances_mean)}
                importance_analysis['permutation_importance'] = perm_importance_dict

            except Exception as e:
                self.logger.warning(f'Permutation importance failed: {e}')
                importance_analysis['permutation_importance'] = {}

            # Combine importance scores
            all_features = set()
            if importance_analysis['model_based_importance']:
                all_features.update(importance_analysis['model_based_importance'].keys())
            if importance_analysis['permutation_importance']:
                all_features.update(importance_analysis['permutation_importance'].keys())

            combined_importance = {}
            for feature in all_features:
                scores = []
                if feature in importance_analysis['model_based_importance']:
                    scores.append(importance_analysis['model_based_importance'][feature])
                if feature in importance_analysis['permutation_importance']:
                    scores.append(importance_analysis['permutation_importance'][feature])

                if scores:
                    combined_importance[feature] = float(np.mean(scores))

            # Sort by combined importance
            sorted_combined = sorted(combined_importance.items(), key=lambda x: x[1], reverse=True)
            importance_analysis['combined_importance'] = {
                'scores': combined_importance,
                'top_features': sorted_combined[:10],
                'mean_importance': float(np.mean(list(combined_importance.values()))) if combined_importance else 0.0
            }

            return importance_analysis

        except Exception as e:
            self.logger.warning(f'Feature importance analysis failed: {e}')
            return {'error': str(e)}

    def log_explanations(self, explanation_results: Dict[str, Any], model_name: str) -> None:
        """Log model explanations in a formatted way."""
        try:
            self.logger.info(f'🔍 === MODEL EXPLANATIONS: {model_name} ===')

            # Log SHAP results
            if 'shap_explanations' in explanation_results and 'error' not in explanation_results['shap_explanations']:
                shap_results = explanation_results['shap_explanations']
                if 'top_features' in shap_results:
                    self.logger.info(f'🎯 SHAP Top Features: {", ".join([f"{feat}({imp:.3f})" for feat, imp in shap_results["top_features"][:5]])}')

            # Log LIME results
            if 'lime_explanations' in explanation_results and 'error' not in explanation_results['lime_explanations']:
                lime_results = explanation_results['lime_explanations']
                if 'top_features' in lime_results:
                    self.logger.info(f'🍋 LIME Top Features: {", ".join([f"{feat}({imp:.3f})" for feat, imp in lime_results["top_features"][:5]])}')

            # Log combined feature importance
            if 'feature_importance' in explanation_results and 'combined_importance' in explanation_results['feature_importance']:
                combined = explanation_results['feature_importance']['combined_importance']
                if 'top_features' in combined:
                    self.logger.info(f'🔗 Combined Top Features: {", ".join([f"{feat}({imp:.3f})" for feat, imp in combined["top_features"][:5]])}')

        except Exception as e:
            self.logger.warning(f'Failed to log explanations for {model_name}: {e}')

def explain_model_with_shap_lime(model: Any, X_train: np.ndarray, X_test: np.ndarray,
                                feature_names: Optional[List[str]] = None,
                                model_name: str = "model",
                                config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Convenience function to explain a model using SHAP and LIME.

    Args:
        model: The trained model to explain
        X_train: Training features
        X_test: Test features
        feature_names: List of feature names
        model_name: Name of the model
        config: Configuration dictionary

    Returns:
        Comprehensive explanation results
    """
    explainer = ModelExplainer(config)
    return explainer.explain_model(model, X_train, X_test, feature_names, model_name)
