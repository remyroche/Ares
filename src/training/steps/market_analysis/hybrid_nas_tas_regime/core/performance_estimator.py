"""
Unified Performance Estimator for Architecture Search

This module provides fast performance estimation for neural and tree-based architectures
without requiring full training, using meta-learning and ensemble methods.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
import time
from datetime import datetime
from pathlib import Path
import pickle
import json
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score

logger = logging.getLogger(__name__)


class EstimatorType(Enum):
    """Types of performance estimators."""
    LINEAR = "linear"
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    SVM = "svm"
    ENSEMBLE = "ensemble"
    META_LEARNER = "meta_learner"
    NEURAL = "neural"


@dataclass
class PerformancePrediction:
    """Result from performance estimation."""
    predicted_performance: float
    confidence_interval: Tuple[float, float]
    uncertainty: float
    feature_importance: Dict[str, float]
    estimation_method: str
    estimation_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UnifiedPerformanceEstimatorConfig:
    """Configuration for unified performance estimator."""
    estimator_type: EstimatorType = EstimatorType.ENSEMBLE
    neural_config: Dict[str, Any] = field(default_factory=dict)
    tree_config: Dict[str, Any] = field(default_factory=dict)

    # Training settings
    train_meta_learner: bool = True
    meta_learner_update_frequency: int = 100
    cross_validation_folds: int = 5

    # Architecture features
    max_architecture_features: int = 50
    use_architecture_encoding: bool = True
    encode_categorical_features: bool = True

    # Performance estimation
    confidence_level: float = 0.95
    min_training_samples: int = 50
    max_training_samples: int = 1000

    # Output settings
    save_predictions: bool = True
    prediction_history_size: int = 1000


class UnifiedPerformanceEstimator:
    """
    Unified Performance Estimator for NAS-TAS architectures.

    Provides fast performance estimation using meta-learning approaches,
    ensemble methods, and architecture feature analysis.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize the unified performance estimator."""
        self.config = UnifiedPerformanceEstimatorConfig(**config)
        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize estimators
        self.estimators = {}
        self.meta_learner = None
        self.feature_scaler = StandardScaler()

        # Training data
        self.architecture_features = []
        self.performance_targets = []
        self.prediction_history = []

        # Architecture encoders
        self.architecture_encoder = None

        self._initialize_estimators()
        self.logger.info("✅ Unified Performance Estimator initialized")

    def _initialize_estimators(self):
        """Initialize all performance estimators."""
        try:
            # Linear regression (baseline)
            self.estimators['linear'] = LinearRegression()

            # Random Forest (good for non-linear relationships)
            self.estimators['random_forest'] = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )

            # Gradient Boosting (best performance)
            self.estimators['gradient_boosting'] = GradientBoostingRegressor(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                random_state=42
            )

            # SVM (for robustness)
            self.estimators['svm'] = SVR(kernel='rbf', C=1.0, gamma='scale')

            self.logger.info("✅ All base estimators initialized")

            # Initialize meta-learner if required
            if self.config.train_meta_learner:
                self._initialize_meta_learner()

        except Exception as e:
            self.logger.error(f"❌ Estimator initialization failed: {e}")
            raise

    def _initialize_meta_learner(self):
        """Initialize meta-learner for ensemble predictions."""
        try:
            # Meta-learner uses predictions from base estimators
            self.meta_learner = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )

            self.logger.info("✅ Meta-learner initialized")

        except Exception as e:
            self.logger.error(f"❌ Meta-learner initialization failed: {e}")
            self.meta_learner = None

    def predict_performance(self, architecture: Dict[str, Any]) -> PerformancePrediction:
        """Predict architecture performance without training."""
        start_time = time.time()

        try:
            # Extract architecture features
            features = self._extract_architecture_features(architecture)
            features_scaled = self.feature_scaler.transform(features.reshape(1, -1))

            # Get predictions from base estimators
            base_predictions = {}
            for name, estimator in self.estimators.items():
                try:
                    pred = estimator.predict(features_scaled)[0]
                    base_predictions[name] = pred
                except Exception as e:
                    self.logger.warning(f"Base estimator {name} failed: {e}")
                    base_predictions[name] = 0.5  # Default prediction

            # Ensemble prediction
            if self.config.estimator_type == EstimatorType.ENSEMBLE and self.meta_learner:
                # Use meta-learner for final prediction
                meta_features = np.array(list(base_predictions.values())).reshape(1, -1)
                final_prediction = self.meta_learner.predict(meta_features)[0]
                method_used = "meta_learner"
            else:
                # Simple averaging of base predictions
                final_prediction = np.mean(list(base_predictions.values()))
                method_used = "ensemble_average"

            # Calculate confidence interval
            prediction_std = np.std(list(base_predictions.values()))
            confidence_interval = self._calculate_confidence_interval(
                final_prediction, prediction_std
            )

            # Calculate feature importance (simplified)
            feature_importance = self._calculate_feature_importance(
                features, features_scaled, final_prediction
            )

            # Calculate uncertainty
            uncertainty = prediction_std / max(abs(final_prediction), 0.1)

            estimation_time = time.time() - start_time

            # Create prediction result
            prediction = PerformancePrediction(
                predicted_performance=max(0.0, min(1.0, final_prediction)),  # Clamp to [0,1]
                confidence_interval=confidence_interval,
                uncertainty=min(1.0, uncertainty),
                feature_importance=feature_importance,
                estimation_method=method_used,
                estimation_time=estimation_time,
                metadata={
                    'base_predictions': base_predictions,
                    'features_extracted': len(features),
                    'architecture_type': architecture.get('type', 'unknown')
                }
            )

            # Store prediction in history
            self._store_prediction(architecture, prediction)

            self.logger.debug(f"Performance prediction: {prediction.predicted_performance:.4f} "
                            f"(confidence: {prediction.confidence_interval})")

            return prediction

        except Exception as e:
            estimation_time = time.time() - start_time
            self.logger.error(f"Performance prediction failed: {e}")

            # Return default prediction
            return PerformancePrediction(
                predicted_performance=0.5,
                confidence_interval=(0.3, 0.7),
                uncertainty=0.5,
                feature_importance={},
                estimation_method="fallback",
                estimation_time=estimation_time,
                metadata={'error': str(e)}
            )

    def _extract_architecture_features(self, architecture: Dict[str, Any]) -> np.ndarray:
        """Extract numerical features from architecture for estimation."""
        try:
            features = []

            # Basic architecture properties
            architecture_type = architecture.get('type', 'neural')
            layers = architecture.get('layers', [])

            # Type encoding (one-hot)
            type_encoding = [1 if architecture_type == 'neural' else 0,
                           1 if architecture_type == 'tree' else 0,
                           1 if architecture_type == 'hybrid' else 0]
            features.extend(type_encoding)

            # Layer information
            n_layers = len(layers)
            total_parameters = 0
            max_layer_size = 0
            min_layer_size = float('inf')

            for layer in layers:
                hidden_size = layer.get('hidden_size', 100)
                total_parameters += hidden_size
                max_layer_size = max(max_layer_size, hidden_size)
                min_layer_size = min(min_layer_size, hidden_size)

            features.extend([
                n_layers,
                total_parameters,
                max_layer_size,
                min_layer_size if min_layer_size != float('inf') else 100,
                np.mean([layer.get('hidden_size', 100) for layer in layers])
            ])

            # Activation functions (simplified count)
            activation_counts = {}
            for layer in layers:
                activation = layer.get('activation', 'relu')
                activation_counts[activation] = activation_counts.get(activation, 0) + 1

            # Add top 5 most common activations
            sorted_activations = sorted(activation_counts.items(), key=lambda x: x[1], reverse=True)
            for activation, count in sorted_activations[:5]:
                features.append(count)

            # Pad with zeros if fewer than 5
            while len(features) < len(type_encoding) + 5 + 5:
                features.append(0)

            # Dropout and regularization
            dropout_count = sum(1 for layer in layers if layer.get('dropout_rate', 0) > 0)
            batch_norm_count = sum(1 for layer in layers if layer.get('batch_norm', False))

            features.extend([dropout_count, batch_norm_count])

            # Complexity metrics
            complexity_score = self._calculate_complexity_score(architecture)
            efficiency_score = self._calculate_efficiency_score(architecture)

            features.extend([complexity_score, efficiency_score])

            # Ensure we don't exceed max features
            features = features[:self.config.max_architecture_features]

            # Pad with zeros if needed
            while len(features) < self.config.max_architecture_features:
                features.append(0)

            return np.array(features)

        except Exception as e:
            self.logger.warning(f"Feature extraction failed: {e}")
            # Return default features
            return np.zeros(self.config.max_architecture_features)

    def _calculate_complexity_score(self, architecture: Dict[str, Any]) -> float:
        """Calculate architecture complexity score (0-1, higher = more complex)."""
        try:
            layers = architecture.get('layers', [])
            n_layers = len(layers)
            total_parameters = sum(layer.get('hidden_size', 100) for layer in layers)

            # Normalize components
            normalized_layers = min(n_layers / 10.0, 1.0)
            normalized_params = min(total_parameters / 1000000.0, 1.0)

            # Weighted combination
            complexity = 0.6 * normalized_layers + 0.4 * normalized_params
            return complexity

        except Exception:
            return 0.5

    def _calculate_efficiency_score(self, architecture: Dict[str, Any]) -> float:
        """Calculate architecture efficiency score (0-1, higher = more efficient)."""
        try:
            layers = architecture.get('layers', [])
            if not layers:
                return 0.0

            total_parameters = sum(layer.get('hidden_size', 100) for layer in layers)

            # Efficiency based on parameter utilization
            layer_sizes = [layer.get('hidden_size', 100) for layer in layers]
            parameter_efficiency = 1.0 / (1.0 + np.std(layer_sizes) / np.mean(layer_sizes))

            return min(parameter_efficiency, 1.0)

        except Exception:
            return 0.5

    def _calculate_confidence_interval(self, prediction: float, std: float) -> Tuple[float, float]:
        """Calculate confidence interval for prediction."""
        try:
            # Use t-distribution approximation for confidence interval
            z_score = 1.96  # 95% confidence interval

            margin_of_error = z_score * std
            lower_bound = max(0.0, prediction - margin_of_error)
            upper_bound = min(1.0, prediction + margin_of_error)

            return (lower_bound, upper_bound)

        except Exception:
            return (max(0.0, prediction - 0.2), min(1.0, prediction + 0.2))

    def _calculate_feature_importance(self, features: np.ndarray,
                                   features_scaled: np.ndarray,
                                   prediction: float) -> Dict[str, float]:
        """Calculate feature importance for the prediction."""
        try:
            importance_dict = {}

            # Use gradient boosting feature importance if available
            if 'gradient_boosting' in self.estimators:
                gb = self.estimators['gradient_boosting']
                if hasattr(gb, 'feature_importances_'):
                    importance = gb.feature_importances_
                else:
                    # Fallback: use random forest importance
                    rf = self.estimators.get('random_forest')
                    if rf and hasattr(rf, 'feature_importances_'):
                        importance = rf.feature_importances_
                    else:
                        importance = np.ones(len(features)) / len(features)
            else:
                importance = np.ones(len(features)) / len(features)

            # Map importance to feature names (simplified)
            feature_names = [
                'type_neural', 'type_tree', 'type_hybrid',
                'n_layers', 'total_params', 'max_layer', 'min_layer', 'mean_layer',
                'activation_1', 'activation_2', 'activation_3', 'activation_4', 'activation_5',
                'dropout_count', 'batch_norm_count',
                'complexity', 'efficiency'
            ]

            for i, name in enumerate(feature_names):
                if i < len(importance):
                    importance_dict[name] = float(importance[i])

            return importance_dict

        except Exception as e:
            self.logger.warning(f"Feature importance calculation failed: {e}")
            return {}

    def _store_prediction(self, architecture: Dict[str, Any], prediction: PerformancePrediction):
        """Store prediction in history for analysis."""
        try:
            if len(self.prediction_history) >= self.config.prediction_history_size:
                self.prediction_history.pop(0)  # Remove oldest

            self.prediction_history.append({
                'architecture': architecture,
                'prediction': prediction.predicted_performance,
                'confidence': prediction.confidence_interval,
                'uncertainty': prediction.uncertainty,
                'method': prediction.estimation_method,
                'timestamp': datetime.now().isoformat()
            })

        except Exception as e:
            self.logger.warning(f"Could not store prediction: {e}")

    def train_on_actual_performance(self, architecture: Dict[str, Any],
                                  actual_performance: float):
        """Train estimator on actual architecture performance."""
        try:
            # Extract features
            features = self._extract_architecture_features(architecture)

            # Store training data
            self.architecture_features.append(features)
            self.performance_targets.append(actual_performance)

            # Limit training data size
            if len(self.architecture_features) > self.config.max_training_samples:
                self.architecture_features.pop(0)
                self.performance_targets.pop(0)

            # Retrain estimators if we have enough data
            if len(self.architecture_features) >= self.config.min_training_samples:
                self._retrain_estimators()

        except Exception as e:
            self.logger.warning(f"Training on actual performance failed: {e}")

    def _retrain_estimators(self):
        """Retrain all estimators on accumulated data."""
        try:
            if len(self.architecture_features) < self.config.min_training_samples:
                return

            # Convert to numpy arrays
            X = np.array(self.architecture_features)
            y = np.array(self.performance_targets)

            # Fit scaler
            self.feature_scaler.fit(X)

            # Scale features
            X_scaled = self.feature_scaler.transform(X)

            # Train base estimators
            for name, estimator in self.estimators.items():
                try:
                    estimator.fit(X_scaled, y)
                except Exception as e:
                    self.logger.warning(f"Training estimator {name} failed: {e}")

            # Train meta-learner if available
            if self.meta_learner and len(self.estimators) > 1:
                # Get predictions from base estimators for meta-learner training
                meta_features = []
                for i in range(len(X_scaled)):
                    base_preds = []
                    for name, estimator in self.estimators.items():
                        try:
                            pred = estimator.predict(X_scaled[i:i+1])[0]
                            base_preds.append(pred)
                        except:
                            base_preds.append(0.5)
                    meta_features.append(base_preds)

                meta_features = np.array(meta_features)
                self.meta_learner.fit(meta_features, y)

            self.logger.info(f"✅ Retrained estimators on {len(X)} samples")

        except Exception as e:
            self.logger.error(f"❌ Estimator retraining failed: {e}")

    def get_estimator_performance(self) -> Dict[str, Any]:
        """Get performance metrics for all estimators."""
        try:
            if len(self.architecture_features) < 10:
                return {'error': 'Insufficient training data'}

            X = np.array(self.architecture_features)
            y = np.array(self.performance_targets)
            X_scaled = self.feature_scaler.transform(X)

            performance_metrics = {}

            for name, estimator in self.estimators.items():
                try:
                    # Cross-validation score
                    cv_scores = cross_val_score(estimator, X_scaled, y, cv=3, scoring='r2')
                    mse_scores = cross_val_score(estimator, X_scaled, y, cv=3, scoring='neg_mean_squared_error')

                    performance_metrics[name] = {
                        'r2_score': float(np.mean(cv_scores)),
                        'r2_std': float(np.std(cv_scores)),
                        'mse_score': float(-np.mean(mse_scores)),
                        'mse_std': float(np.std(mse_scores))
                    }
                except Exception as e:
                    performance_metrics[name] = {'error': str(e)}

            # Add meta-learner performance
            if self.meta_learner:
                try:
                    meta_features = []
                    for i in range(len(X_scaled)):
                        base_preds = []
                        for name, estimator in self.estimators.items():
                            try:
                                pred = estimator.predict(X_scaled[i:i+1])[0]
                                base_preds.append(pred)
                            except:
                                base_preds.append(0.5)
                        meta_features.append(base_preds)

                    meta_features = np.array(meta_features)
                    cv_scores = cross_val_score(self.meta_learner, meta_features, y, cv=3, scoring='r2')

                    performance_metrics['meta_learner'] = {
                        'r2_score': float(np.mean(cv_scores)),
                        'r2_std': float(np.std(cv_scores))
                    }
                except Exception as e:
                    performance_metrics['meta_learner'] = {'error': str(e)}

            return performance_metrics

        except Exception as e:
            return {'error': str(e)}

    def save_estimator_state(self, filepath: str) -> bool:
        """Save estimator state to disk."""
        try:
            state = {
                'config': self.config.__dict__,
                'estimators': {},
                'meta_learner': self.meta_learner,
                'feature_scaler': self.feature_scaler,
                'architecture_features': self.architecture_features,
                'performance_targets': self.performance_targets,
                'prediction_history': self.prediction_history
            }

            # Serialize estimators
            for name, estimator in self.estimators.items():
                try:
                    state['estimators'][name] = estimator
                except Exception as e:
                    self.logger.warning(f"Could not serialize estimator {name}: {e}")

            with open(filepath, 'wb') as f:
                pickle.dump(state, f)

            self.logger.info(f"✅ Estimator state saved to {filepath}")
            return True

        except Exception as e:
            self.logger.error(f"❌ Failed to save estimator state: {e}")
            return False

    def load_estimator_state(self, filepath: str) -> bool:
        """Load estimator state from disk."""
        try:
            with open(filepath, 'rb') as f:
                state = pickle.load(f)

            self.config = UnifiedPerformanceEstimatorConfig(**state['config'])
            self.estimators = state['estimators']
            self.meta_learner = state.get('meta_learner')
            self.feature_scaler = state.get('feature_scaler', StandardScaler())
            self.architecture_features = state.get('architecture_features', [])
            self.performance_targets = state.get('performance_targets', [])
            self.prediction_history = state.get('prediction_history', [])

            self.logger.info(f"✅ Estimator state loaded from {filepath}")
            return True

        except Exception as e:
            self.logger.error(f"❌ Failed to load estimator state: {e}")
            return False


def create_unified_performance_estimator(config: Dict[str, Any]) -> UnifiedPerformanceEstimator:
    """Create a unified performance estimator instance."""
    return UnifiedPerformanceEstimator(config)


def quick_performance_estimate(architecture: Dict[str, Any],
                              config: Optional[Dict[str, Any]] = None) -> float:
    """Quick performance estimation with default settings."""
    if config is None:
        config = {
            'estimator_type': 'ensemble',
            'neural_config': {},
            'tree_config': {}
        }

    estimator = UnifiedPerformanceEstimator(config)
    prediction = estimator.predict_performance(architecture)
    return prediction.predicted_performance