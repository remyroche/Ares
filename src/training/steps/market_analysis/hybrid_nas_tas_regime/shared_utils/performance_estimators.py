"""
Performance Estimators for NAS and TAS Systems

This module provides surrogate models and performance predictors for quick architecture
evaluation without full training runs. Supports both neural and tree architectures.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
import time
from datetime import datetime
from src.utils.tprint import (
    tprint, tprint_debug, tprint_info, tprint_warning, tprint_error,
    tprint_success, tprint_progress, tprint_performance, tprint_timer
)
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import os
from pathlib import Path

logger = logging.getLogger(__name__)

class PerformanceMetric(Enum):
    """Performance metrics for architecture evaluation."""
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    AUC = "auc"
    MSE = "mse"
    MAE = "mae"
    R2 = "r2"
    SHARPE_RATIO = "sharpe_ratio"
    MAX_DRAWDOWN = "max_drawdown"
    WIN_RATE = "win_rate"
    PROFIT_FACTOR = "profit_factor"

class EstimatorType(Enum):
    """Types of performance estimators."""
    LINEAR_REGRESSION = "linear_regression"
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    GAUSSIAN_PROCESS = "gaussian_process"
    SUPPORT_VECTOR = "support_vector"
    ENSEMBLE = "ensemble"

@dataclass
class ArchitectureFeatures:
    """Features extracted from an architecture for performance estimation."""
    n_layers: int = 0
    total_parameters: int = 0
    max_layer_size: int = 0
    avg_layer_size: float = 0.0
    depth: int = 0
    width: int = 0
    complexity_score: float = 0.0
    memory_estimate: float = 0.0
    training_time_estimate: float = 0.0

    # Neural network specific
    n_conv_layers: int = 0
    n_recurrent_layers: int = 0
    n_attention_layers: int = 0
    has_residual_connections: bool = False
    has_batch_norm: bool = False
    has_dropout: bool = False
    activation_complexity: float = 0.0

    # Tree specific
    n_trees: int = 0
    max_tree_depth: int = 0
    avg_tree_depth: float = 0.0
    ensemble_method: str = "single"
    has_boosting: bool = False
    has_bagging: bool = False

    # Connection features
    n_connections: int = 0
    n_skip_connections: int = 0
    n_residual_connections: int = 0
    connection_density: float = 0.0

    # Meta features
    architecture_hash: str = ""
    architecture_type: str = "unknown"  # "neural" or "tree"

@dataclass
class PerformancePrediction:
    """Prediction result from a performance estimator."""
    predicted_performance: float
    confidence_interval: Tuple[float, float]
    feature_importance: Dict[str, float]
    prediction_time: float
    model_used: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TrainingHistory:
    """Historical training data for performance estimator."""
    architectures: List[ArchitectureFeatures]
    true_performances: List[float]
    metadata: List[Dict[str, Any]]
    timestamp: datetime = field(default_factory=datetime.now)

class BasePerformanceEstimator:
    """Base class for performance estimators."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize the performance estimator."""
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.is_trained = False
        self.feature_scaler = StandardScaler()
        self.model = None
        self.training_history = TrainingHistory([], [], [])

    def extract_features(self, architecture: Any) -> ArchitectureFeatures:
        """Extract features from an architecture."""
        try:
            # Extract basic architecture features
            features = {}

            # Get architecture parameters if available
            if hasattr(architecture, 'parameters'):
                features['num_parameters'] = sum(p.numel() for p in architecture.parameters())
            else:
                features['num_parameters'] = 0

            # Get architecture depth if available
            if hasattr(architecture, 'depth'):
                features['depth'] = architecture.depth
            elif hasattr(architecture, 'layers'):
                features['depth'] = len(architecture.layers)
            else:
                features['depth'] = 1

            # Get architecture width if available
            if hasattr(architecture, 'width'):
                features['width'] = architecture.width
            elif hasattr(architecture, 'hidden_size'):
                features['width'] = architecture.hidden_size
            else:
                features['width'] = 64

            # Get activation function if available
            if hasattr(architecture, 'activation'):
                features['activation'] = str(architecture.activation)
            else:
                features['activation'] = 'relu'

            # Get dropout rate if available
            if hasattr(architecture, 'dropout'):
                features['dropout'] = architecture.dropout
            else:
                features['dropout'] = 0.0

            # Get learning rate if available
            if hasattr(architecture, 'learning_rate'):
                features['learning_rate'] = architecture.learning_rate
            else:
                features['learning_rate'] = 0.001

            # Get batch size if available
            if hasattr(architecture, 'batch_size'):
                features['batch_size'] = architecture.batch_size
            else:
                features['batch_size'] = 32

            # Get optimizer if available
            if hasattr(architecture, 'optimizer'):
                features['optimizer'] = str(architecture.optimizer)
            else:
                features['optimizer'] = 'adam'

            # Get regularization if available
            if hasattr(architecture, 'regularization'):
                features['regularization'] = architecture.regularization
            else:
                features['regularization'] = 0.0

            # Get architecture type if available
            if hasattr(architecture, 'architecture_type'):
                features['architecture_type'] = str(architecture.architecture_type)
            else:
                features['architecture_type'] = 'unknown'

            return ArchitectureFeatures(features)

        except Exception as e:
            tprint(f"⚠️ [PERFORMANCE] Error extracting features: {e}", color="yellow")
            # Return default features
            return ArchitectureFeatures({
                'num_parameters': 0,
                'depth': 1,
                'width': 64,
                'activation': 'relu',
                'dropout': 0.0,
                'learning_rate': 0.001,
                'batch_size': 32,
                'optimizer': 'adam',
                'regularization': 0.0,
                'architecture_type': 'unknown'
            })

    def predict_performance(self, architecture: Any) -> PerformancePrediction:
        """Predict performance of an architecture."""
        try:
            # Extract features from the architecture
            features = self.extract_features(architecture)

            # Check if the model is trained
            if not self.is_trained or self.model is None:
                # Return a default prediction
                return PerformancePrediction(
                    predicted_performance=0.5,
                    confidence=0.1,
                    feature_importance={},
                    prediction_interval=(0.0, 1.0)
                )

            # Prepare features for prediction
            feature_vector = self._prepare_feature_vector(features)

            # Make prediction
            if hasattr(self.model, 'predict'):
                prediction = self.model.predict([feature_vector])[0]
            else:
                prediction = self.model([feature_vector])[0]

            # Calculate confidence (simplified)
            confidence = min(0.9, max(0.1, abs(prediction - 0.5) * 2))

            # Calculate feature importance (simplified)
            feature_importance = self._calculate_feature_importance(features)

            # Calculate prediction interval
            prediction_interval = (
                max(0.0, prediction - 0.1),
                min(1.0, prediction + 0.1)
            )

            return PerformancePrediction(
                predicted_performance=prediction,
                confidence=confidence,
                feature_importance=feature_importance,
                prediction_interval=prediction_interval
            )

        except Exception as e:
            tprint_error(f"Error predicting performance: {e}")
            tprint_debug(f"Error details: {type(e).__name__}: {str(e)}")
            # Return a default prediction
            return PerformancePrediction(
                predicted_performance=0.5,
                confidence=0.1,
                feature_importance={},
                prediction_interval=(0.0, 1.0)
            )

    def train(self, architectures: List[ArchitectureFeatures], performances: List[float]) -> Dict[str, float]:
        """Train the performance estimator."""
        try:
            tprint_info("Training performance estimator")
            tprint_debug(f"Training data: {len(architectures)} architectures, {len(performances)} performances")
            if len(architectures) == 0 or len(performances) == 0:
                tprint_warning("No training data provided")
                return {'error': 'No training data provided'}

            # Prepare training data
            X = []
            y = performances

            for arch_features in architectures:
                feature_vector = self._prepare_feature_vector(arch_features)
                X.append(feature_vector)

            # Scale features
            X_scaled = self.feature_scaler.fit_transform(X)

            # Train the model
            if self.model is None:
                # Create a simple model if none exists
                from sklearn.ensemble import RandomForestRegressor
                self.model = RandomForestRegressor(n_estimators=100, random_state=42)

            # Fit the model
            self.model.fit(X_scaled, y)

            # Make predictions for evaluation
            y_pred = self.model.predict(X_scaled)

            # Calculate training metrics
            from sklearn.metrics import mean_squared_error, r2_score
            mse = mean_squared_error(y, y_pred)
            r2 = r2_score(y, y_pred)

            # Update training status
            self.is_trained = True

            # Update training history
            self.training_history.epochs.append(len(self.training_history.epochs))
            self.training_history.losses.append(mse)
            self.training_history.metrics.append(r2)

            return {
                'mse': mse,
                'r2_score': r2,
                'num_samples': len(architectures),
                'training_completed': True
            }

        except Exception as e:
            tprint(f"⚠️ [PERFORMANCE] Error training estimator: {e}", color="yellow")
            return {'error': str(e)}

    def _prepare_feature_vector(self, features: ArchitectureFeatures) -> List[float]:
        """Prepare feature vector for model input."""
        try:
            # Convert features to a list of numerical values
            feature_vector = []

            # Add numerical features
            feature_vector.append(features.features.get('num_parameters', 0))
            feature_vector.append(features.features.get('depth', 1))
            feature_vector.append(features.features.get('width', 64))
            feature_vector.append(features.features.get('dropout', 0.0))
            feature_vector.append(features.features.get('learning_rate', 0.001))
            feature_vector.append(features.features.get('batch_size', 32))
            feature_vector.append(features.features.get('regularization', 0.0))

            # Add categorical features as one-hot encoded
            activation = features.features.get('activation', 'relu')
            if activation == 'relu':
                feature_vector.extend([1, 0, 0])
            elif activation == 'sigmoid':
                feature_vector.extend([0, 1, 0])
            else:
                feature_vector.extend([0, 0, 1])

            optimizer = features.features.get('optimizer', 'adam')
            if optimizer == 'adam':
                feature_vector.extend([1, 0, 0])
            elif optimizer == 'sgd':
                feature_vector.extend([0, 1, 0])
            else:
                feature_vector.extend([0, 0, 1])

            return feature_vector

        except Exception as e:
            tprint(f"⚠️ [PERFORMANCE] Error preparing feature vector: {e}", color="yellow")
            # Return default feature vector
            return [0, 1, 64, 0.0, 0.001, 32, 0.0, 1, 0, 0, 1, 0, 0]

    def _calculate_feature_importance(self, features: ArchitectureFeatures) -> Dict[str, float]:
        """Calculate feature importance for the prediction."""
        try:
            if self.model is None or not hasattr(self.model, 'feature_importances_'):
                # Return default importance
                return {
                    'num_parameters': 0.2,
                    'depth': 0.2,
                    'width': 0.2,
                    'dropout': 0.1,
                    'learning_rate': 0.1,
                    'batch_size': 0.1,
                    'regularization': 0.1
                }

            # Get feature importance from the model
            importances = self.model.feature_importances_

            # Map importance to feature names
            feature_names = [
                'num_parameters', 'depth', 'width', 'dropout',
                'learning_rate', 'batch_size', 'regularization',
                'activation_relu', 'activation_sigmoid', 'activation_other',
                'optimizer_adam', 'optimizer_sgd', 'optimizer_other'
            ]

            importance_dict = {}
            for i, name in enumerate(feature_names):
                if i < len(importances):
                    importance_dict[name] = float(importances[i])
                else:
                    importance_dict[name] = 0.0

            return importance_dict

        except Exception as e:
            tprint(f"⚠️ [PERFORMANCE] Error calculating feature importance: {e}", color="yellow")
            # Return default importance
            return {
                'num_parameters': 0.2,
                'depth': 0.2,
                'width': 0.2,
                'dropout': 0.1,
                'learning_rate': 0.1,
                'batch_size': 0.1,
                'regularization': 0.1
            }

    def save(self, filepath: str) -> bool:
        """Save the trained estimator."""
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'wb') as f:
                pickle.dump({
                    'model': self.model,
                    'feature_scaler': self.feature_scaler,
                    'config': self.config,
                    'is_trained': self.is_trained,
                    'training_history': self.training_history
                }, f)
            self.logger.info(f"✅ Performance estimator saved to {filepath}")
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to save performance estimator: {e}")
            return False

    def load(self, filepath: str) -> bool:
        """Load a trained estimator."""
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                self.model = data['model']
                self.feature_scaler = data['feature_scaler']
                self.config = data['config']
                self.is_trained = data['is_trained']
                self.training_history = data['training_history']
            self.logger.info(f"✅ Performance estimator loaded from {filepath}")
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to load performance estimator: {e}")
            return False

class NeuralPerformanceEstimator(BasePerformanceEstimator):
    """Performance estimator for neural architectures."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize neural performance estimator."""
        super().__init__(config)
        self.estimator_type = config.get('estimator_type', 'ensemble')
        self.logger.info("✅ Neural Performance Estimator initialized")

    def extract_features(self, architecture: Any) -> ArchitectureFeatures:
        """Extract features from a neural architecture."""
        from ..search_spaces import NeuralArchitecture, LayerType, ConnectionType

        if not isinstance(architecture, NeuralArchitecture):
            raise ValueError("Architecture must be a NeuralArchitecture instance")

        features = ArchitectureFeatures()

        # Basic architecture features
        features.n_layers = len(architecture.layers)
        features.total_parameters = sum(
            layer.hidden_size * layer.hidden_size for layer in architecture.layers
        )
        features.max_layer_size = max((layer.hidden_size for layer in architecture.layers), default=0)
        features.avg_layer_size = sum(layer.hidden_size for layer in architecture.layers) / max(len(architecture.layers), 1)
        features.depth = len(architecture.layers)
        features.width = max((layer.hidden_size for layer in architecture.layers), default=0)

        # Neural network specific features
        for layer in architecture.layers:
            if layer.layer_type == LayerType.CONV1D:
                features.n_conv_layers += 1
            elif layer.layer_type in [LayerType.LSTM, LayerType.GRU]:
                features.n_recurrent_layers += 1
            elif layer.layer_type == LayerType.ATTENTION:
                features.n_attention_layers += 1
            elif layer.layer_type == LayerType.RESIDUAL_BLOCK:
                features.has_residual_connections = True
            elif layer.layer_type == LayerType.BATCH_NORM:
                features.has_batch_norm = True
            elif layer.layer_type == LayerType.DROPOUT:
                features.has_dropout = True

        # Connection features
        features.n_connections = len(architecture.connections)
        for conn in architecture.connections:
            if conn[2] == ConnectionType.RESIDUAL:
                features.n_residual_connections += 1
            elif conn[2] == ConnectionType.SKIP:
                features.n_skip_connections += 1

        features.connection_density = len(architecture.connections) / max(len(architecture.layers) * len(architecture.layers), 1)

        # Complexity and resource estimates
        features.complexity_score = architecture.estimated_complexity
        features.memory_estimate = architecture.estimated_memory_usage
        features.training_time_estimate = architecture.estimated_training_time

        # Activation complexity (simplified)
        activations = [layer.activation for layer in architecture.layers if layer.activation]
        if activations:
            features.activation_complexity = len(set(act.value if act else 'none' for act in activations)) / len(activations)

        # Architecture metadata
        features.architecture_type = "neural"
        features.architecture_hash = hash(str(architecture.layers) + str(architecture.connections))

        return features

    def predict_performance(self, architecture: Any) -> PerformancePrediction:
        """Predict performance of a neural architecture."""
        if not self.is_trained:
            raise RuntimeError("Performance estimator must be trained before making predictions")

        start_time = time.time()

        # Extract features
        features = self.extract_features(architecture)

        # Convert to feature vector
        feature_vector = self._features_to_vector(features)

        # Scale features
        feature_vector_scaled = self.feature_scaler.transform([feature_vector])

        # Make prediction
        if self.estimator_type == 'ensemble':
            predictions = []
            for model in self.model:
                pred = model.predict(feature_vector_scaled)[0]
                predictions.append(pred)

            mean_pred = np.mean(predictions)
            std_pred = np.std(predictions)
            confidence_interval = (mean_pred - 1.96 * std_pred, mean_pred + 1.96 * std_pred)

            # Feature importance (average across models)
            feature_importance = {}
            for i, feature_name in enumerate(self._get_feature_names()):
                importance = np.mean([model.feature_importances_[i] if hasattr(model, 'feature_importances_') else 0.0
                                    for model in self.model])
                feature_importance[feature_name] = importance

        else:
            pred = self.model.predict(feature_vector_scaled)[0]
            std_pred = np.sqrt(self.model.predict(feature_vector_scaled, return_std=True)[1][0]) if hasattr(self.model, 'predict') and hasattr(self.model, 'return_std') else 0.1
            confidence_interval = (pred - 1.96 * std_pred, pred + 1.96 * std_pred)

            # Feature importance
            if hasattr(self.model, 'feature_importances_'):
                feature_importance = dict(zip(self._get_feature_names(), self.model.feature_importances_))
            else:
                feature_importance = {name: 0.0 for name in self._get_feature_names()}

        prediction_time = time.time() - start_time

        return PerformancePrediction(
            predicted_performance=mean_pred if self.estimator_type == 'ensemble' else pred,
            confidence_interval=confidence_interval,
            feature_importance=feature_importance,
            prediction_time=prediction_time,
            model_used=self.estimator_type,
            metadata={'architecture_type': 'neural', 'n_layers': features.n_layers}
        )

    def train(self, architectures: List[ArchitectureFeatures], performances: List[float]) -> Dict[str, float]:
        """Train the neural performance estimator."""
        try:
            self.logger.info("🧠 Training neural performance estimator...")

            # Convert features to matrix
            feature_matrix = np.array([self._features_to_vector(arch) for arch in architectures])
            target_values = np.array(performances)

            # Scale features
            feature_matrix_scaled = self.feature_scaler.fit_transform(feature_matrix)

            # Train models based on type
            if self.estimator_type == 'ensemble':
                # Train multiple models
                models = [
                    RandomForestRegressor(n_estimators=100, random_state=42),
                    GradientBoostingRegressor(n_estimators=100, random_state=42),
                    Ridge(alpha=1.0, random_state=42),
                    SVR(kernel='rbf', C=1.0)
                ]

                trained_models = []
                cv_scores = []

                for model in models:
                    model.fit(feature_matrix_scaled, target_values)
                    scores = cross_val_score(model, feature_matrix_scaled, target_values, cv=5, scoring='r2')
                    cv_scores.append(scores.mean())
                    trained_models.append(model)

                self.model = trained_models
                self.logger.info(f"✅ Ensemble trained with CV R²: {cv_scores}")

            elif self.estimator_type == 'gaussian_process':
                kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
                self.model = GaussianProcessRegressor(kernel=kernel, random_state=42)
                self.model.fit(feature_matrix_scaled, target_values)
                self.logger.info("✅ Gaussian Process trained")

            elif self.estimator_type == 'random_forest':
                self.model = RandomForestRegressor(n_estimators=100, random_state=42)
                self.model.fit(feature_matrix_scaled, target_values)
                self.logger.info("✅ Random Forest trained")

            elif self.estimator_type == 'gradient_boosting':
                self.model = GradientBoostingRegressor(n_estimators=100, random_state=42)
                self.model.fit(feature_matrix_scaled, target_values)
                self.logger.info("✅ Gradient Boosting trained")

            else:  # linear_regression
                self.model = LinearRegression()
                self.model.fit(feature_matrix_scaled, target_values)
                self.logger.info("✅ Linear Regression trained")

            # Store training history
            self.training_history = TrainingHistory(architectures, performances)

            self.is_trained = True

            # Calculate training metrics
            train_predictions = self.model.predict(feature_matrix_scaled) if not isinstance(self.model, list) else np.mean([m.predict(feature_matrix_scaled) for m in self.model], axis=0)
            training_metrics = {
                'r2_score': r2_score(target_values, train_predictions),
                'mse': mean_squared_error(target_values, train_predictions),
                'n_samples': len(architectures),
                'n_features': feature_matrix_scaled.shape[1]
            }

            self.logger.info(f"✅ Neural performance estimator trained: {training_metrics}")
            return training_metrics

        except Exception as e:
            self.logger.error(f"❌ Failed to train neural performance estimator: {e}")
            raise

    def _features_to_vector(self, features: ArchitectureFeatures) -> List[float]:
        """Convert architecture features to a numerical vector."""
        return [
            features.n_layers,
            features.total_parameters,
            features.max_layer_size,
            features.avg_layer_size,
            features.depth,
            features.width,
            features.complexity_score,
            features.memory_estimate,
            features.training_time_estimate,
            features.n_conv_layers,
            features.n_recurrent_layers,
            features.n_attention_layers,
            1.0 if features.has_residual_connections else 0.0,
            1.0 if features.has_batch_norm else 0.0,
            1.0 if features.has_dropout else 0.0,
            features.activation_complexity,
            features.n_connections,
            features.n_skip_connections,
            features.n_residual_connections,
            features.connection_density
        ]

    def _get_feature_names(self) -> List[str]:
        """Get names of features used in the estimator."""
        return [
            'n_layers', 'total_parameters', 'max_layer_size', 'avg_layer_size',
            'depth', 'width', 'complexity_score', 'memory_estimate', 'training_time_estimate',
            'n_conv_layers', 'n_recurrent_layers', 'n_attention_layers', 'has_residual',
            'has_batch_norm', 'has_dropout', 'activation_complexity', 'n_connections',
            'n_skip_connections', 'n_residual_connections', 'connection_density'
        ]

class TreePerformanceEstimator(BasePerformanceEstimator):
    """Performance estimator for tree architectures."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize tree performance estimator."""
        super().__init__(config)
        self.estimator_type = config.get('estimator_type', 'ensemble')
        self.logger.info("✅ Tree Performance Estimator initialized")

    def extract_features(self, architecture: Any) -> ArchitectureFeatures:
        """Extract features from a tree architecture."""

        if not isinstance(architecture, TreeArchitecture):
            raise ValueError("Architecture must be a TreeArchitecture instance")

        features = ArchitectureFeatures()

        # Basic architecture features
        features.n_layers = len(architecture.trees)
        features.depth = max((tree.max_depth or 10 for tree in architecture.trees), default=10)
        features.width = max((tree.n_estimators for tree in architecture.trees), default=1)

        # Tree specific features
        features.n_trees = len(architecture.trees)
        features.max_tree_depth = max((tree.max_depth or 10 for tree in architecture.trees), default=10)
        features.avg_tree_depth = sum(tree.max_depth or 10 for tree in architecture.trees) / max(len(architecture.trees), 1)

        for tree in architecture.trees:
            if tree.tree_type == LayerType.RANDOM_FOREST:
                features.has_bagging = True
            elif tree.tree_type in [LayerType.GRADIENT_BOOSTING, LayerType.XGBOOST]:
                features.has_boosting = True

        features.ensemble_method = architecture.ensemble_method

        # Complexity estimates
        features.complexity_score = architecture.estimated_complexity
        features.memory_estimate = architecture.estimated_memory_usage
        features.training_time_estimate = architecture.estimated_training_time

        # Architecture metadata
        features.architecture_type = "tree"
        features.architecture_hash = hash(str([str(tree.__dict__) for tree in architecture.trees]) + architecture.ensemble_method)

        return features

    def predict_performance(self, architecture: Any) -> PerformancePrediction:
        """Predict performance of a tree architecture."""
        if not self.is_trained:
            raise RuntimeError("Performance estimator must be trained before making predictions")

        start_time = time.time()

        # Extract features
        features = self.extract_features(architecture)

        # Convert to feature vector
        feature_vector = self._features_to_vector(features)

        # Scale features
        feature_vector_scaled = self.feature_scaler.transform([feature_vector])

        # Make prediction
        if self.estimator_type == 'ensemble':
            predictions = []
            for model in self.model:
                pred = model.predict(feature_vector_scaled)[0]
                predictions.append(pred)

            mean_pred = np.mean(predictions)
            std_pred = np.std(predictions)
            confidence_interval = (mean_pred - 1.96 * std_pred, mean_pred + 1.96 * std_pred)

            # Feature importance (average across models)
            feature_importance = {}
            for i, feature_name in enumerate(self._get_feature_names()):
                importance = np.mean([model.feature_importances_[i] if hasattr(model, 'feature_importances_') else 0.0
                                    for model in self.model])
                feature_importance[feature_name] = importance

        else:
            pred = self.model.predict(feature_vector_scaled)[0]
            std_pred = 0.1  # Simplified for non-GP models
            confidence_interval = (pred - 1.96 * std_pred, pred + 1.96 * std_pred)

            # Feature importance
            if hasattr(self.model, 'feature_importances_'):
                feature_importance = dict(zip(self._get_feature_names(), self.model.feature_importances_))
            else:
                feature_importance = {name: 0.0 for name in self._get_feature_names()}

        prediction_time = time.time() - start_time

        return PerformancePrediction(
            predicted_performance=mean_pred if self.estimator_type == 'ensemble' else pred,
            confidence_interval=confidence_interval,
            feature_importance=feature_importance,
            prediction_time=prediction_time,
            model_used=self.estimator_type,
            metadata={'architecture_type': 'tree', 'n_trees': features.n_trees}
        )

    def train(self, architectures: List[ArchitectureFeatures], performances: List[float]) -> Dict[str, float]:
        """Train the tree performance estimator."""
        try:
            self.logger.info("🌳 Training tree performance estimator...")

            # Convert features to matrix
            feature_matrix = np.array([self._features_to_vector(arch) for arch in architectures])
            target_values = np.array(performances)

            # Scale features
            feature_matrix_scaled = self.feature_scaler.fit_transform(feature_matrix)

            # Train models based on type
            if self.estimator_type == 'ensemble':
                # Train multiple models
                models = [
                    RandomForestRegressor(n_estimators=100, random_state=42),
                    GradientBoostingRegressor(n_estimators=100, random_state=42),
                    Ridge(alpha=1.0, random_state=42),
                    SVR(kernel='rbf', C=1.0)
                ]

                trained_models = []
                cv_scores = []

                for model in models:
                    model.fit(feature_matrix_scaled, target_values)
                    scores = cross_val_score(model, feature_matrix_scaled, target_values, cv=5, scoring='r2')
                    cv_scores.append(scores.mean())
                    trained_models.append(model)

                self.model = trained_models
                self.logger.info(f"✅ Ensemble trained with CV R²: {cv_scores}")

            elif self.estimator_type == 'random_forest':
                self.model = RandomForestRegressor(n_estimators=100, random_state=42)
                self.model.fit(feature_matrix_scaled, target_values)
                self.logger.info("✅ Random Forest trained")

            elif self.estimator_type == 'gradient_boosting':
                self.model = GradientBoostingRegressor(n_estimators=100, random_state=42)
                self.model.fit(feature_matrix_scaled, target_values)
                self.logger.info("✅ Gradient Boosting trained")

            else:  # linear_regression
                self.model = LinearRegression()
                self.model.fit(feature_matrix_scaled, target_values)
                self.logger.info("✅ Linear Regression trained")

            # Store training history
            self.training_history = TrainingHistory(architectures, performances)

            self.is_trained = True

            # Calculate training metrics
            if isinstance(self.model, list):
                train_predictions = np.mean([m.predict(feature_matrix_scaled) for m in self.model], axis=0)
            else:
                train_predictions = self.model.predict(feature_matrix_scaled)

            training_metrics = {
                'r2_score': r2_score(target_values, train_predictions),
                'mse': mean_squared_error(target_values, train_predictions),
                'n_samples': len(architectures),
                'n_features': feature_matrix_scaled.shape[1]
            }

            self.logger.info(f"✅ Tree performance estimator trained: {training_metrics}")
            return training_metrics

        except Exception as e:
            self.logger.error(f"❌ Failed to train tree performance estimator: {e}")
            raise

    def _features_to_vector(self, features: ArchitectureFeatures) -> List[float]:
        """Convert architecture features to a numerical vector."""
        return [
            features.n_layers,
            features.depth,
            features.width,
            features.complexity_score,
            features.memory_estimate,
            features.training_time_estimate,
            features.n_trees,
            features.max_tree_depth,
            features.avg_tree_depth,
            1.0 if features.has_boosting else 0.0,
            1.0 if features.has_bagging else 0.0,
            1.0 if features.ensemble_method == 'stacking' else 0.0,
            1.0 if features.ensemble_method == 'averaging' else 0.0,
            1.0 if features.ensemble_method == 'voting' else 0.0
        ]

    def _get_feature_names(self) -> List[str]:
        """Get names of features used in the estimator."""
        return [
            'n_layers', 'depth', 'width', 'complexity_score', 'memory_estimate',
            'training_time_estimate', 'n_trees', 'max_tree_depth', 'avg_tree_depth',
            'has_boosting', 'has_bagging', 'is_stacking', 'is_averaging', 'is_voting'
        ]

class UnifiedPerformanceEstimator:
    """Unified performance estimator that handles both neural and tree architectures."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize unified performance estimator."""
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize specialized estimators
        self.neural_estimator = NeuralPerformanceEstimator(config.get('neural_config', {}))
        self.tree_estimator = TreePerformanceEstimator(config.get('tree_config', {}))

        self.logger.info("✅ Unified Performance Estimator initialized")

    def predict_performance(self, architecture: Any) -> PerformancePrediction:
        """Predict performance of any architecture type."""

        if isinstance(architecture, NeuralArchitecture):
            return self.neural_estimator.predict_performance(architecture)
        elif isinstance(architecture, TreeArchitecture):
            return self.tree_estimator.predict_performance(architecture)
        else:
            raise ValueError(f"Unsupported architecture type: {type(architecture)}")

    def train(self, architectures: List[Any], performances: List[float]) -> Dict[str, float]:
        """Train estimators on mixed architecture types."""
        neural_archs = []
        neural_perfs = []
        tree_archs = []
        tree_perfs = []

        # Separate architectures by type
        for arch, perf in zip(architectures, performances):
            if isinstance(arch, NeuralArchitecture):
                neural_archs.append(arch)
                neural_perfs.append(perf)
            elif isinstance(arch, TreeArchitecture):
                tree_archs.append(arch)
                tree_perfs.append(perf)

        # Train specialized estimators
        metrics = {}

        if neural_archs:
            neural_features = [self.neural_estimator.extract_features(arch) for arch in neural_archs]
            neural_metrics = self.neural_estimator.train(neural_features, neural_perfs)
            metrics['neural'] = neural_metrics

        if tree_archs:
            tree_features = [self.tree_estimator.extract_features(arch) for arch in tree_archs]
            tree_metrics = self.tree_estimator.train(tree_features, tree_perfs)
            metrics['tree'] = tree_metrics

        self.logger.info(f"✅ Unified performance estimator trained: {metrics}")
        return metrics

    def save(self, filepath: str) -> bool:
        """Save the unified estimator."""
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'wb') as f:
                pickle.dump({
                    'neural_estimator': self.neural_estimator,
                    'tree_estimator': self.tree_estimator,
                    'config': self.config
                }, f)
            self.logger.info(f"✅ Unified performance estimator saved to {filepath}")
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to save unified performance estimator: {e}")
            return False

    def load(self, filepath: str) -> bool:
        """Load a unified estimator."""
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                self.neural_estimator = data['neural_estimator']
                self.tree_estimator = data['tree_estimator']
                self.config = data['config']
            self.logger.info(f"✅ Unified performance estimator loaded from {filepath}")
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to load unified performance estimator: {e}")
            return False

def create_neural_performance_estimator(config: Dict[str, Any]) -> NeuralPerformanceEstimator:
    """Create a neural performance estimator."""
    return NeuralPerformanceEstimator(config)

def create_tree_performance_estimator(config: Dict[str, Any]) -> TreePerformanceEstimator:
    """Create a tree performance estimator."""
    return TreePerformanceEstimator(config)

def create_unified_performance_estimator(config: Dict[str, Any]) -> UnifiedPerformanceEstimator:
    """Create a unified performance estimator."""
    return UnifiedPerformanceEstimator(config)
