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
            features = ArchitectureFeatures()
            
            # Basic architecture analysis
            if hasattr(architecture, 'layers'):
                # Neural network architecture
                features.architecture_type = "neural"
                features.n_layers = len(architecture.layers)
                
                # Analyze layers
                layer_sizes = []
                for layer in architecture.layers:
                    if hasattr(layer, 'units'):
                        layer_sizes.append(layer.units)
                    elif hasattr(layer, 'filters'):
                        layer_sizes.append(layer.filters)
                    else:
                        layer_sizes.append(128)  # Default size
                
                if layer_sizes:
                    features.max_layer_size = max(layer_sizes)
                    features.avg_layer_size = sum(layer_sizes) / len(layer_sizes)
                    features.total_parameters = sum(layer_sizes)
                
                # Count layer types
                for layer in architecture.layers:
                    layer_type = getattr(layer, 'type', 'unknown').lower()
                    if 'conv' in layer_type:
                        features.n_conv_layers += 1
                    elif 'lstm' in layer_type or 'gru' in layer_type:
                        features.n_recurrent_layers += 1
                    elif 'attention' in layer_type:
                        features.n_attention_layers += 1
                    elif 'batch_norm' in layer_type:
                        features.has_batch_norm = True
                    elif 'dropout' in layer_type:
                        features.has_dropout = True
                
                # Check for residual connections
                if hasattr(architecture, 'connections'):
                    features.n_connections = len(architecture.connections)
                    for conn in architecture.connections:
                        if conn.get('type') == 'residual':
                            features.has_residual_connections = True
                            features.n_residual_connections += 1
                        elif conn.get('type') == 'skip':
                            features.n_skip_connections += 1
                
                # Calculate complexity
                features.complexity_score = (
                    features.n_layers * 0.1 +
                    features.total_parameters * 0.001 +
                    features.n_conv_layers * 0.2 +
                    features.n_recurrent_layers * 0.3 +
                    features.n_attention_layers * 0.4
                )
                
            elif hasattr(architecture, 'trees'):
                # Tree architecture
                features.architecture_type = "tree"
                features.n_trees = len(architecture.trees)
                
                # Analyze trees
                tree_depths = []
                for tree in architecture.trees:
                    if hasattr(tree, 'depth'):
                        tree_depths.append(tree.depth)
                    else:
                        tree_depths.append(1)  # Default depth
                
                if tree_depths:
                    features.max_tree_depth = max(tree_depths)
                    features.avg_tree_depth = sum(tree_depths) / len(tree_depths)
                
                # Check ensemble methods
                for tree in architecture.trees:
                    tree_type = getattr(tree, 'tree_type', 'single').lower()
                    if 'gradient' in tree_type or 'xgboost' in tree_type:
                        features.has_boosting = True
                    elif 'random' in tree_type or 'forest' in tree_type:
                        features.has_bagging = True
                
                # Calculate complexity
                features.complexity_score = (
                    features.n_trees * 0.1 +
                    features.max_tree_depth * 0.2 +
                    (1.0 if features.has_boosting else 0.0) * 0.3 +
                    (1.0 if features.has_bagging else 0.0) * 0.2
                )
            
            # Calculate memory and training time estimates
            features.memory_estimate = features.total_parameters * 4 / (1024 * 1024)  # MB
            features.training_time_estimate = features.complexity_score * 10  # seconds
            
            # Generate architecture hash
            arch_str = str(architecture)
            features.architecture_hash = hashlib.md5(arch_str.encode()).hexdigest()[:16]
            
            return features
            
        except Exception as e:
            self.logger.error(f"Feature extraction failed: {e}")
            # Return default features
            return ArchitectureFeatures(
                architecture_type="unknown",
                complexity_score=1.0,
                memory_estimate=1.0,
                training_time_estimate=10.0
            )

    def predict_performance(self, architecture: Any) -> PerformancePrediction:
        """Predict performance of an architecture."""
        try:
            if not self.is_trained or self.model is None:
                # Return default prediction if not trained
                return PerformancePrediction(
                    predicted_value=0.5,
                    confidence=0.1,
                    prediction_time=0.001,
                    features_used=ArchitectureFeatures(),
                    model_type="untrained"
                )
            
            # Extract features from architecture
            features = self.extract_features(architecture)
            
            # Convert features to array for prediction
            feature_array = np.array([
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
                float(features.has_residual_connections),
                float(features.has_batch_norm),
                float(features.has_dropout),
                features.activation_complexity,
                features.n_trees,
                features.max_tree_depth,
                features.avg_tree_depth,
                float(features.has_boosting),
                float(features.has_bagging),
                features.n_connections,
                features.n_skip_connections,
                features.n_residual_connections,
                features.connection_density
            ]).reshape(1, -1)
            
            # Scale features if scaler is available
            if hasattr(self.feature_scaler, 'transform'):
                try:
                    feature_array = self.feature_scaler.transform(feature_array)
                except:
                    pass  # Use unscaled features if scaling fails
            
            # Make prediction
            start_time = time.time()
            predicted_value = self.model.predict(feature_array)[0]
            prediction_time = time.time() - start_time
            
            # Calculate confidence based on model type
            confidence = 0.5  # Default confidence
            if hasattr(self.model, 'predict_proba'):
                try:
                    proba = self.model.predict_proba(feature_array)[0]
                    confidence = max(proba) if len(proba) > 1 else proba[0]
                except:
                    pass
            elif hasattr(self.model, 'score'):
                try:
                    confidence = min(0.9, max(0.1, self.model.score(feature_array, [predicted_value])))
                except:
                    pass
            
            return PerformancePrediction(
                predicted_value=float(predicted_value),
                confidence=float(confidence),
                prediction_time=prediction_time,
                features_used=features,
                model_type=self.estimator_type.value
            )
            
        except Exception as e:
            self.logger.error(f"Performance prediction failed: {e}")
            # Return default prediction on error
            return PerformancePrediction(
                predicted_value=0.5,
                confidence=0.1,
                prediction_time=0.001,
                features_used=ArchitectureFeatures(),
                model_type="error"
            )

    def train(self, architectures: List[ArchitectureFeatures], performances: List[float]) -> Dict[str, float]:
        """Train the performance estimator."""
        try:
            if len(architectures) != len(performances):
                raise ValueError("Number of architectures must match number of performances")
            
            if len(architectures) < 2:
                self.logger.warning("Not enough data for training, using default model")
                return {"error": "insufficient_data"}
            
            # Convert features to arrays
            X = []
            for arch in architectures:
                feature_array = np.array([
                    arch.n_layers,
                    arch.total_parameters,
                    arch.max_layer_size,
                    arch.avg_layer_size,
                    arch.depth,
                    arch.width,
                    arch.complexity_score,
                    arch.memory_estimate,
                    arch.training_time_estimate,
                    arch.n_conv_layers,
                    arch.n_recurrent_layers,
                    arch.n_attention_layers,
                    float(arch.has_residual_connections),
                    float(arch.has_batch_norm),
                    float(arch.has_dropout),
                    arch.activation_complexity,
                    arch.n_trees,
                    arch.max_tree_depth,
                    arch.avg_tree_depth,
                    float(arch.has_boosting),
                    float(arch.has_bagging),
                    arch.n_connections,
                    arch.n_skip_connections,
                    arch.n_residual_connections,
                    arch.connection_density
                ])
                X.append(feature_array)
            
            X = np.array(X)
            y = np.array(performances)
            
            # Scale features
            X_scaled = self.feature_scaler.fit_transform(X)
            
            # Train model based on estimator type
            start_time = time.time()
            
            if self.estimator_type == EstimatorType.LINEAR_REGRESSION:
                self.model = LinearRegression()
            elif self.estimator_type == EstimatorType.RANDOM_FOREST:
                self.model = RandomForestRegressor(n_estimators=100, random_state=42)
            elif self.estimator_type == EstimatorType.GRADIENT_BOOSTING:
                self.model = GradientBoostingRegressor(n_estimators=100, random_state=42)
            elif self.estimator_type == EstimatorType.GAUSSIAN_PROCESS:
                kernel = ConstantKernel(1.0) * RBF(1.0) + WhiteKernel(1.0)
                self.model = GaussianProcessRegressor(kernel=kernel, random_state=42)
            elif self.estimator_type == EstimatorType.SUPPORT_VECTOR:
                self.model = SVR(kernel='rbf', C=1.0, gamma='scale')
            else:
                # Default to linear regression
                self.model = LinearRegression()
            
            # Fit the model
            self.model.fit(X_scaled, y)
            
            training_time = time.time() - start_time
            
            # Calculate training metrics
            y_pred = self.model.predict(X_scaled)
            mse = mean_squared_error(y, y_pred)
            r2 = r2_score(y, y_pred)
            
            # Cross-validation if enough data
            cv_scores = []
            if len(X_scaled) >= 5:
                try:
                    cv_scores = cross_val_score(self.model, X_scaled, y, cv=min(5, len(X_scaled)), scoring='r2')
                except:
                    pass
            
            # Update training history
            self.training_history.architectures.extend(architectures)
            self.training_history.performances.extend(performances)
            self.training_history.training_times.append(training_time)
            
            self.is_trained = True
            
            return {
                "training_time": training_time,
                "mse": mse,
                "r2": r2,
                "cv_mean": np.mean(cv_scores) if cv_scores else 0.0,
                "cv_std": np.std(cv_scores) if cv_scores else 0.0,
                "n_samples": len(architectures)
            }
            
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            return {"error": str(e)}

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