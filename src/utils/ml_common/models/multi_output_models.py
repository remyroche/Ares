"""
Multi-Output Models for Stacking Ensemble

This module provides comprehensive multi-output model support for the Analyst (5m) and
Tactician (1m) stacking ensemble system.

Key Features:
- MultiOutputConfig dataclass for 4-output configuration
- MultiOutputModel abstract base class
- MultiOutputStackingModel implementation
- Data preparation utilities for multi-output targets
- Prediction combination logic
- M1 hardware optimization integration

Now inherits from the production-ready MultiOutputModel in core module.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import logging
import time
from datetime import datetime

# Import the production-ready MultiOutputModel
from src.core.abstract_base_classes import MultiOutputModel as ProductionMultiOutputModel

# M1 Optimization imports
from src.utils.hardware.m1_memory_optimizer import get_m1_memory_optimizer, M1MemoryOptimizer
from src.utils.hardware.memory_optimization import get_memory_manager, MemoryMonitor

# Common utilities
from src.utils.common_operations import (
    safe_json_dump, safe_json_load, safe_file_exists, ensure_directory,
    safe_mean, safe_std, safe_float, safe_int, get_current_datetime,
    safe_append, safe_extend, safe_dict_get, safe_lower, safe_upper,
    format_datetime, validate_file_path, get_file_size
)
from src.utils.math_validation import (
    safe_divide, safe_log, safe_sqrt, safe_power, validate_finite,
    validate_positive, validate_range, safe_kelly_calculation,
    safe_weighted_average, safe_percentage_change, MathValidationError
)
from src.core.decorators import (
    handles_errors, validates, traced, log_execution_time,
    timeout, error_boundary, compose
)
from src.core.errors import (
    ValidationError, DataIntegrityError, TimeoutError
)

# CV and cloning utilities
from sklearn.model_selection import TimeSeriesSplit
from sklearn.base import clone as skl_clone, BaseEstimator
from sklearn.linear_model import LinearRegression
import inspect

# Purged CV (if available)
try:
    from src.utils.ml_common.validation.consolidated_cv import ConsolidatedCrossValidator as PurgedKFoldTime  # type: ignore
    _PURGED_AVAILABLE = True
except Exception:
    _PURGED_AVAILABLE = False

# Overfitting prevention (regularization/early stopping settings)
try:
    from src.utils.ml_common.optimization.overfitting_prevention import (
        OverfittingPrevention,
        OverfittingPreventionConfig,
    )
    _OVERFITTING_AVAILABLE = True
except Exception:
    _OVERFITTING_AVAILABLE = False
    OverfittingPrevention = None  # type: ignore
    OverfittingPreventionConfig = None  # type: ignore

logger = logging.getLogger(__name__)

@dataclass
class MultiOutputConfig:
    """Configuration for multi-output models."""
    # Basic configuration
    model_name: str
    n_outputs: int = 4
    output_names: List[str] = field(default_factory=lambda: ["output_1", "output_2", "output_3", "output_4"])
    # Output format for predictions: 'array' or 'dict'
    output_format: str = 'array'

    # Model configuration
    base_models: Dict[str, Any] = field(default_factory=dict)
    meta_model: Optional[Any] = None

    # Training configuration
    enable_cross_validation: bool = True
    cv_folds: int = 5
    enable_early_stopping: bool = True
    early_stopping_patience: int = 10

    # Multi-output specific settings
    output_weights: Optional[List[float]] = None
    output_loss_weights: Optional[List[float]] = None
    enable_output_correlation: bool = True
    correlation_threshold: float = 0.7

    # M1 optimization settings
    enable_gpu_acceleration: bool = True
    enable_memory_optimization: bool = True
    enable_parallel_processing: bool = True
    memory_limit_gb: float = 8.0
    max_workers: Optional[int] = None

    # Performance settings
    enable_caching: bool = True
    cache_size_mb: int = 100
    enable_profiling: bool = False

    # Validation settings
    validation_split: float = 0.2
    test_split: float = 0.1
    enable_online_learning: bool = False

    # Output settings
    save_models: bool = True
    save_predictions: bool = True
    generate_reports: bool = True

@dataclass
class MultiOutputResult:
    """Result from multi-output model operations."""
    # Basic info
    model_name: str
    n_outputs: int
    output_names: List[str]
    created_at: datetime
    total_duration: float

    # Predictions
    predictions: np.ndarray = field(default_factory=lambda: np.array([]))
    prediction_probabilities: Optional[np.ndarray] = None
    confidence_scores: np.ndarray = field(default_factory=lambda: np.array([]))

    # Performance metrics
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    per_output_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Model characteristics
    model_weights: Optional[np.ndarray] = None
    output_correlations: Optional[np.ndarray] = None
    feature_importance: Optional[Dict[str, Any]] = None

    # Metadata
    config: MultiOutputConfig = field(default_factory=MultiOutputConfig)
    execution_time: float = 0.0
    memory_usage_mb: float = 0.0
    optimization_used: List[str] = field(default_factory=list)

class MultiOutputModel(ProductionMultiOutputModel):
    """
    Abstract base class for multi-output models.
    
    This class provides a comprehensive interface for multi-output machine learning models
    with production-ready features including error handling, validation, logging, and
    hardware optimization.
    """

    def __init__(self, config: MultiOutputConfig):
        """Initialize the multi-output model."""
        self.config = config
        self.logger = logger.getChild(f'MultiOutputModel.{config.model_name}')
        self.logger.info(f"🚀 Initializing MultiOutputModel: {config.model_name}")

        # Initialize M1 optimizers
        self.m1_gpu = get_m1_memory_optimizer() if config.enable_gpu_acceleration else None
        self.m1_memory = get_m1_memory_optimizer(
            memory_limit_gb=config.memory_limit_gb
        ) if config.enable_memory_optimization else None
        self.m1_cpu = get_memory_manager() if config.enable_parallel_processing else None

        # Model state
        self.is_fitted = False
        self.output_weights = config.output_weights or [1.0] * config.n_outputs
        self.output_loss_weights = config.output_loss_weights or [1.0] * config.n_outputs

        # Performance tracking
        self.training_history: List[Dict[str, Any]] = []
        self.prediction_history: List[Dict[str, Any]] = []

        self.logger.info(f"✅ MultiOutputModel initialized with {config.n_outputs} outputs")
        self.logger.info(f"📊 Output names: {config.output_names}")
        self.logger.info(f"⚖️ Output weights: {self.output_weights}")

    def _create_single_output_model(self, output_index: int, target: np.ndarray) -> Any:
        """
        Create a single-output model for the specified output.
        
        Args:
            output_index: Index of the output (0 to n_outputs-1)
            target: Target values for this output
            
        Returns:
            Fitted single-output model
        """
        # Default implementation - subclasses should override
        try:
            # Determine if this is classification or regression
            is_classification = self._determine_task_type(target.reshape(-1, 1))
            
            if is_classification:
                from sklearn.linear_model import LogisticRegression
                model = LogisticRegression(max_iter=1000, random_state=42)
            else:
                from sklearn.ensemble import RandomForestRegressor
                model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            
            self.logger.info(f"Created default {type(model).__name__} for output {output_index}")
            return model
            
        except Exception as e:
            self.logger.error(f"Failed to create model for output {output_index}: {e}")
            # Fallback to simple model
            try:
                from sklearn.linear_model import LinearRegression
                return LinearRegression()
            except Exception as e:
                self.logger.error(f"Failed to create any model for output {output_index}: {e}")
                raise RuntimeError(f"Unable to create model for output {output_index}: {e}")

    def _calculate_output_weights(self, X: np.ndarray, y: np.ndarray) -> List[float]:
        """
        Calculate optimal weights for each output.
        
        Args:
            X: Input features
            y: Target values
            
        Returns:
            List of weights for each output
        """
        # Default implementation - subclasses should override
        try:
            # Simple equal weighting
            n_outputs = y.shape[1] if len(y.shape) > 1 else 1
            weights = [1.0 / n_outputs] * n_outputs
            
            self.logger.info(f"Using equal weights for {n_outputs} outputs: {weights}")
            return weights
            
        except Exception as e:
            self.logger.error(f"Failed to calculate output weights: {e}")
            return [1.0] * self.config.n_outputs

    def _validate_output_consistency(self, predictions: Dict[str, np.ndarray]) -> bool:
        """
        Validate that predictions from all outputs are consistent.
        
        Args:
            predictions: Dictionary of predictions from each output
            
        Returns:
            True if predictions are consistent, False otherwise
        """
        # Default implementation - subclasses should override
        try:
            if not predictions:
                self.logger.warning("No predictions provided for validation")
                return False
            
            # Check that all predictions have the same length
            lengths = [len(pred) for pred in predictions.values()]
            if len(set(lengths)) > 1:
                self.logger.error(f"Inconsistent prediction lengths: {lengths}")
                return False
            
            # Check for NaN values
            for output_name, pred in predictions.items():
                if np.any(np.isnan(pred)):
                    self.logger.error(f"NaN values found in {output_name} predictions")
                    return False
            
            self.logger.info("Output consistency validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Output consistency validation failed: {e}")
            return False

    def _calculate_confidence_scores(self, predictions: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Calculate confidence scores for each output prediction.
        
        Args:
            predictions: Dictionary of predictions from each output
            
        Returns:
            Dictionary of confidence scores for each output
        """
        # Default implementation - subclasses should override
        try:
            confidence_scores = {}
            
            for output_name, pred in predictions.items():
                # Simple confidence based on prediction magnitude
                pred_abs = np.abs(pred)
                if np.max(pred_abs) > 0:
                    confidence = pred_abs / np.max(pred_abs)
                else:
                    confidence = np.ones_like(pred)
                
                confidence_scores[output_name] = confidence
                self.logger.debug(f"Calculated confidence scores for {output_name}: mean={np.mean(confidence):.3f}")
            
            return confidence_scores
            
        except Exception as e:
            self.logger.error(f"Failed to calculate confidence scores: {e}")
            # Return default confidence scores
            return {name: np.ones(len(pred)) for name, pred in predictions.items()}

    def _get_feature_importance(self, output_index: int) -> Optional[np.ndarray]:
        """
        Get feature importance for a specific output.
        
        Args:
            output_index: Index of the output
            
        Returns:
            Feature importance array or None if not available
        """
        # Default implementation - subclasses should override
        try:
            if not hasattr(self, 'models') or not self.models:
                return None
            
            output_name = self.config.output_names[output_index] if output_index < len(self.config.output_names) else f"output_{output_index}"
            
            if output_name not in self.models:
                return None
            
            model = self.models[output_name]
            
            # Try to get feature importance from the model
            if hasattr(model, 'feature_importances_'):
                return model.feature_importances_
            elif hasattr(model, 'coef_'):
                return np.abs(model.coef_)
            else:
                self.logger.debug(f"No feature importance available for {output_name}")
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to get feature importance for output {output_index}: {e}")
            return None

    def _get_model_metadata(self) -> Dict[str, Any]:
        """
        Get metadata about the model.
        
        Returns:
            Dictionary containing model metadata
        """
        # Default implementation - subclasses should override
        try:
            metadata = {
                'model_type': self.__class__.__name__,
                'n_outputs': self.config.n_outputs,
                'output_names': self.config.output_names,
                'is_fitted': self.is_fitted,
                'output_weights': self.output_weights,
                'output_loss_weights': self.output_loss_weights,
                'created_at': time.time(),
                'training_history_length': len(self.training_history),
                'prediction_history_length': len(self.prediction_history)
            }
            
            # Add model-specific metadata if available
            if hasattr(self, 'models') and self.models:
                metadata['individual_models'] = {
                    name: type(model).__name__ for name, model in self.models.items()
                }
            
            return metadata
            
        except Exception as e:
            self.logger.error(f"Failed to get model metadata: {e}")
            return {'error': str(e)}

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'MultiOutputModel':
        """Fit the multi-output model."""
        try:
            self.logger.info(f"🔧 Fitting multi-output model with {X.shape[0]} samples, {X.shape[1]} features")

            # Validate inputs
            if not self.validate_inputs(X, y):
                raise ValueError("Input validation failed")

            # Reshape y if needed for multi-output
            y_reshaped = self._reshape_targets(y)
            self.logger.debug(f"📊 Target shape after reshaping: {y_reshaped.shape}")

            # Initialize individual models for each output
            self.models = {}
            self.is_fitted = True

            # Determine if this is a classification or regression task
            self.is_classification = self._determine_task_type(y_reshaped)

            # Train models for each output
            for i in range(self.config.n_outputs):
                output_name = self.config.output_names[i] if i < len(self.config.output_names) else f"output_{i}"
                output_target = y_reshaped[:, i]

                self.logger.info(f"🤖 Training model for {output_name} (output {i+1}/{self.config.n_outputs})")

                # Create model for this output
                model = self._create_single_output_model(i, output_target)

                # Train the model
                model.fit(X, output_target)

                self.models[output_name] = model

                # Log training progress
                if hasattr(model, 'score'):
                    train_score = model.score(X, output_target)
                    self.logger.info(f"✅ {output_name} model trained - Training score: {train_score:.4f}")
                else:
                    self.logger.info(f"✅ {output_name} model trained successfully")

            self.logger.info(f"✅ Multi-output model fitted successfully with {len(self.models)} individual models")
            return self

        except Exception as e:
            self.logger.error(f"❌ Model fitting failed: {e}")
            raise

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions for all outputs."""
        try:
            if not self.is_fitted:
                raise ValueError("Model must be fitted before making predictions")

            if not self.validate_inputs(X):
                raise ValueError("Input validation failed")

            self.logger.info(f"🔮 Making predictions for {X.shape[0]} samples, {X.shape[1]} features")

            # Collect predictions from all models
            predictions = []
            prediction_details = {}

            for i, (output_name, model) in enumerate(self.models.items()):
                self.logger.debug(f"📊 Predicting {output_name} (output {i+1}/{len(self.models)})")

                # Make prediction for this output
                output_pred = model.predict(X)

                # Apply any output-specific transformations
                if hasattr(self.config, 'output_transforms') and output_name in self.config.output_transforms:
                    transform_func = self.config.output_transforms[output_name]
                    output_pred = transform_func(output_pred)
                    self.logger.debug(f"🔄 Applied transformation to {output_name}")

                predictions.append(output_pred)
                prediction_details[output_name] = {
                    'shape': output_pred.shape,
                    'range': (output_pred.min(), output_pred.max()) if len(output_pred) > 0 else None
                }

            # Combine predictions into multi-output format
            if getattr(self.config, 'output_format', 'array') == 'array':
                result = np.column_stack(predictions)
            else:
                # Return as dictionary
                result = dict(zip(self.config.output_names, predictions))

            self.logger.info(f"✅ Predictions completed - Shape: {result.shape if hasattr(result, 'shape') else len(result)}")
            self.logger.debug(f"📊 Prediction details: {prediction_details}")

            return result

        except Exception as e:
            self.logger.error(f"❌ Prediction failed: {e}")
            raise

    def predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        """Make probability predictions for all outputs."""
        try:
            if not self.is_fitted:
                raise ValueError("Model must be fitted before making predictions")

            if not self.is_classification:
                self.logger.info("ℹ️ predict_proba called on regression model - returning None")
                return None

            if not self.validate_inputs(X):
                raise ValueError("Input validation failed")

            self.logger.info(f"🎲 Making probability predictions for {X.shape[0]} samples")

            # Collect probability predictions from all models
            probas = []
            proba_details = {}

            for i, (output_name, model) in enumerate(self.models.items()):
                self.logger.debug(f"🎯 Predicting probabilities for {output_name}")

                if hasattr(model, 'predict_proba'):
                    output_proba = model.predict_proba(X)
                    probas.append(output_proba)

                    proba_details[output_name] = {
                        'shape': output_proba.shape,
                        'classes': getattr(model, 'classes_', None)
                    }
                else:
                    # If model doesn't support predict_proba, create pseudo-probabilities
                    predictions = model.predict(X)
                    # Convert predictions to pseudo-probabilities (simple approach)
                    if hasattr(model, 'classes_'):
                        # For classification models with classes
                        n_classes = len(model.classes_)
                        pseudo_proba = np.zeros((len(predictions), n_classes))
                        for j, pred in enumerate(predictions):
                            class_idx = np.where(model.classes_ == pred)[0]
                            if len(class_idx) > 0:
                                pseudo_proba[j, class_idx[0]] = 1.0
                            else:
                                # Fallback: assign equal probability
                                pseudo_proba[j, :] = 1.0 / n_classes
                    else:
                        # For regression models, create binary-like probabilities
                        pseudo_proba = np.column_stack([1 - predictions, predictions])
                        pseudo_proba = pseudo_proba / pseudo_proba.sum(axis=1, keepdims=True)

                    probas.append(pseudo_proba)
                    proba_details[output_name] = {
                        'shape': pseudo_proba.shape,
                        'note': 'Pseudo-probabilities generated'
                    }

            # Combine probabilities into multi-output format
            if getattr(self.config, 'output_format', 'array') == 'array':
                result = np.concatenate(probas, axis=1)
            else:
                result = dict(zip(self.config.output_names, probas))

            self.logger.info(f"✅ Probability predictions completed - Shape: {result.shape if hasattr(result, 'shape') else len(result)}")
            self.logger.debug(f"📊 Probability details: {proba_details}")

            return result

        except Exception as e:
            self.logger.error(f"❌ Probability prediction failed: {e}")
            raise

    # ===== Helper and validation methods =====
    def validate_inputs(self, X: Any, y: Optional[np.ndarray] = None) -> bool:
        """Validate input features (and optionally targets) for basic shape and type correctness."""
        try:
            # Validate X
            if X is None:
                self.logger.error("❌ Input features X is None")
                return False
            if isinstance(X, pd.DataFrame):
                if X.shape[0] == 0 or X.shape[1] == 0:
                    self.logger.error("❌ Input DataFrame has zero rows or columns")
                    return False
            else:
                X_arr = np.asarray(X)
                if X_arr.ndim != 2 or X_arr.shape[0] == 0 or X_arr.shape[1] == 0:
                    self.logger.error(f"❌ Input array must be 2D with positive shape, got {X_arr.shape}")
                    return False

            # Validate y if provided
            if y is not None:
                y_arr = np.asarray(y)
                if y_arr.ndim not in (1, 2):
                    self.logger.error(f"❌ Target array must be 1D or 2D, got {y_arr.ndim}D")
                    return False
                # Check lengths
                n_samples = X.shape[0] if isinstance(X, np.ndarray) else (len(X) if hasattr(X, '__len__') else None)
                if n_samples is None or y_arr.shape[0] != n_samples:
                    self.logger.error("❌ X and y sample size mismatch")
                    return False
            return True
        except Exception as e:
            self.logger.error(f"❌ Input validation failed: {e}")
            return False

    def _reshape_targets(self, y: np.ndarray) -> np.ndarray:
        """Ensure targets are 2D and align with configured outputs; adjust config if needed."""
        y_arr = np.asarray(y)
        if y_arr.ndim == 1:
            if self.config.n_outputs <= 1:
                return y_arr.reshape(-1, 1)
            # Duplicate single target across outputs for compatibility
            return np.column_stack([y_arr] * self.config.n_outputs)
        elif y_arr.ndim == 2:
            # Align configuration with provided outputs if mismatched
            if y_arr.shape[1] != self.config.n_outputs:
                self.logger.warning(f"⚠️ Adjusting n_outputs from {self.config.n_outputs} to {y_arr.shape[1]}")
                self.config.n_outputs = int(y_arr.shape[1])
                self.config.output_names = [f"output_{i+1}" for i in range(self.config.n_outputs)]
            return y_arr
        else:
            raise ValueError(f"Target array must be 1D or 2D, got shape {y_arr.shape}")

    def _determine_task_type(self, y_2d: np.ndarray) -> bool:
        """Determine if the problem is classification based on target characteristics.
        Returns True for classification, False for regression.
        """
        try:
            # Consider first output to determine task type
            y_first = y_2d[:, 0]
            unique_vals = np.unique(y_first)
            # If all unique values are integers and few classes, treat as classification
            is_integer_like = np.all(np.equal(np.mod(unique_vals, 1), 0))
            return bool(is_integer_like and len(unique_vals) <= 10)
        except Exception:
            # Safe default: treat as regression
            return False

    def _create_single_output_model(self, output_index: int, y_target: np.ndarray) -> Any:
        """Create a reasonable default model for a single output."""
        try:
            if self._determine_task_type(y_target.reshape(-1, 1)):
                from sklearn.linear_model import LogisticRegression
                return LogisticRegression(max_iter=1000)
            else:
                from sklearn.ensemble import RandomForestRegressor
                return RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        except Exception as e:
            self.logger.warning(f"⚠️ Model creation failed for output {output_index}: {e}. Using LinearRegression fallback")
            try:
                from sklearn.linear_model import LinearRegression
                return LinearRegression()
            except Exception as e:
                self.logger.error(f"Failed to create any model for output {output_index}: {e}")
                raise RuntimeError(f"Unable to create model for output {output_index}: {e}")

    def validate_outputs(self, y: np.ndarray) -> bool:
        """Validate output data format."""
        if len(y.shape) == 1:
            # This should not happen if reshape was done properly in fit()
            self.logger.warning(f"⚠️ Received 1D output data with shape: {y.shape} - this should have been reshaped already")
            return False
        elif len(y.shape) == 2:
            if y.shape[1] != self.config.n_outputs:
                self.logger.error(f"❌ Expected {self.config.n_outputs} outputs, got {y.shape[1]}")
                return False
            self.logger.debug(f"✅ Output validation passed: {y.shape}")
            return True
        else:
            self.logger.error(f"❌ Output data must be 2D after reshaping, got {y.ndim}D shape: {y.shape}")
            return False

    def calculate_output_correlations(self, y: np.ndarray) -> np.ndarray:
        """Calculate correlations between outputs."""
        if not self.config.enable_output_correlation:
            return None

        try:
            correlations = np.corrcoef(y.T)
            self.logger.debug(f"📊 Output correlations calculated: {correlations.shape}")
            return correlations
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to calculate output correlations: {e}")
            return None

    def apply_output_weights(self, predictions: np.ndarray) -> np.ndarray:
        """Apply output weights to predictions."""
        if len(self.output_weights) != self.config.n_outputs:
            self.logger.warning("⚠️ Output weights length mismatch, using equal weights")
            weights = np.ones(self.config.n_outputs) / self.config.n_outputs
        else:
            weights = np.array(self.output_weights)

        # Normalize weights
        weights = weights / weights.sum()

        # Apply weights
        weighted_predictions = predictions * weights

        self.logger.debug(f"⚖️ Applied output weights: {weights}")
        return weighted_predictions

    def calculate_confidence_scores(self, predictions: np.ndarray) -> np.ndarray:
        """Calculate confidence scores for predictions."""
        try:
            # Simple confidence based on prediction magnitude
            confidence = np.abs(predictions)
            confidence = confidence / (confidence.max(axis=0, keepdims=True) + 1e-8)

            # Average confidence across outputs
            avg_confidence = np.mean(confidence, axis=1)

            self.logger.debug(f"📊 Confidence scores calculated: {avg_confidence.shape}")
            return avg_confidence
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to calculate confidence scores: {e}")
            return np.ones(len(predictions))

    def get_feature_importance(self) -> Optional[Dict[str, Any]]:
        """Get feature importance if available."""
        try:
            if not self.is_fitted or not hasattr(self, 'models'):
                return None

            feature_importance = {}

            for output_name, model in self.models.items():
                try:
                    if hasattr(model, 'feature_importances_'):
                        # Tree-based models
                        feature_importance[output_name] = {
                            'type': 'tree_based',
                            'importances': model.feature_importances_.tolist(),
                            'feature_names': getattr(model, 'feature_names_in_', None)
                        }
                    elif hasattr(model, 'coef_'):
                        # Linear models
                        if len(model.coef_.shape) == 1:
                            feature_importance[output_name] = {
                                'type': 'linear',
                                'coefficients': model.coef_.tolist(),
                                'feature_names': getattr(model, 'feature_names_in_', None)
                            }
                        else:
                            # Multi-class linear models
                            feature_importance[output_name] = {
                                'type': 'linear_multi_class',
                                'coefficients': model.coef_.tolist(),
                                'feature_names': getattr(model, 'feature_names_in_', None)
                            }
                    elif hasattr(model, 'permutation_importance_'):
                        # Models with permutation importance
                        feature_importance[output_name] = {
                            'type': 'permutation',
                            'importances': model.permutation_importance_.tolist(),
                            'feature_names': getattr(model, 'feature_names_in_', None)
                        }
                    else:
                        # Calculate permutation importance as fallback
                        try:
                            from sklearn.inspection import permutation_importance
                            # Use a small sample for efficiency
                            n_samples = min(1000, len(self.X_train) if hasattr(self, 'X_train') and self.X_train is not None else 100)
                            if hasattr(self, 'X_train') and self.X_train is not None:
                                X_sample = self.X_train[:n_samples]
                                y_sample = self.y_train[:n_samples] if hasattr(self, 'y_train') and self.y_train is not None else None
                                if y_sample is not None:
                                    perm_importance = permutation_importance(
                                        model, X_sample, y_sample,
                                        n_repeats=5, random_state=42, n_jobs=1
                                    )
                                    feature_importance[output_name] = {
                                        'type': 'permutation_calculated',
                                        'importances': perm_importance.importances_mean.tolist(),
                                        'std': perm_importance.importances_std.tolist(),
                                        'feature_names': getattr(model, 'feature_names_in_', None)
                                    }
                        except Exception as e:
                            self.logger.debug(f"Could not calculate permutation importance for {output_name}: {e}")
                            continue

                except Exception as e:
                    self.logger.warning(f"Could not extract feature importance for {output_name}: {e}")
                    continue

            if feature_importance:
                self.logger.info(f"✅ Extracted feature importance for {len(feature_importance)} outputs")
                return feature_importance
            else:
                self.logger.warning("⚠️ No feature importance available for any output")
                return None

        except Exception as e:
            self.logger.error(f"❌ Feature importance extraction failed: {e}")
            return None

    def save_model(self, file_path: str) -> None:
        """Save the model to disk."""
        try:
            import pickle

            model_data = {
                'config': self.config,
                'is_fitted': self.is_fitted,
                'output_weights': self.output_weights,
                'output_loss_weights': self.output_loss_weights,
                'training_history': self.training_history,
                'prediction_history': self.prediction_history
            }

            with open(file_path, 'wb') as f:
                pickle.dump(model_data, f)

            self.logger.info(f"💾 Model saved to {file_path}")

        except Exception as e:
            self.logger.error(f"❌ Failed to save model: {e}")
            raise

    def load_model(self, file_path: str) -> None:
        """Load the model from disk."""
        try:

            with open(file_path, 'rb') as f:
                model_data = pickle.load(f)

            self.config = model_data['config']
            self.is_fitted = model_data['is_fitted']
            self.output_weights = model_data['output_weights']
            self.output_loss_weights = model_data['output_loss_weights']
            self.training_history = model_data['training_history']
            self.prediction_history = model_data['prediction_history']

            self.logger.info(f"📂 Model loaded from {file_path}")

        except Exception as e:
            self.logger.error(f"❌ Failed to load model: {e}")
            raise

class MultiOutputStackingModel(MultiOutputModel):
    """Multi-output stacking ensemble model."""

    def __init__(self, config: MultiOutputConfig):
        """Initialize the multi-output stacking model."""
        super().__init__(config)
        self.logger = logger.getChild(f'MultiOutputStackingModel.{config.model_name}')

        # Base models for each output
        self.base_models: Dict[str, Dict[str, Any]] = {}
        self.meta_models: Dict[str, Any] = {}

        # Training data
        self.X_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None

        self.logger.info(f"✅ MultiOutputStackingModel initialized for {config.n_outputs} outputs")
        # OOF storage
        self._oof_base_predictions: Dict[str, np.ndarray] = {}
        self._oof_meta_predictions: Optional[np.ndarray] = None
        self._cv_splits: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None
        # Regularization helper
        self._overfit_helper = (
            OverfittingPrevention(OverfittingPreventionConfig())
            if _OVERFITTING_AVAILABLE
            else None
        )

    def add_base_model(self, output_name: str, model_name: str, model: Any) -> None:
        """Add a base model for a specific output."""
        if output_name not in self.base_models:
            self.base_models[output_name] = {}

        self.base_models[output_name][model_name] = model
        self.logger.info(f"➕ Added base model {model_name} for output {output_name}")

    def add_meta_model(self, output_name: str, model: Any) -> None:
        """Add a meta model for a specific output."""
        self.meta_models[output_name] = model
        self.logger.info(f"➕ Added meta model for output {output_name}")

    def _create_default_base_models(self, output_name: str) -> None:
        """Create default base models for an output if none exist."""
        if output_name not in self.base_models:
            self.base_models[output_name] = {}

        if len(self.base_models[output_name]) == 0:
            try:
                from sklearn.ensemble import GradientBoostingRegressor, ExtraTreesRegressor
                from sklearn.linear_model import Ridge, Lasso
                from sklearn.svm import SVR
                from sklearn.neighbors import KNeighborsRegressor

                # Create diverse base models for better ensemble performance
                default_models = {
                    'rf': RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=1),
                    'et': ExtraTreesRegressor(n_estimators=50, random_state=42, n_jobs=1),
                    'lr': LinearRegression(),
                    'ridge': Ridge(alpha=1.0, random_state=42),
                    'lasso': Lasso(alpha=0.1, random_state=42, max_iter=1000),
                    'gbr': GradientBoostingRegressor(n_estimators=50, random_state=42),
                    'svr': SVR(kernel='rbf', C=1.0, gamma='scale'),
                    'knn': KNeighborsRegressor(n_neighbors=5, weights='distance')
                }

                # Only add models that can be imported successfully
                successful_models = {}
                for model_name, model in default_models.items():
                    try:
                        # Test if model can be instantiated
                        test_model = model.__class__(**model.get_params() if hasattr(model, 'get_params') else {})
                        successful_models[model_name] = model
                    except Exception as e:
                        self.logger.debug(f"Could not create {model_name} model: {e}")
                        continue

                for model_name, model in successful_models.items():
                    self.base_models[output_name][model_name] = model

                self.logger.info(f"🔧 Created {len(successful_models)} default base models for output {output_name}")

            except ImportError as e:
                self.logger.warning(f"⚠️ Could not create default base models for {output_name}: {e}")
                # Fallback to minimal models
                try:
                    from sklearn.linear_model import LinearRegression
                    from sklearn.ensemble import RandomForestRegressor

                    minimal_models = {
                        'lr': LinearRegression(),
                        'rf': RandomForestRegressor(n_estimators=10, random_state=42)
                    }

                    for model_name, model in minimal_models.items():
                        self.base_models[output_name][model_name] = model

                    self.logger.info(f"🔧 Created {len(minimal_models)} minimal base models for output {output_name}")

                except Exception as fallback_e:
                    self.logger.error(f"❌ Could not create any base models for {output_name}: {fallback_e}")

    def _create_default_meta_model(self, output_name: str) -> None:
        """Create default meta model for an output if none exist."""
        if output_name not in self.meta_models:
            try:
                from sklearn.ensemble import RandomForestRegressor
                from sklearn.linear_model import Ridge
                from sklearn.svm import SVR

                # Try different meta models in order of preference
                meta_model_candidates = [
                    ('ridge', Ridge(alpha=1.0, random_state=42)),
                    ('rf', RandomForestRegressor(n_estimators=20, random_state=42, n_jobs=1)),
                    ('svr', SVR(kernel='rbf', C=1.0, gamma='scale')),
                    ('lr', LinearRegression())
                ]

                for model_name, model in meta_model_candidates:
                    try:
                        # Test if model can be instantiated
                        test_model = model.__class__(**model.get_params() if hasattr(model, 'get_params') else {})
                        self.meta_models[output_name] = model
                        self.logger.info(f"🔧 Created default meta model ({model_name}) for output {output_name}")
                        break
                    except Exception as e:
                        self.logger.debug(f"Could not create {model_name} meta model: {e}")
                        continue

                # If no model was created, create a simple fallback
                if output_name not in self.meta_models:
                    self.meta_models[output_name] = LinearRegression()
                    self.logger.info(f"🔧 Created fallback meta model for output {output_name}")

            except ImportError as e:
                self.logger.warning(f"⚠️ Could not create default meta model for {output_name}: {e}")
                # Last resort fallback
                try:
                    self.meta_models[output_name] = LinearRegression()
                    self.logger.info(f"🔧 Created minimal meta model for output {output_name}")
                except Exception as fallback_e:
                    self.logger.error(f"❌ Could not create any meta model for {output_name}: {fallback_e}")

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'MultiOutputStackingModel':
        """Fit the multi-output stacking model with OUT-OF-FOLD stacking.

        - Generate OOF base predictions using a purged time-series splitter when possible
        - Train the meta-learner on OOF base predictions (optionally with original features passthrough)
        - Fit final base models on full data for inference; meta is fit on OOF features
        """

        # Normalize inputs to pandas DataFrame/ndarray as needed
        X_is_df = isinstance(X, pd.DataFrame)
        y_is_df = isinstance(y, (pd.DataFrame, pd.Series))

        if not X_is_df:
            X_df = pd.DataFrame(X)
        else:
            X_df = X

        if isinstance(y, pd.Series):
            y_df = y.to_frame()
        elif isinstance(y, pd.DataFrame):
            y_df = y
        else:
            y_arr = y
            if len(y_arr.shape) == 1:
                y_arr = y_arr.reshape(-1, 1)
            y_df = pd.DataFrame(y_arr)

        self.logger.info(f"🔄 Fitting MultiOutputStackingModel with {len(X_df)} samples")
        start_time = time.time()

        try:
            # Align outputs
            if y_df.shape[1] != self.config.n_outputs:
                if y_df.shape[1] == 1 and self.config.n_outputs != 1:
                    # Single output provided; adjust config
                    self.config.n_outputs = 1
                    self.config.output_names = ["output_1"]
                    self.output_weights = [1.0]
                    self.output_loss_weights = [1.0]
                elif self.config.n_outputs != y_df.shape[1]:
                    self.logger.warning(
                        f"⚠️ Adjusting output_names to match provided y columns: {y_df.shape[1]}"
                    )
                    self.config.n_outputs = y_df.shape[1]
                    self.config.output_names = [f"output_{i+1}" for i in range(self.config.n_outputs)]

            # Validate outputs
            if not self.validate_outputs(y_df.values):
                raise ValidationError("Invalid output data format")

            # Store training data for later reference
            self.X_train = X_df.values if not X_is_df else X_df
            self.y_train = y_df.values

            # Calculate output correlations (on provided y)
            output_correlations = self.calculate_output_correlations(y_df.values)

            # Build CV splits (purged if possible)
            self._cv_splits = self._build_time_series_splits(X_df, n_splits=self.config.cv_folds)
            n_splits_actual = len(self._cv_splits)
            self.logger.info(f"🔧 Using time-series CV with {n_splits_actual} folds (purged={_PURGED_AVAILABLE and isinstance(X_df.index, pd.DatetimeIndex)})")

            # Generate OOF base predictions per output
            self._oof_base_predictions = {}

            for output_idx, output_name in enumerate(self.config.output_names):
                self.logger.info(f"🔄 Generating OOF base predictions for {output_name}...")

                # Ensure we have base models for this output
                if output_name not in self.base_models or len(self.base_models[output_name]) == 0:
                    self._create_default_base_models(output_name)
                if output_name not in self.base_models or len(self.base_models[output_name]) == 0:
                    self.logger.warning(f"⚠️ No base models configured for output {output_name}")
                    continue

                model_names = list(self.base_models[output_name].keys())
                n_models = len(model_names)
                Z_oof = np.zeros((len(X_df), n_models), dtype=float)

                # Pull target for this output
                if y_df.shape[1] > output_idx:
                    y_output = y_df.iloc[:, output_idx].values
                else:
                    self.logger.error(f"❌ Missing target column for output index {output_idx}")
                    continue

                # Per-fold training for OOF base predictions
                for fold_idx, (tr_idx, va_idx) in enumerate(self._cv_splits):
                    X_tr, X_va = self._safe_index(X_df, tr_idx), self._safe_index(X_df, va_idx)
                    y_tr, y_va = y_output[tr_idx], y_output[va_idx]

                    for m_i, model_name in enumerate(model_names):
                        model = self.base_models[output_name][model_name]
                        model_fold = self._clone_and_regularize(model, model_name)
                        # Fit with early stopping when supported
                        self._fit_with_optional_early_stopping(model_fold, X_tr, y_tr, X_va, y_va, model_name)

                        # Predict on validation fold
                        pred = self._predict_1d(model_fold, X_va)
                        Z_oof[va_idx, m_i] = pred

                self._oof_base_predictions[output_name] = Z_oof
                self.logger.info(f"✅ OOF base predictions ready for {output_name} (shape={Z_oof.shape})")

            # Train meta models on OOF base predictions (+ passthrough original features)
            self.logger.info("🔄 Training meta models on OOF features...")
            oof_meta_preds_list = []

            for output_idx, output_name in enumerate(self.config.output_names):
                if output_name not in self._oof_base_predictions:
                    continue

                # Ensure meta model exists
                if output_name not in self.meta_models:
                    self._create_default_meta_model(output_name)
                if output_name not in self.meta_models:
                    self.logger.warning(f"⚠️ No meta model available for output {output_name}")
                    continue

                Z_oof = self._oof_base_predictions[output_name]
                # Passthrough of original features
                X_passthrough = X_df.values if isinstance(X_df, pd.DataFrame) else np.asarray(X_df)
                meta_X = np.hstack([X_passthrough, Z_oof])
                y_output = y_df.iloc[:, output_idx].values

                # Create OOF meta predictions using same CV splits
                meta_oof = np.zeros(len(X_df), dtype=float)
                for fold_idx, (tr_idx, va_idx) in enumerate(self._cv_splits):
                    meta_model_clone = self._clone_and_regularize(self.meta_models[output_name], type(self.meta_models[output_name]).__name__)
                    X_tr_m, X_va_m = meta_X[tr_idx], meta_X[va_idx]
                    y_tr_m, y_va_m = y_output[tr_idx], y_output[va_idx]
                    # Early stopping if supported
                    self._fit_with_optional_early_stopping(meta_model_clone, X_tr_m, y_tr_m, X_va_m, y_va_m, f"meta_{output_name}")
                    meta_oof[va_idx] = self._predict_1d(meta_model_clone, X_va_m)

                oof_meta_preds_list.append(meta_oof.reshape(-1, 1))

                # Fit final META model on full OOF features (train on all OOF rows)
                final_meta = self.meta_models[output_name]
                final_meta = self._clone_and_regularize(final_meta, type(final_meta).__name__)
                self._fit_with_optional_early_stopping(final_meta, meta_X, y_output, None, None, f"meta_full_{output_name}")
                self.meta_models[output_name] = final_meta
                self.logger.info(f"✅ Meta model trained (full OOF) for {output_name}")

            # Stack OOF meta predictions for OOF evaluation
            if oof_meta_preds_list:
                self._oof_meta_predictions = np.column_stack(oof_meta_preds_list)
            else:
                self._oof_meta_predictions = None

            # Finally, fit base models on FULL data for inference
            for output_idx, output_name in enumerate(self.config.output_names):
                if output_name not in self.base_models or len(self.base_models[output_name]) == 0:
                    continue
                # Simple holdout for early stopping: last 10% as validation
                n = len(X_df)
                val_size = max(1, int(0.1 * n))
                tr_idx = np.arange(0, n - val_size)
                va_idx = np.arange(n - val_size, n)
                X_tr, X_va = self._safe_index(X_df, tr_idx), self._safe_index(X_df, va_idx)
                y_output = y_df.iloc[:, output_idx].values
                y_tr, y_va = y_output[tr_idx], y_output[va_idx]

                for model_name, model in list(self.base_models[output_name].items()):
                    model_full = self._clone_and_regularize(model, model_name)
                    self._fit_with_optional_early_stopping(model_full, X_tr, y_tr, X_va, y_va, model_name)
                    self.base_models[output_name][model_name] = model_full

            # Update state
            self.is_fitted = True

            # Record training history
            training_time = time.time() - start_time
            self.training_history.append({
                'timestamp': datetime.now(),
                'duration': training_time,
                'n_samples': int(len(X_df)),
                'n_features': int(X_df.shape[1]),
                'n_outputs': int(y_df.shape[1]),
                'base_models_per_output': {name: len(models) for name, models in self.base_models.items()},
                'output_correlations': output_correlations.tolist() if output_correlations is not None else None
            })

            self.logger.info(f"✅ MultiOutputStackingModel fitted in {training_time:.3f}s")
            self.logger.info(f"📊 OOF meta predictions available: {self._oof_meta_predictions is not None}")
            return self

        except Exception as e:
            training_time = time.time() - start_time
            self.logger.error(f"❌ Failed to fit MultiOutputStackingModel after {training_time:.3f}s: {e}")
            raise

    # --------------------- Internal utilities ---------------------
    def _build_time_series_splits(self, X_df: pd.DataFrame, n_splits: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Create time-series CV splits; use purged splits when possible."""
        splits: List[Tuple[np.ndarray, np.ndarray]] = []
        try:
            if _PURGED_AVAILABLE and isinstance(X_df.index, pd.DatetimeIndex):
                splitter = PurgedKFoldTime(n_splits=n_splits)
                for tr, va in splitter.split(X_df):
                    splits.append((np.asarray(tr), np.asarray(va)))
            else:
                tscv = TimeSeriesSplit(n_splits=n_splits)
                for tr, va in tscv.split(np.arange(len(X_df))):
                    splits.append((np.asarray(tr), np.asarray(va)))
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to build purged splits, falling back to TimeSeriesSplit: {e}")
            tscv = TimeSeriesSplit(n_splits=n_splits)
            for tr, va in tscv.split(np.arange(len(X_df))):
                splits.append((np.asarray(tr), np.asarray(va)))
        return splits

    def _safe_index(self, X: Union[pd.DataFrame, np.ndarray], idx: np.ndarray) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            return X.iloc[idx].values
        return X[idx]

    def _clone_and_regularize(self, model: Any, model_type: str) -> Any:
        """Clone model and apply regularization via OverfittingPrevention if available."""
        try:
            cloned = skl_clone(model) if hasattr(model, 'get_params') else model.__class__(**getattr(model, 'get_params', lambda: {})())
        except Exception:
            cloned = model
        if self._overfit_helper is not None:
            try:
                return self._overfit_helper.apply_regularization(cloned, model_type)
            except Exception:
                return cloned
        return cloned

    def _fit_with_optional_early_stopping(
        self,
        model: Any,
        X_tr: np.ndarray,
        y_tr: np.ndarray,
        X_va: Optional[np.ndarray],
        y_va: Optional[np.ndarray],
        model_name: str,
    ) -> None:
        """Fit model; if it supports eval_set/early_stopping, pass them."""
        try:
            sig = inspect.signature(model.fit)
            kwargs = {}
            if X_va is not None and y_va is not None:
                # XGBoost / LightGBM pattern: eval_set, early_stopping_rounds
                if 'eval_set' in sig.parameters:
                    kwargs['eval_set'] = [(X_va, y_va)]
                if 'early_stopping_rounds' in sig.parameters:
                    kwargs['early_stopping_rounds'] = 50
                if 'verbose' in sig.parameters:
                    kwargs['verbose'] = False
            # If no eval_set path, enable native sklearn early stopping parameters when available
            try:
                if hasattr(model, 'get_params') and hasattr(model, 'set_params'):
                    params = model.get_params()
                    updates = {}
                    # Generic patience from prevention helper if available
                    patience = 10
                    tol = 1e-4
                    if self._overfit_helper is not None:
                        try:
                            patience = int(getattr(self._overfit_helper.config, 'early_stopping_patience', 10))
                            tol = float(getattr(self._overfit_helper.config, 'early_stopping_min_delta', 1e-4))
                        except Exception as e:
                            self.logger.debug(f"Could not extract early stopping parameters: {e}")
                            patience = 10
                            tol = 1e-4
                    # Models with early_stopping flag (e.g., HistGradientBoosting, MLP, SGD)
                    if 'early_stopping' in params and 'eval_set' not in kwargs:
                        updates['early_stopping'] = True
                    # Patience-like parameter
                    if 'n_iter_no_change' in params:
                        updates['n_iter_no_change'] = patience
                    # Validation fraction for internal split if we have a validation set size
                    if 'validation_fraction' in params:
                        if X_va is not None and X_tr is not None:
                            total = len(X_tr) + len(X_va)
                            if total > 0:
                                val_frac = max(0.05, min(0.2, len(X_va) / total))
                                updates['validation_fraction'] = val_frac
                        else:
                            updates['validation_fraction'] = max(0.05, 0.1)
                    # Tolerance mapping
                    if 'tol' in params:
                        updates['tol'] = tol
                    if updates:
                        model.set_params(**updates)
            except Exception as e:
                self.logger.debug(f"Could not update model parameters: {e}")
                # Continue with original parameters

            model.fit(X_tr, y_tr, **kwargs)  # type: ignore[arg-type]
        except Exception:
            # Fallback simple fit
            model.fit(X_tr, y_tr)

    def _predict_1d(self, model: Any, X: np.ndarray) -> np.ndarray:
        """Predict and return a 1D array, handling predict_proba when applicable."""
        if hasattr(model, 'predict_proba'):
            try:
                proba = model.predict_proba(X)
                if hasattr(proba, 'shape') and len(proba.shape) > 1 and proba.shape[1] > 1:
                    return proba[:, 1].astype(float)
                return np.asarray(proba).astype(float).ravel()
            except Exception as e:
                self.logger.debug(f"predict_proba failed, falling back to predict: {e}")
                # Fall through to predict method
        pred = model.predict(X)
        pred_arr = np.asarray(pred).astype(float)
        return pred_arr.ravel()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions for all outputs."""

        if not self.is_fitted:
            raise ValueError("Model not fitted")

        self.logger.debug(f"🔮 Making predictions for {X.shape[0]} samples")
        start_time = time.time()

        try:
            predictions = []

            for output_idx, output_name in enumerate(self.config.output_names):
                # Ensure base and meta models exist. If missing, create sensible defaults.
                if output_name not in self.base_models or len(self.base_models.get(output_name, {})) == 0:
                    self.logger.debug(f"🔧 Creating default base models for prediction: {output_name}")
                    self._create_default_base_models(output_name)
                if output_name not in self.meta_models:
                    self.logger.debug(f"🔧 Creating default meta model for prediction: {output_name}")
                    self._create_default_meta_model(output_name)
                # If still missing, use zeros as safe fallback (avoid crash)
                if output_name not in self.base_models or output_name not in self.meta_models or len(self.base_models[output_name]) == 0:
                    self.logger.warning(f"⚠️ Missing base/meta models for output {output_name}, returning zeros for this output")
                    predictions.append(np.zeros(X.shape[0]))
                    continue

                # Get base model predictions
                base_predictions = []
                X_arr = X.values if isinstance(X, pd.DataFrame) else X
                for model_name, model in self.base_models[output_name].items():
                    pred = self._predict_1d(model, X_arr)
                    base_predictions.append(pred)

                # Stack base predictions
                base_pred_array = np.column_stack(base_predictions)

                # Combine original features with base model predictions
                meta_features = np.hstack([X, base_pred_array])

                # Get meta model prediction
                meta_model = self.meta_models[output_name]
                try:
                    meta_pred = np.asarray(meta_model.predict(meta_features)).ravel()
                except Exception:
                    # Fallback to averaging base predictions if meta model fails
                    self.logger.warning(f"⚠️ Meta model prediction failed for {output_name}, using mean of base predictions")
                    meta_pred = base_pred_array.mean(axis=1)

                predictions.append(meta_pred)
                self.logger.debug(f"✅ Predictions generated for {output_name}: {len(meta_pred)} samples")

            # Stack all predictions
            final_predictions = np.column_stack(predictions)

            # Apply output weights
            weighted_predictions = self.apply_output_weights(final_predictions)

            # Calculate confidence scores
            confidence_scores = self.calculate_confidence_scores(weighted_predictions)

            # Record prediction history
            prediction_time = time.time() - start_time
            self.prediction_history.append({
                'timestamp': datetime.now(),
                'duration': prediction_time,
                'n_samples': X.shape[0],
                'confidence_mean': float(np.mean(confidence_scores)),
                'confidence_std': float(np.std(confidence_scores))
            })

            self.logger.info(f"✅ Predictions completed in {prediction_time:.3f}s")
            self.logger.info(f"📊 Confidence: {np.mean(confidence_scores):.3f} ± {np.std(confidence_scores):.3f}")

            return weighted_predictions

        except Exception as e:
            prediction_time = time.time() - start_time
            self.logger.error(f"❌ Failed to make predictions after {prediction_time:.3f}s: {e}")
            raise

    def predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        """Make probability predictions for all outputs."""

        if not self.is_fitted:
            raise ValueError("Model not fitted")

        self.logger.debug(f"🔮 Making probability predictions for {X.shape[0]} samples")

        try:
            # Check if this is a classification task
            if not self.is_classification:
                self.logger.info("ℹ️ predict_proba called on regression model - returning None")
                return None

            # Collect probability predictions from all models
            probas = []
            proba_details = {}

            for output_idx, output_name in enumerate(self.config.output_names):
                self.logger.debug(f"🎯 Predicting probabilities for {output_name}")

                # Ensure base and meta models exist
                if output_name not in self.base_models or output_name not in self.meta_models:
                    self.logger.warning(f"⚠️ Missing models for output {output_name}, skipping probability prediction")
                    continue

                # Get base model probability predictions
                base_probas = []
                X_arr = X.values if isinstance(X, pd.DataFrame) else X
                
                for model_name, model in self.base_models[output_name].items():
                    try:
                        if hasattr(model, 'predict_proba'):
                            base_proba = model.predict_proba(X_arr)
                            # For binary classification, use positive class probability
                            if base_proba.ndim > 1 and base_proba.shape[1] > 1:
                                base_proba = base_proba[:, 1]  # Use positive class
                            base_probas.append(base_proba)
                        else:
                            # Convert predictions to pseudo-probabilities
                            pred = model.predict(X_arr)
                            # Simple sigmoid-like transformation
                            pseudo_proba = 1 / (1 + np.exp(-pred))
                            base_probas.append(pseudo_proba)
                    except Exception as e:
                        self.logger.warning(f"Failed to get probabilities from {model_name}: {e}")
                        # Fallback to uniform probabilities
                        base_probas.append(np.full(X_arr.shape[0], 0.5))

                if not base_probas:
                    self.logger.warning(f"No valid base model probabilities for {output_name}")
                    continue

                # Stack base probabilities
                base_proba_array = np.column_stack(base_probas)

                # Combine original features with base model probabilities
                meta_features = np.hstack([X_arr, base_proba_array])

                # Get meta model probability prediction
                meta_model = self.meta_models[output_name]
                try:
                    if hasattr(meta_model, 'predict_proba'):
                        meta_proba = meta_model.predict_proba(meta_features)
                        if meta_proba.ndim > 1 and meta_proba.shape[1] > 1:
                            meta_proba = meta_proba[:, 1]  # Use positive class
                    else:
                        # Convert meta prediction to probability
                        meta_pred = meta_model.predict(meta_features)
                        meta_proba = 1 / (1 + np.exp(-meta_pred))
                    
                    probas.append(meta_proba)
                    proba_details[output_name] = {
                        'shape': meta_proba.shape,
                        'range': (meta_proba.min(), meta_proba.max()) if len(meta_proba) > 0 else None
                    }
                    
                except Exception as e:
                    self.logger.warning(f"Meta model probability prediction failed for {output_name}: {e}")
                    # Fallback to average of base probabilities
                    meta_proba = base_proba_array.mean(axis=1)
                    probas.append(meta_proba)
                    proba_details[output_name] = {
                        'shape': meta_proba.shape,
                        'note': 'Fallback to base model average'
                    }

            if not probas:
                self.logger.warning("⚠️ No valid probability predictions generated")
                return None

            # Stack all probabilities
            final_probas = np.column_stack(probas)

            self.logger.info(f"✅ Probability predictions completed - Shape: {final_probas.shape}")
            self.logger.debug(f"📊 Probability details: {proba_details}")

            return final_probas

        except Exception as e:
            self.logger.error(f"❌ Failed to make probability predictions: {e}")
            return None

    def get_base_model_predictions(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Get predictions from all base models."""

        if not self.is_fitted:
            raise ValueError("Model not fitted")

        base_predictions = {}

        for output_name, models in self.base_models.items():
            output_predictions = {}

            for model_name, model in models.items():
                if hasattr(model, 'predict_proba'):
                    pred = model.predict_proba(X)
                    if pred.ndim > 1 and pred.shape[1] > 1:
                        pred = pred[:, 1]  # Use positive class probability
                else:
                    pred = model.predict(X)

                output_predictions[model_name] = pred

            base_predictions[output_name] = output_predictions

        return base_predictions

    def evaluate_performance(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Evaluate model performance."""

        if not self.is_fitted:
            raise ValueError("Model not fitted")

        self.logger.info(f"📊 Evaluating performance on {X.shape[0]} samples")

        try:
            # If evaluating on training data and OOF meta predictions exist, prefer OOF
            use_oof = False
            if self._oof_meta_predictions is not None:
                try:
                    if X is self.X_train or (
                        isinstance(X, (np.ndarray, pd.DataFrame))
                        and (X.shape[0] == (self.X_train.shape[0] if isinstance(self.X_train, np.ndarray) else len(self.X_train)))
                    ):
                        use_oof = True
                except Exception:
                    use_oof = False

            if use_oof:
                y_pred = self._oof_meta_predictions
            else:
                y_pred = self.predict(X)

            # Ensure y and y_pred are 2D arrays for consistent indexing
            if len(y.shape) == 1:
                self.logger.info(f"📊 Reshaping 1D y array from {y.shape} to ({y.shape[0]}, 1)")
                y = y.reshape(-1, 1)

            if len(y_pred.shape) == 1:
                self.logger.info(f"📊 Reshaping 1D y_pred array from {y_pred.shape} to ({y_pred.shape[0]}, 1)")
                y_pred = y_pred.reshape(-1, 1)

            # Validate that we have the expected number of outputs
            expected_outputs = len(self.config.output_names)
            actual_outputs = y.shape[1] if len(y.shape) > 1 else 1

            if actual_outputs != expected_outputs:
                self.logger.warning(f"⚠️ Output count mismatch: expected {expected_outputs}, got {actual_outputs}")
                # Adjust to handle the actual number of outputs
                num_outputs_to_process = min(expected_outputs, actual_outputs)
            else:
                num_outputs_to_process = expected_outputs

            # Calculate metrics for each output
            per_output_metrics = {}
            overall_metrics = {}

            for output_idx in range(num_outputs_to_process):
                output_name = self.config.output_names[output_idx] if output_idx < len(self.config.output_names) else f"output_{output_idx + 1}"

                # Safe indexing with bounds checking
                if y.shape[1] > output_idx:
                    y_true_output = y[:, output_idx]
                else:
                    self.logger.warning(f"⚠️ No target data for output {output_idx}, using zeros")
                    y_true_output = np.zeros(y.shape[0])

                if y_pred.shape[1] > output_idx:
                    y_pred_output = y_pred[:, output_idx]
                else:
                    self.logger.warning(f"⚠️ No prediction data for output {output_idx}, using zeros")
                    y_pred_output = np.zeros(y_pred.shape[0])

                # Calculate basic metrics
                mse = np.mean((y_true_output - y_pred_output) ** 2)
                mae = np.mean(np.abs(y_true_output - y_pred_output))
                r2 = 1 - (np.sum((y_true_output - y_pred_output) ** 2) /
                         np.sum((y_true_output - np.mean(y_true_output)) ** 2))

                per_output_metrics[output_name] = {
                    'mse': float(mse),
                    'mae': float(mae),
                    'r2': float(r2)
                }

                # Add to overall metrics
                overall_metrics[f'{output_name}_mse'] = float(mse)
                overall_metrics[f'{output_name}_mae'] = float(mae)
                overall_metrics[f'{output_name}_r2'] = float(r2)

            # Calculate overall metrics
            overall_metrics['overall_mse'] = float(np.mean([m['mse'] for m in per_output_metrics.values()]))
            overall_metrics['overall_mae'] = float(np.mean([m['mae'] for m in per_output_metrics.values()]))
            overall_metrics['overall_r2'] = float(np.mean([m['r2'] for m in per_output_metrics.values()]))

            self.logger.info(f"📊 Overall performance - MSE: {overall_metrics['overall_mse']:.4f}, "
                           f"MAE: {overall_metrics['overall_mae']:.4f}, R²: {overall_metrics['overall_r2']:.4f}")

            return {
                'per_output_metrics': per_output_metrics,
                'overall_metrics': overall_metrics,
                'predictions': y_pred,
                'targets': y
            }

        except Exception as e:
            self.logger.error(f"❌ Failed to evaluate performance: {e}")
            return {'error': str(e)}

    def evaluate_oof_performance(self) -> Dict[str, Any]:
        """Evaluate performance using enhanced consolidated OOF utilities."""
        if self._oof_meta_predictions is None or self.y_train is None:
            return {'error': 'OOF predictions not available'}
        
        try:
            # Import enhanced consolidated utilities
            from src.utils.ml_common.validation.enhanced_consolidated_oof_oos import (
                create_enhanced_oos_validator,
                OOSValidationType
            )
            
            # Create OOS validator for performance evaluation
            oos_validator = create_enhanced_oos_validator(
                validation_type=OOSValidationType.PERFORMANCE_METRICS,
                metrics=['mse', 'mae', 'r2', 'accuracy']
            )
            
            # Perform OOS validation
            oos_result = oos_validator.validate_oos(
                predictions=self._oof_meta_predictions,
                targets=self.y_train
            )
            
            # Extract results
            y = self.y_train
            y_pred = self._oof_meta_predictions
            # Ensure 2D
            if len(y.shape) == 1:
                y = y.reshape(-1, 1)
            if len(y_pred.shape) == 1:
                y_pred = y_pred.reshape(-1, 1)

            per_output_metrics = {}
            overall_metrics = {}
            num_outputs_to_process = min(y.shape[1], y_pred.shape[1])
            for output_idx in range(num_outputs_to_process):
                output_name = self.config.output_names[output_idx] if output_idx < len(self.config.output_names) else f"output_{output_idx+1}"
                y_true_output = y[:, output_idx]
                y_pred_output = y_pred[:, output_idx]
                mse = np.mean((y_true_output - y_pred_output) ** 2)
                mae = np.mean(np.abs(y_true_output - y_pred_output))
                r2 = 1 - (np.sum((y_true_output - y_pred_output) ** 2) / np.sum((y_true_output - np.mean(y_true_output)) ** 2))
                per_output_metrics[output_name] = {
                    'mse': float(mse),
                    'mae': float(mae),
                    'r2': float(r2)
                }
                overall_metrics[f'{output_name}_mse'] = float(mse)
                overall_metrics[f'{output_name}_mae'] = float(mae)
                overall_metrics[f'{output_name}_r2'] = float(r2)

            overall_metrics['overall_mse'] = float(np.mean([m['mse'] for m in per_output_metrics.values()]))
            overall_metrics['overall_mae'] = float(np.mean([m['mae'] for m in per_output_metrics.values()]))
            overall_metrics['overall_r2'] = float(np.mean([m['r2'] for m in per_output_metrics.values()]))

            return {
                'per_output_metrics': per_output_metrics,
                'overall_metrics': overall_metrics,
                'predictions': self._oof_meta_predictions,
                'targets': self.y_train,
                'oos_validation': oos_result.validation_scores,
                'oos_metrics': oos_result.validation_metrics
            }
        except Exception as e:
            self.logger.error(f"❌ OOF evaluation failed: {e}")
            return {'error': str(e)}

# Utility functions for multi-output data preparation
def prepare_multi_output_targets(y: np.ndarray, output_names: List[str]) -> np.ndarray:
    """Prepare multi-output targets from single output data."""

    if len(y.shape) == 1:
        # Single output - duplicate for multi-output
        y_multi = np.column_stack([y] * len(output_names))
        logger.info(f"📊 Converted single output to multi-output: {y.shape} -> {y_multi.shape}")
        return y_multi

    elif len(y.shape) == 2 and y.shape[1] == len(output_names):
        # Already multi-output
        logger.info(f"📊 Multi-output data already prepared: {y.shape}")
        return y

    else:
        raise ValueError(f"Invalid target shape: {y.shape}, expected (n_samples,) or (n_samples, {len(output_names)})")

def create_analyst_outputs(signal_strength: np.ndarray, confidence: np.ndarray,
                          risk_score: np.ndarray, regime_label: np.ndarray) -> np.ndarray:
    """Create Analyst multi-output targets."""

    outputs = np.column_stack([signal_strength, confidence, risk_score, regime_label])
    logger.info(f"📊 Created Analyst outputs: {outputs.shape}")
    return outputs

def create_tactician_outputs(entry_timing: np.ndarray, position_size: np.ndarray,
                            stop_loss: np.ndarray, take_profit: np.ndarray) -> np.ndarray:
    """Create Tactician multi-output targets."""

    outputs = np.column_stack([entry_timing, position_size, stop_loss, take_profit])
    logger.info(f"📊 Created Tactician outputs: {outputs.shape}")
    return outputs

def create_multi_output_stacking_model(config: MultiOutputConfig) -> MultiOutputStackingModel:
    """Create a multi-output stacking model."""
    return MultiOutputStackingModel(config)
