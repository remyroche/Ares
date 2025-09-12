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
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import logging
import time
from datetime import datetime

# M1 Optimization imports
from ..hardware.m1_memory_optimizer import get_m1_memory_optimizer, M1MemoryOptimizer
from ..hardware.memory_optimization import get_memory_manager, MemoryMonitor

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

logger = logging.getLogger(__name__)


@dataclass
class MultiOutputConfig:
    """Configuration for multi-output models."""
    # Basic configuration
    model_name: str
    n_outputs: int = 4
    output_names: List[str] = field(default_factory=lambda: ["output_1", "output_2", "output_3", "output_4"])
    
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


class MultiOutputModel(ABC):
    """Abstract base class for multi-output models."""
    
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
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'MultiOutputModel':
        """Fit the multi-output model."""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions for all outputs."""
        pass
    
    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        """Make probability predictions for all outputs."""
        pass
    
    def validate_outputs(self, y: np.ndarray) -> bool:
        """Validate output data format."""
        if len(y.shape) != 2:
            self.logger.error(f"❌ Output data must be 2D, got shape: {y.shape}")
            return False
        
        if y.shape[1] != self.config.n_outputs:
            self.logger.error(f"❌ Expected {self.config.n_outputs} outputs, got {y.shape[1]}")
            return False
        
        self.logger.debug(f"✅ Output validation passed: {y.shape}")
        return True
    
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
        # This is a placeholder implementation
        # In practice, you would extract feature importance from the underlying models
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
            import pickle
            
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
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'MultiOutputStackingModel':
        """Fit the multi-output stacking model."""
        
        self.logger.info(f"🔄 Fitting MultiOutputStackingModel with {X.shape[0]} samples")
        start_time = time.time()
        
        try:
            # Validate inputs
            if not self.validate_outputs(y):
                raise ValidationError("Invalid output data format")
            
            # Store training data
            self.X_train = X
            self.y_train = y
            
            # Calculate output correlations
            output_correlations = self.calculate_output_correlations(y)
            
            # Train base models for each output
            self.logger.info("🔄 Training base models...")
            base_predictions = {}
            
            for output_idx, output_name in enumerate(self.config.output_names):
                self.logger.debug(f"🔄 Training base models for output {output_name}...")
                
                if output_name not in self.base_models:
                    self.logger.warning(f"⚠️ No base models for output {output_name}")
                    continue
                
                # Get target for this output
                y_output = y[:, output_idx]
                
                # Train base models
                output_predictions = []
                for model_name, model in self.base_models[output_name].items():
                    self.logger.debug(f"🔄 Training {model_name} for {output_name}...")
                    
                    # Train model
                    model.fit(X, y_output)
                    
                    # Make predictions for meta-training
                    if hasattr(model, 'predict_proba'):
                        pred = model.predict_proba(X)
                        if pred.ndim > 1 and pred.shape[1] > 1:
                            pred = pred[:, 1]  # Use positive class probability
                    else:
                        pred = model.predict(X)
                    
                    output_predictions.append(pred)
                    self.logger.debug(f"✅ {model_name} trained for {output_name}")
                
                # Store base predictions
                base_predictions[output_name] = np.column_stack(output_predictions)
                self.logger.info(f"✅ Base models trained for {output_name}: {len(output_predictions)} models")
            
            # Train meta models
            self.logger.info("🔄 Training meta models...")
            for output_idx, output_name in enumerate(self.config.output_names):
                if output_name not in self.meta_models:
                    self.logger.warning(f"⚠️ No meta model for output {output_name}")
                    continue
                
                if output_name not in base_predictions:
                    self.logger.warning(f"⚠️ No base predictions for output {output_name}")
                    continue
                
                # Get target for this output
                y_output = y[:, output_idx]
                
                # Train meta model
                meta_model = self.meta_models[output_name]
                meta_model.fit(base_predictions[output_name], y_output)
                
                self.logger.info(f"✅ Meta model trained for {output_name}")
            
            # Update state
            self.is_fitted = True
            
            # Record training history
            training_time = time.time() - start_time
            self.training_history.append({
                'timestamp': datetime.now(),
                'duration': training_time,
                'n_samples': X.shape[0],
                'n_features': X.shape[1],
                'n_outputs': y.shape[1],
                'base_models_per_output': {name: len(models) for name, models in self.base_models.items()},
                'output_correlations': output_correlations.tolist() if output_correlations is not None else None
            })
            
            self.logger.info(f"✅ MultiOutputStackingModel fitted in {training_time:.3f}s")
            self.logger.info(f"📊 Trained {sum(len(models) for models in self.base_models.values())} base models")
            self.logger.info(f"📊 Trained {len(self.meta_models)} meta models")
            
            return self
            
        except Exception as e:
            training_time = time.time() - start_time
            self.logger.error(f"❌ Failed to fit MultiOutputStackingModel after {training_time:.3f}s: {e}")
            raise
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions for all outputs."""
        
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        
        self.logger.debug(f"🔮 Making predictions for {X.shape[0]} samples")
        start_time = time.time()
        
        try:
            predictions = []
            
            for output_idx, output_name in enumerate(self.config.output_names):
                if output_name not in self.base_models or output_name not in self.meta_models:
                    self.logger.warning(f"⚠️ Missing models for output {output_name}, using zeros")
                    predictions.append(np.zeros(X.shape[0]))
                    continue
                
                # Get base model predictions
                base_predictions = []
                for model_name, model in self.base_models[output_name].items():
                    if hasattr(model, 'predict_proba'):
                        pred = model.predict_proba(X)
                        if pred.ndim > 1 and pred.shape[1] > 1:
                            pred = pred[:, 1]  # Use positive class probability
                    else:
                        pred = model.predict(X)
                    
                    base_predictions.append(pred)
                
                # Stack base predictions
                base_pred_array = np.column_stack(base_predictions)
                
                # Get meta model prediction
                meta_model = self.meta_models[output_name]
                meta_pred = meta_model.predict(base_pred_array)
                
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
            # For now, return None as most base models don't support predict_proba
            # In practice, you would implement probability prediction logic
            self.logger.warning("⚠️ Probability predictions not implemented for stacking model")
            return None
            
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
            # Make predictions
            y_pred = self.predict(X)
            
            # Calculate metrics for each output
            per_output_metrics = {}
            overall_metrics = {}
            
            for output_idx, output_name in enumerate(self.config.output_names):
                y_true_output = y[:, output_idx]
                y_pred_output = y_pred[:, output_idx]
                
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