"""
Stacking Ensemble Manager for Multi-Output Models

This module provides comprehensive stacking ensemble management for the Analyst (5m) and
Tactician (1m) multi-output stacking ensemble system.

Key Features:
- StackingEnsembleManager for base models + meta model coordination
- Multi-output stacking training logic
- Prediction with confidence calibration
- Performance tracking and evaluation
- M1 hardware optimization integration
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
import logging
import time
from datetime import datetime

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
from src.utils.tprint import tprint_data_format, LogLevel

# Import multi-output models
from ..models.multi_output_models import MultiOutputConfig, MultiOutputStackingModel, MultiOutputResult

logger = logging.getLogger(__name__)

@dataclass
class StackingEnsembleConfig:
    """Configuration for stacking ensemble manager."""
    # Basic configuration
    ensemble_name: str
    output_dir: str

    # Multi-output configuration
    n_outputs: int = 4
    output_names: List[str] = field(default_factory=lambda: ["output_1", "output_2", "output_3", "output_4"])

    # Base model configuration
    base_models: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    meta_models: Dict[str, Any] = field(default_factory=dict)

    # Training configuration
    enable_cross_validation: bool = True
    cv_folds: int = 5
    enable_early_stopping: bool = True
    early_stopping_patience: int = 10

    # Stacking configuration
    stacking_method: str = "blending"  # blending, stacking, voting
    enable_meta_learning: bool = True
    meta_learning_rate: float = 0.01
    meta_learning_iterations: int = 1000

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
class StackingEnsembleResult:
    """Result from stacking ensemble operations."""
    # Basic info
    ensemble_name: str
    n_outputs: int
    output_names: List[str]
    created_at: datetime
    total_duration: float

    # Model information
    base_model_count: int
    meta_model_count: int
    base_model_performance: Dict[str, Dict[str, float]] = field(default_factory=dict)
    meta_model_performance: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Ensemble performance
    ensemble_performance: Dict[str, float] = field(default_factory=dict)
    per_output_performance: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Predictions
    predictions: np.ndarray = field(default_factory=lambda: np.array([]))
    prediction_probabilities: Optional[np.ndarray] = None
    confidence_scores: np.ndarray = field(default_factory=lambda: np.array([]))

    # Model characteristics
    model_weights: Optional[np.ndarray] = None
    output_correlations: Optional[np.ndarray] = None
    feature_importance: Optional[Dict[str, Any]] = None

    # Metadata
    config: StackingEnsembleConfig = field(default_factory=StackingEnsembleConfig)
    execution_time: float = 0.0
    memory_usage_mb: float = 0.0
    optimization_used: List[str] = field(default_factory=list)

class StackingEnsembleManager:
    """Comprehensive stacking ensemble manager with M1 optimizations."""

    def __init__(self, config: StackingEnsembleConfig):
        """Initialize the stacking ensemble manager."""
        self.logger = logger.getChild('StackingEnsembleManager')
        self.logger.info(f"🚀 Initializing StackingEnsembleManager for {config.ensemble_name}...")
        start_time = time.time()

        self.config = config

        # Initialize M1 optimizers
        self.logger.debug("🔧 Initializing M1 optimizers...")
        self.m1_gpu = get_m1_memory_optimizer() if config.enable_gpu_acceleration else None
        self.m1_memory = get_m1_memory_optimizer(
            memory_limit_gb=config.memory_limit_gb
        ) if config.enable_memory_optimization else None
        self.m1_cpu = get_memory_manager() if config.enable_parallel_processing else None

        self.logger.debug("✅ M1 optimizers initialized")

        # Initialize multi-output stacking model
        self.logger.debug("🔧 Initializing multi-output stacking model...")
        multi_output_config = MultiOutputConfig(
            model_name=config.ensemble_name,
            n_outputs=config.n_outputs,
            output_names=config.output_names,
            base_models=config.base_models,
            meta_model=config.meta_models,
            enable_cross_validation=config.enable_cross_validation,
            cv_folds=config.cv_folds,
            enable_early_stopping=config.enable_early_stopping,
            early_stopping_patience=config.early_stopping_patience,
            output_weights=config.output_weights,
            output_loss_weights=config.output_loss_weights,
            enable_output_correlation=config.enable_output_correlation,
            correlation_threshold=config.correlation_threshold,
            enable_gpu_acceleration=config.enable_gpu_acceleration,
            enable_memory_optimization=config.enable_memory_optimization,
            enable_parallel_processing=config.enable_parallel_processing,
            memory_limit_gb=config.memory_limit_gb,
            max_workers=config.max_workers,
            enable_caching=config.enable_caching,
            cache_size_mb=config.cache_size_mb,
            enable_profiling=config.enable_profiling,
            validation_split=config.validation_split,
            test_split=config.test_split,
            enable_online_learning=config.enable_online_learning,
            save_models=config.save_models,
            save_predictions=config.save_predictions,
            generate_reports=config.generate_reports
        )

        self.stacking_model = MultiOutputStackingModel(multi_output_config)
        self.logger.debug("✅ Multi-output stacking model initialized")

        # Performance tracking
        self.performance_history: List[Dict[str, Any]] = []
        self.prediction_history: List[Dict[str, Any]] = []

        # Ensure output directory exists
        self.logger.debug(f"🔧 Ensuring output directory exists: {config.output_dir}")
        ensure_directory(config.output_dir)
        self.logger.debug("✅ Output directory ready")

        init_time = time.time() - start_time
        self.logger.info(f"✅ StackingEnsembleManager initialized for {config.ensemble_name} in {init_time:.3f}s")
        self.logger.info(f"⚡ GPU acceleration: {config.enable_gpu_acceleration}")
        self.logger.info(f"🧠 Memory optimization: {config.enable_memory_optimization}")
        self.logger.info(f"🔄 Parallel processing: {config.enable_parallel_processing}")
        self.logger.info(f"📊 Outputs: {config.n_outputs} ({config.output_names})")
        self.logger.info(f"🎯 Stacking method: {config.stacking_method}")
        self.logger.info(f"💾 Output directory: {config.output_dir}")

    def add_base_model(self, output_name: str, model_name: str, model: Any) -> None:
        """Add a base model for a specific output."""
        self.stacking_model.add_base_model(output_name, model_name, model)
        self.logger.info(f"➕ Added base model {model_name} for output {output_name}")

    def add_meta_model(self, output_name: str, model: Any) -> None:
        """Add a meta model for a specific output."""
        self.stacking_model.add_meta_model(output_name, model)
        self.logger.info(f"➕ Added meta model for output {output_name}")

    @traced(span_name='train_ensemble')
    def train_ensemble(self, X_train: pd.DataFrame, y_train: pd.DataFrame,
                      X_val: Optional[pd.DataFrame] = None,
                      y_val: Optional[pd.DataFrame] = None) -> StackingEnsembleResult:
        """Train the stacking ensemble."""

        self.logger.info("🚀 Training stacking ensemble...")
        start_time = time.time()

        self.logger.info(f"📊 Training data shape: {X_train.shape}")
        self.logger.info(f"📊 Target data shape: {y_train.shape}")
        if X_val is not None:
            self.logger.info(f"📊 Validation data shape: {X_val.shape}")
        if y_val is not None:
            self.logger.info(f"📊 Validation target shape: {y_val.shape}")

        # Convert to numpy arrays
        X_train_array = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
        y_train_array = y_train.values if isinstance(y_train, pd.DataFrame) else y_train

        if X_val is not None:
            X_val_array = X_val.values if isinstance(X_val, pd.DataFrame) else X_val
        else:
            X_val_array = None

        if y_val is not None:
            y_val_array = y_val.values if isinstance(y_val, pd.DataFrame) else y_val
        else:
            y_val_array = None

        # Memory optimization - direct training without context manager
        if self.m1_memory:
            self.logger.debug("🧠 Using memory optimization...")
        else:
            self.logger.debug("🧠 No memory optimization available, proceeding normally...")

        result = self._train_ensemble_internal(X_train_array, y_train_array, X_val_array, y_val_array)

        execution_time = time.time() - start_time
        result.execution_time = execution_time

        # Log memory usage
        if self.m1_memory:
            result.memory_usage_mb = self.m1_memory.get_current_memory_usage_mb()
            self.logger.info(f"🧠 Memory usage: {result.memory_usage_mb:.1f} MB")

        self.logger.info(f"✅ Stacking ensemble trained in {execution_time:.2f}s")
        self.logger.info(f"📊 Ensemble performance: {result.ensemble_performance}")
        self.logger.info(f"🎯 Base models: {result.base_model_count}")
        self.logger.info(f"🎯 Meta models: {result.meta_model_count}")

        return result

    def _train_ensemble_internal(self, X_train: np.ndarray, y_train: np.ndarray,
                                X_val: Optional[np.ndarray], y_val: Optional[np.ndarray]) -> StackingEnsembleResult:
        """Internal ensemble training logic."""

        self.logger.debug("🔄 Starting internal ensemble training...")
        internal_start_time = time.time()

        # Train the stacking model
        self.logger.debug("🔄 Training multi-output stacking model...")
        training_start_time = time.time()

        self.stacking_model.fit(X_train, y_train)

        training_time = time.time() - training_start_time
        self.logger.info(f"✅ Stacking model training completed in {training_time:.3f}s")

        # Evaluate performance
        self.logger.debug("🔍 Evaluating ensemble performance...")
        eval_start_time = time.time()

        if X_val is not None and y_val is not None:
            # Proper out-of-sample evaluation using provided holdout
            evaluation_results = self.stacking_model.evaluate_performance(X_val, y_val)
        else:
            # Fall back to OOF-based evaluation rather than in-sample training data
            evaluation_results = self.stacking_model.evaluate_oof_performance()

        eval_time = time.time() - eval_start_time
        self.logger.info(f"✅ Ensemble evaluation completed in {eval_time:.3f}s")

        # Calculate ensemble characteristics
        self.logger.debug("📊 Calculating ensemble characteristics...")
        char_start_time = time.time()

        # Get base model predictions for analysis
        base_predictions = self.stacking_model.get_base_model_predictions(X_train)

        # Calculate output correlations
        output_correlations = self.stacking_model.calculate_output_correlations(y_train)

        char_time = time.time() - char_start_time
        self.logger.info(f"✅ Characteristic calculation completed in {char_time:.3f}s")

        # Create results
        self.logger.debug("📊 Creating ensemble results...")
        result = StackingEnsembleResult(
            ensemble_name=self.config.ensemble_name,
            n_outputs=self.config.n_outputs,
            output_names=self.config.output_names,
            created_at=datetime.now(),
            total_duration=0.0,  # Will be set by caller
            base_model_count=sum(len(models) for models in self.stacking_model.base_models.values()),
            meta_model_count=len(self.stacking_model.meta_models),
            base_model_performance={},  # Will be populated by evaluation
            meta_model_performance={},  # Will be populated by evaluation
            ensemble_performance=evaluation_results.get('overall_metrics', {}),
            per_output_performance=evaluation_results.get('per_output_metrics', {}),
            predictions=evaluation_results.get('predictions', np.array([])),
            model_weights=self.stacking_model.output_weights,
            output_correlations=output_correlations,
            config=self.config,
            optimization_used=self._get_optimization_used()
        )

        internal_time = time.time() - internal_start_time
        self.logger.info(f"✅ Internal ensemble training completed in {internal_time:.3f}s")

        return result

    @traced(span_name='predict')
    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
        """Make predictions using the stacking ensemble."""

        if not self.stacking_model.is_fitted:
            raise ValueError("Ensemble not trained yet")

        self.logger.debug(f"🔮 Making predictions for {X.shape[0]} samples")
        start_time = time.time()

        try:
            # Convert to numpy array
            X_array = X.values if isinstance(X, pd.DataFrame) else X

            # Make predictions
            predictions = self.stacking_model.predict(X_array)
            probabilities = self.stacking_model.predict_proba(X_array)
            confidence_scores = self.stacking_model.calculate_confidence_scores(predictions)

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

            return predictions, probabilities, confidence_scores

        except Exception as e:
            prediction_time = time.time() - start_time
            self.logger.error(f"❌ Failed to make predictions after {prediction_time:.3f}s: {e}")
            raise

    def evaluate_performance(self, X: pd.DataFrame, y: pd.DataFrame) -> Dict[str, Any]:
        """Evaluate ensemble performance."""

        if not self.stacking_model.is_fitted:
            raise ValueError("Ensemble not trained yet")

        self.logger.info(f"📊 Evaluating performance on {X.shape[0]} samples")

        try:
            # Convert to numpy arrays
            X_array = X.values if isinstance(X, pd.DataFrame) else X
            y_array = y.values if isinstance(y, pd.DataFrame) else y

            # Evaluate performance
            evaluation_results = self.stacking_model.evaluate_performance(X_array, y_array)

            # Log performance metrics
            overall_metrics = evaluation_results.get('overall_metrics', {})
            self.logger.info(f"📊 Overall performance - MSE: {overall_metrics.get('overall_mse', 0):.4f}, "
                           f"MAE: {overall_metrics.get('overall_mae', 0):.4f}, R²: {overall_metrics.get('overall_r2', 0):.4f}")

            return evaluation_results

        except Exception as e:
            self.logger.error(f"❌ Failed to evaluate performance: {e}")
            return {'error': str(e)}

    def get_base_model_predictions(self, X: pd.DataFrame) -> Dict[str, Dict[str, np.ndarray]]:
        """Get predictions from all base models."""

        if not self.stacking_model.is_fitted:
            raise ValueError("Ensemble not trained yet")

        try:
            # Convert to numpy array
            X_array = X.values if isinstance(X, pd.DataFrame) else X

            # Get base model predictions
            base_predictions = self.stacking_model.get_base_model_predictions(X_array)

            return base_predictions

        except Exception as e:
            self.logger.error(f"❌ Failed to get base model predictions: {e}")
            return {}

    def save_ensemble(self, file_path: str) -> None:
        """Save the ensemble to disk."""

        try:
            # Save the stacking model
            model_path = file_path.replace('.pkl', '_model.pkl')
            self.stacking_model.save_model(model_path)

            # Save ensemble metadata
            ensemble_data = {
                'config': self.config,
                'performance_history': self.performance_history,
                'prediction_history': self.prediction_history,
                'optimization_used': self._get_optimization_used()
            }

            with open(file_path, 'wb') as f:
                import pickle
                pickle.dump(ensemble_data, f)

            self.logger.info(f"💾 Ensemble saved to {file_path}")

        except Exception as e:
            self.logger.error(f"❌ Failed to save ensemble: {e}")
            raise

    def load_ensemble(self, file_path: str) -> None:
        """Load the ensemble from disk."""

        try:
            # Load ensemble metadata
            with open(file_path, 'rb') as f:
                ensemble_data = pickle.load(f)

            self.config = ensemble_data['config']
            self.performance_history = ensemble_data.get('performance_history', [])
            self.prediction_history = ensemble_data.get('prediction_history', [])

            # Load the stacking model
            model_path = file_path.replace('.pkl', '_model.pkl')
            self.stacking_model.load_model(model_path)

            self.logger.info(f"📂 Ensemble loaded from {file_path}")

        except Exception as e:
            self.logger.error(f"❌ Failed to load ensemble: {e}")
            raise

    def _get_optimization_used(self) -> List[str]:
        """Get list of optimizations used."""
        self.logger.debug("🔍 Getting list of optimizations used...")

        optimizations = []

        if self.config.enable_gpu_acceleration and self.m1_gpu:
            optimizations.append("m1_gpu_acceleration")
            self.logger.debug("✅ M1 GPU acceleration enabled")

        if self.config.enable_memory_optimization and self.m1_memory:
            optimizations.append("m1_memory_optimization")
            self.logger.debug("✅ M1 memory optimization enabled")

        if self.config.enable_parallel_processing and self.m1_cpu:
            optimizations.append("m1_parallel_processing")
            self.logger.debug("✅ M1 parallel processing enabled")

        self.logger.debug(f"📊 Optimizations used: {optimizations}")
        return optimizations

    def get_ensemble_info(self) -> Dict[str, Any]:
        """Get comprehensive ensemble information."""

        return {
            'ensemble_name': self.config.ensemble_name,
            'n_outputs': self.config.n_outputs,
            'output_names': self.config.output_names,
            'base_model_count': sum(len(models) for models in self.stacking_model.base_models.values()),
            'meta_model_count': len(self.stacking_model.meta_models),
            'is_fitted': self.stacking_model.is_fitted,
            'stacking_method': self.config.stacking_method,
            'optimization_used': self._get_optimization_used(),
            'performance_history_count': len(self.performance_history),
            'prediction_history_count': len(self.prediction_history)
        }

# Convenience functions for creating specific ensemble types
def create_analyst_ensemble(base_models: Dict[str, Any], meta_models: Dict[str, Any],
                           output_dir: str = "./analyst_ensemble") -> StackingEnsembleManager:
    """Create an Analyst (5m) stacking ensemble."""

    config = StackingEnsembleConfig(
        ensemble_name="analyst_ensemble",
        output_dir=output_dir,
        n_outputs=4,
        output_names=["signal_strength", "confidence", "risk_score", "regime_label"],
        base_models=base_models,
        meta_models=meta_models,
        stacking_method="blending",
        enable_meta_learning=True
    )

    return StackingEnsembleManager(config)

def create_tactician_ensemble(base_models: Dict[str, Any], meta_models: Dict[str, Any],
                             output_dir: str = "./tactician_ensemble") -> StackingEnsembleManager:
    """Create a Tactician (1m) stacking ensemble."""

    config = StackingEnsembleConfig(
        ensemble_name="tactician_ensemble",
        output_dir=output_dir,
        n_outputs=4,
        output_names=["entry_timing", "position_size", "stop_loss", "take_profit"],
        base_models=base_models,
        meta_models=meta_models,
        stacking_method="blending",
        enable_meta_learning=True
    )

    return StackingEnsembleManager(config)
