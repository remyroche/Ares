"""
Simplified Stacking Model Training Integration

This module provides simplified integration for the multi-output stacking ensemble
training system, making it easy to use in existing training pipelines.

Key Features:
- StackingModelTrainer integration class
- Configuration mapping from general to stacking
- Simplified training interface
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

# Import stacking ensemble training
from ..stacking_ensemble_training import (
    AnalystStackingTrainer, TacticianStackingTrainer,
    StackingTrainingResult, StackingEnsembleConfig
)

logger = logging.getLogger(__name__)


@dataclass
class SimplifiedStackingConfig:
    """Simplified configuration for stacking model training."""
    # Basic configuration
    model_type: str  # "analyst" or "tactician"
    output_dir: str = "./stacking_models"
    
    # Training configuration
    enable_cross_validation: bool = True
    cv_folds: int = 5
    enable_early_stopping: bool = True
    early_stopping_patience: int = 10
    
    # Stacking configuration
    stacking_method: str = "blending"
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


class StackingModelTrainer:
    """Simplified stacking model trainer for easy integration."""
    
    def __init__(self, config: SimplifiedStackingConfig):
        """Initialize the simplified stacking model trainer."""
        self.logger = logger.getChild('StackingModelTrainer')
        self.logger.info(f"🚀 Initializing StackingModelTrainer for {config.model_type}...")
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
        
        # Initialize the appropriate trainer
        self.logger.debug(f"🔧 Initializing {config.model_type} trainer...")
        if config.model_type.lower() == "analyst":
            self.trainer = AnalystStackingTrainer(self._convert_to_trainer_config())
        elif config.model_type.lower() == "tactician":
            self.trainer = TacticianStackingTrainer(self._convert_to_trainer_config())
        else:
            raise ValueError(f"Unsupported model type: {config.model_type}")
        
        self.logger.debug(f"✅ {config.model_type} trainer initialized")
        
        # Performance tracking
        self.training_history: List[Dict[str, Any]] = []
        self.prediction_history: List[Dict[str, Any]] = []
        
        init_time = time.time() - start_time
        self.logger.info(f"✅ StackingModelTrainer initialized in {init_time:.3f}s")
        self.logger.info(f"🎯 Model type: {config.model_type}")
        self.logger.info(f"📊 Output directory: {config.output_dir}")
        self.logger.info(f"⚡ GPU acceleration: {config.enable_gpu_acceleration}")
        self.logger.info(f"🧠 Memory optimization: {config.enable_memory_optimization}")
        self.logger.info(f"🔄 Parallel processing: {config.enable_parallel_processing}")
    
    def _convert_to_trainer_config(self) -> Dict[str, Any]:
        """Convert simplified config to trainer config."""
        
        return {
            'output_dir': self.config.output_dir,
            'enable_cross_validation': self.config.enable_cross_validation,
            'cv_folds': self.config.cv_folds,
            'enable_early_stopping': self.config.enable_early_stopping,
            'early_stopping_patience': self.config.early_stopping_patience,
            'stacking_method': self.config.stacking_method,
            'enable_meta_learning': self.config.enable_meta_learning,
            'meta_learning_rate': self.config.meta_learning_rate,
            'meta_learning_iterations': self.config.meta_learning_iterations,
            'output_weights': self.config.output_weights,
            'output_loss_weights': self.config.output_loss_weights,
            'enable_output_correlation': self.config.enable_output_correlation,
            'correlation_threshold': self.config.correlation_threshold,
            'enable_gpu_acceleration': self.config.enable_gpu_acceleration,
            'enable_memory_optimization': self.config.enable_memory_optimization,
            'enable_parallel_processing': self.config.enable_parallel_processing,
            'memory_limit_gb': self.config.memory_limit_gb,
            'max_workers': self.config.max_workers,
            'enable_caching': self.config.enable_caching,
            'cache_size_mb': self.config.cache_size_mb,
            'enable_profiling': self.config.enable_profiling,
            'validation_split': self.config.validation_split,
            'test_split': self.config.test_split,
            'enable_online_learning': self.config.enable_online_learning,
            'save_models': self.config.save_models,
            'save_predictions': self.config.save_predictions,
            'generate_reports': self.config.generate_reports
        }
    
    @traced(span_name='train_model')
    def train_model(self, X_train: pd.DataFrame, y_train: pd.DataFrame,
                   X_val: Optional[pd.DataFrame] = None,
                   y_val: Optional[pd.DataFrame] = None) -> StackingTrainingResult:
        """Train the stacking model."""
        
        self.logger.info(f"🚀 Training {self.config.model_type} stacking model...")
        start_time = time.time()
        
        self.logger.info(f"📊 Training data shape: {X_train.shape}")
        self.logger.info(f"📊 Target data shape: {y_train.shape}")
        if X_val is not None:
            self.logger.info(f"📊 Validation data shape: {X_val.shape}")
        if y_val is not None:
            self.logger.info(f"📊 Validation target shape: {y_val.shape}")
        
        try:
            # Train the model
            result = self.trainer.train_ensemble(X_train, y_train, X_val, y_val)
            
            # Record training history
            training_time = time.time() - start_time
            self.training_history.append({
                'timestamp': datetime.now(),
                'duration': training_time,
                'model_type': self.config.model_type,
                'n_samples': X_train.shape[0],
                'n_features': X_train.shape[1],
                'n_outputs': y_train.shape[1],
                'ensemble_performance': result.ensemble_performance,
                'base_model_count': result.base_model_count,
                'meta_model_count': result.meta_model_count
            })
            
            self.logger.info(f"✅ {self.config.model_type} stacking model trained in {training_time:.2f}s")
            self.logger.info(f"📊 Ensemble performance: {result.ensemble_performance}")
            self.logger.info(f"🎯 Base models: {result.base_model_count}")
            self.logger.info(f"🎯 Meta models: {result.meta_model_count}")
            
            return result
            
        except Exception as e:
            training_time = time.time() - start_time
            self.logger.error(f"❌ Failed to train {self.config.model_type} model after {training_time:.3f}s: {e}")
            raise
    
    @traced(span_name='predict')
    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
        """Make predictions using the trained model."""
        
        if not hasattr(self.trainer, 'ensemble_manager') or not self.trainer.ensemble_manager.stacking_model.is_fitted:
            raise ValueError("Model not trained yet")
        
        self.logger.debug(f"🔮 Making predictions for {X.shape[0]} samples")
        start_time = time.time()
        
        try:
            # Make predictions
            predictions, probabilities, confidence_scores = self.trainer.predict(X)
            
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
        """Evaluate model performance."""
        
        if not hasattr(self.trainer, 'ensemble_manager') or not self.trainer.ensemble_manager.stacking_model.is_fitted:
            raise ValueError("Model not trained yet")
        
        self.logger.info(f"📊 Evaluating performance on {X.shape[0]} samples")
        
        try:
            # Evaluate performance
            evaluation_results = self.trainer.ensemble_manager.evaluate_performance(X, y)
            
            # Log performance metrics
            overall_metrics = evaluation_results.get('overall_metrics', {})
            self.logger.info(f"📊 Overall performance - MSE: {overall_metrics.get('overall_mse', 0):.4f}, "
                           f"MAE: {overall_metrics.get('overall_mae', 0):.4f}, R²: {overall_metrics.get('overall_r2', 0):.4f}")
            
            return evaluation_results
            
        except Exception as e:
            self.logger.error(f"❌ Failed to evaluate performance: {e}")
            return {'error': str(e)}
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information."""
        
        if not hasattr(self.trainer, 'ensemble_manager'):
            return {'error': 'Model not initialized'}
        
        return {
            'model_type': self.config.model_type,
            'output_dir': self.config.output_dir,
            'is_fitted': self.trainer.ensemble_manager.stacking_model.is_fitted,
            'n_outputs': self.trainer.ensemble_manager.config.n_outputs,
            'output_names': self.trainer.ensemble_manager.config.output_names,
            'base_model_count': sum(len(models) for models in self.trainer.ensemble_manager.stacking_model.base_models.values()),
            'meta_model_count': len(self.trainer.ensemble_manager.stacking_model.meta_models),
            'stacking_method': self.trainer.ensemble_manager.config.stacking_method,
            'training_history_count': len(self.training_history),
            'prediction_history_count': len(self.prediction_history)
        }
    
    def save_model(self, file_path: str) -> None:
        """Save the trained model."""
        
        try:
            # Save the trainer
            self.trainer.save_ensemble(file_path)
            
            # Save additional metadata
            metadata = {
                'config': self.config,
                'training_history': self.training_history,
                'prediction_history': self.prediction_history
            }
            
            metadata_path = file_path.replace('.pkl', '_metadata.pkl')
            with open(metadata_path, 'wb') as f:
                import pickle
                pickle.dump(metadata, f)
            
            self.logger.info(f"💾 Model saved to {file_path}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save model: {e}")
            raise
    
    def load_model(self, file_path: str) -> None:
        """Load the trained model."""
        
        try:
            # Load the trainer
            self.trainer.load_ensemble(file_path)
            
            # Load additional metadata
            metadata_path = file_path.replace('.pkl', '_metadata.pkl')
            if safe_file_exists(metadata_path):
                with open(metadata_path, 'rb') as f:
                    import pickle
                    metadata = pickle.load(f)
                
                self.training_history = metadata.get('training_history', [])
                self.prediction_history = metadata.get('prediction_history', [])
            
            self.logger.info(f"📂 Model loaded from {file_path}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load model: {e}")
            raise
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get a summary of training history."""
        
        if not self.training_history:
            return {'error': 'No training history available'}
        
        try:
            # Calculate summary statistics
            durations = [entry['duration'] for entry in self.training_history]
            performances = [entry['ensemble_performance'] for entry in self.training_history]
            
            # Extract performance metrics
            mse_scores = [perf.get('overall_mse', 0) for perf in performances if 'overall_mse' in perf]
            mae_scores = [perf.get('overall_mae', 0) for perf in performances if 'overall_mae' in perf]
            r2_scores = [perf.get('overall_r2', 0) for perf in performances if 'overall_r2' in perf]
            
            return {
                'total_training_sessions': len(self.training_history),
                'total_training_time': sum(durations),
                'average_training_time': np.mean(durations),
                'min_training_time': np.min(durations),
                'max_training_time': np.max(durations),
                'latest_performance': performances[-1] if performances else {},
                'performance_trend': {
                    'mse_scores': mse_scores,
                    'mae_scores': mae_scores,
                    'r2_scores': r2_scores
                },
                'model_type': self.config.model_type,
                'last_training': self.training_history[-1]['timestamp'] if self.training_history else None
            }
            
        except Exception as e:
            self.logger.error(f"❌ Failed to generate training summary: {e}")
            return {'error': str(e)}
    
    def get_prediction_summary(self) -> Dict[str, Any]:
        """Get a summary of prediction history."""
        
        if not self.prediction_history:
            return {'error': 'No prediction history available'}
        
        try:
            # Calculate summary statistics
            durations = [entry['duration'] for entry in self.prediction_history]
            confidence_means = [entry['confidence_mean'] for entry in self.prediction_history]
            confidence_stds = [entry['confidence_std'] for entry in self.prediction_history]
            
            return {
                'total_predictions': len(self.prediction_history),
                'total_prediction_time': sum(durations),
                'average_prediction_time': np.mean(durations),
                'min_prediction_time': np.min(durations),
                'max_prediction_time': np.max(durations),
                'average_confidence': np.mean(confidence_means),
                'confidence_std': np.mean(confidence_stds),
                'model_type': self.config.model_type,
                'last_prediction': self.prediction_history[-1]['timestamp'] if self.prediction_history else None
            }
            
        except Exception as e:
            self.logger.error(f"❌ Failed to generate prediction summary: {e}")
            return {'error': str(e)}


# Convenience functions for creating trainers
def create_analyst_stacking_trainer(output_dir: str = "./analyst_models",
                                   config: Optional[Dict[str, Any]] = None) -> StackingModelTrainer:
    """Create an Analyst stacking trainer."""
    
    simplified_config = SimplifiedStackingConfig(
        model_type="analyst",
        output_dir=output_dir,
        **(config or {})
    )
    
    return StackingModelTrainer(simplified_config)


def create_tactician_stacking_trainer(output_dir: str = "./tactician_models",
                                     config: Optional[Dict[str, Any]] = None) -> StackingModelTrainer:
    """Create a Tactician stacking trainer."""
    
    simplified_config = SimplifiedStackingConfig(
        model_type="tactician",
        output_dir=output_dir,
        **(config or {})
    )
    
    return StackingModelTrainer(simplified_config)


def create_stacking_trainer(model_type: str, output_dir: str = "./stacking_models",
                           config: Optional[Dict[str, Any]] = None) -> StackingModelTrainer:
    """Create a stacking trainer for the specified model type."""
    
    if model_type.lower() == "analyst":
        return create_analyst_stacking_trainer(output_dir, config)
    elif model_type.lower() == "tactician":
        return create_tactician_stacking_trainer(output_dir, config)
    else:
        raise ValueError(f"Unsupported model type: {model_type}. Must be 'analyst' or 'tactician'")