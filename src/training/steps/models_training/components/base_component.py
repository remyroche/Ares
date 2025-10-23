"""
Base Models Training Component

This module provides the base component class for all models training components
that inherit from ModularComponent. It includes ML-specific functionality and
common patterns used across training components.

Key Features:
- ML-specific state management (model weights, training progress, validation metrics)
- Training-specific configuration management
- Performance monitoring for training metrics
- Model checkpointing and serialization
- Comprehensive error handling and logging
"""

import logging
import time
from typing import Any, Dict, List, Optional, Union
from abc import abstractmethod

# from ..unified_data_driven_pipeline.core.modular_architecture import (
#     ModularComponent, ErrorInfo, ErrorSeverity, ErrorCategory
# )  # REMOVED - unified pipeline deleted

# Define minimal base classes for compatibility
class ModularComponent:
    """Minimal base class for modular components."""
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None, logger: Optional[logging.Logger] = None):
        self.name = name
        self.config = config or {}
        self.logger = logger or logging.getLogger(name)
        self._initialized = False
        self._performance_stats = {}
    
    def initialize(self) -> bool:
        """Initialize the component."""
        self._initialized = True
        return True
    
    def cleanup(self) -> None:
        """Cleanup the component."""
        self._initialized = False
    
    def get_config(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self.config.get(key, default)
    
    def set_ml_state(self, key: str, value: Any) -> None:
        """Set ML state."""
        if not hasattr(self, '_ml_state'):
            self._ml_state = {}
        self._ml_state[key] = value
    
    def get_ml_state(self, key: str, default: Any = None) -> Any:
        """Get ML state."""
        if not hasattr(self, '_ml_state'):
            return default
        return self._ml_state.get(key, default)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return self._performance_stats.copy()
    
    def process(self, data: Any, **kwargs) -> Any:
        """Process data."""
        return self._process_data(data, **kwargs)
    
    def _process_data(self, data: Any, **kwargs) -> Any:
        """Process data - default implementation with validation."""
        try:
            # Basic data validation
            if data is None:
                self.logger.warning("Received None data, returning empty result")
                return {}
            
            # If data is a dictionary, return as-is
            if isinstance(data, dict):
                return data
            
            # If data is a list, convert to dictionary with indexed keys
            if isinstance(data, list):
                return {f"item_{i}": item for i, item in enumerate(data)}
            
            # For other types, wrap in a result dictionary
            return {
                "processed_data": data,
                "data_type": type(data).__name__,
                "processing_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error processing data: {e}")
            return {
                "error": str(e),
                "data_type": type(data).__name__ if data is not None else "None",
                "processing_timestamp": datetime.now().isoformat()
            }

class ErrorInfo:
    """Error information class."""
    def __init__(self, message: str, severity: str = "ERROR", category: str = "GENERAL"):
        self.message = message
        self.severity = severity
        self.category = category

class ErrorSeverity:
    """Error severity levels."""
    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"

class ErrorCategory:
    """Error categories."""
    GENERAL = "GENERAL"
    VALIDATION = "VALIDATION"
    PROCESSING = "PROCESSING"


class BaseModelsTrainingComponent(ModularComponent):
    """
    Base component class for all models training components.
    
    This class extends ModularComponent with ML-specific functionality
    and common patterns used across training components.
    """
    
    def __init__(
        self,
        name: str,
        config: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the base models training component.
        
        Args:
            name: Unique name for the component
            config: Configuration dictionary
            logger: Logger instance (optional)
        """
        super().__init__(name, config)
        
        # ML-specific configuration
        self.model_config = self.get_config('model', {})
        self.training_config = self.get_config('training', {})
        self.validation_config = self.get_config('validation', {})
        
        # Training-specific state
        self._training_state = {
            'current_epoch': 0,
            'best_epoch': 0,
            'best_metrics': {},
            'training_history': [],
            'validation_history': [],
            'model_checkpoints': [],
            'early_stopping_patience': 0,
            'early_stopping_counter': 0
        }
        
        # ML-specific capabilities
        self._ml_capabilities = {
            'supports_checkpointing': True,
            'supports_early_stopping': True,
            'supports_validation': True,
            'supports_ensemble': False,
            'supports_transfer_learning': False
        }
        
        self.logger.info(f"Initialized BaseModelsTrainingComponent: {name}")
    
    def _initialize_resources(self) -> bool:
        """Initialize component-specific resources."""
        try:
            # Initialize ML-specific state
            self.set_ml_state('initialized_at', time.time())
            self.set_ml_state('training_started', False)
            self.set_ml_state('model_created', False)
            
            # Initialize training state
            self._reset_training_state()
            
            # Initialize model if specified
            if self.training_config.get('auto_initialize_model', False):
                if not self._initialize_model():
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Resource initialization failed: {e}")
            return False
    
    def _cleanup_resources(self) -> None:
        """Cleanup component-specific resources."""
        try:
            # Clear training state
            self._reset_training_state()
            
            # Clear model state
            self.set_ml_state('model_weights', None)
            self.set_ml_state('model_created', False)
            
            self.logger.info(f"Cleaned up resources for {self.name}")
            
        except Exception as e:
            self.logger.error(f"Resource cleanup failed: {e}")
    
    def _reset_training_state(self) -> None:
        """Reset training state to initial values."""
        self._training_state = {
            'current_epoch': 0,
            'best_epoch': 0,
            'best_metrics': {},
            'training_history': [],
            'validation_history': [],
            'model_checkpoints': [],
            'early_stopping_patience': self.training_config.get('early_stopping_patience', 10),
            'early_stopping_counter': 0
        }
    
    def _initialize_model(self) -> bool:
        """Initialize the ML model."""
        try:
            model_type = self.model_config.get('type', 'neural_network')
            
            if model_type == 'neural_network':
                model = self._create_neural_network()
            elif model_type == 'tree_based':
                model = self._create_tree_based_model()
            elif model_type == 'linear':
                model = self._create_linear_model()
            elif model_type == 'ensemble':
                model = self._create_ensemble_model()
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            self.set_ml_state('model', model)
            self.set_ml_state('model_created', True)
            
            self.logger.info(f"Initialized {model_type} model")
            return True
            
        except Exception as e:
            self.logger.error(f"Model initialization failed: {e}")
            return False
    
    def _create_neural_network(self):
        """Create a neural network model."""
        # This would be implemented by subclasses
        return None
    
    def _create_tree_based_model(self):
        """Create a tree-based model."""
        # This would be implemented by subclasses
        return None
    
    def _create_linear_model(self):
        """Create a linear model."""
        # This would be implemented by subclasses
        return None
    
    def _create_ensemble_model(self):
        """Create an ensemble model."""
        # This would be implemented by subclasses
        return None
    
    def start_training(self) -> bool:
        """Start the training process."""
        try:
            if not self._initialized:
                self.logger.error("Component not initialized")
                return False
            
            self.set_ml_state('training_started', True)
            self.set_ml_state('training_start_time', time.time())
            
            self.logger.info("Training started")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start training: {e}")
            return False
    
    def stop_training(self) -> None:
        """Stop the training process."""
        try:
            self.set_ml_state('training_started', False)
            self.set_ml_state('training_end_time', time.time())
            
            # Calculate total training time
            start_time = self.get_ml_state('training_start_time', time.time())
            total_time = time.time() - start_time
            self.set_ml_state('total_training_time', total_time)
            
            self.logger.info(f"Training stopped after {total_time:.2f}s")
            
        except Exception as e:
            self.logger.error(f"Failed to stop training: {e}")
    
    def train_epoch(self, data: Any, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            data: Training data
            epoch: Current epoch number
            
        Returns:
            Training metrics for the epoch
        """
        try:
            start_time = time.time()
            
            # Get model
            model = self.get_ml_state('model')
            if model is None:
                raise RuntimeError("Model not initialized")
            
            # Train epoch (to be implemented by subclasses)
            metrics = self._train_epoch_impl(model, data, epoch)
            
            # Update training state
            self._training_state['current_epoch'] = epoch
            self._training_state['training_history'].append({
                'epoch': epoch,
                'metrics': metrics,
                'timestamp': time.time()
            })
            
            # Update performance stats
            epoch_time = time.time() - start_time
            self._update_performance_stats(True, epoch_time)
            
            # Check for early stopping
            if self._check_early_stopping(metrics):
                self.logger.info(f"Early stopping triggered at epoch {epoch}")
                self.stop_training()
            
            # Save checkpoint if needed
            if self._should_save_checkpoint(epoch):
                self._save_checkpoint(model, epoch, metrics)
            
            self.logger.info(f"Epoch {epoch} completed: {metrics}")
            return metrics
            
        except Exception as e:
            self.logger.error(f"Epoch {epoch} training failed: {e}")
            self._update_performance_stats(False, 0)
            raise
    
    def _train_epoch_impl(self, model: Any, data: Any, epoch: int) -> Dict[str, float]:
        """Implement epoch training logic."""
        try:
            # Extract training data
            if isinstance(data, dict):
                X_train = data.get('X_train')
                y_train = data.get('y_train')
            else:
                # Assume data is a tuple or list
                X_train, y_train = data[0], data[1]
            
            if X_train is None or y_train is None:
                raise ValueError("Training data must contain X_train and y_train")
            
            # Train the model
            if hasattr(model, 'fit'):
                model.fit(X_train, y_train)
            else:
                raise ValueError(f"Model {type(model).__name__} does not have fit method")
            
            # Make predictions for metrics
            if hasattr(model, 'predict'):
                train_predictions = model.predict(X_train)
            else:
                raise ValueError(f"Model {type(model).__name__} does not have predict method")
            
            # Calculate training metrics
            metrics = self._calculate_training_metrics(y_train, train_predictions)
            
            # Add epoch-specific metrics
            metrics['epoch'] = epoch
            metrics['training_loss'] = self._calculate_loss(y_train, train_predictions)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Epoch training implementation failed: {e}")
            return {
                'epoch': epoch,
                'error': str(e),
                'training_loss': float('inf')
            }
    
    def validate_epoch(self, data: Any, epoch: int) -> Dict[str, float]:
        """
        Validate for one epoch.
        
        Args:
            data: Validation data
            epoch: Current epoch number
            
        Returns:
            Validation metrics for the epoch
        """
        try:
            start_time = time.time()
            
            # Get model
            model = self.get_ml_state('model')
            if model is None:
                raise RuntimeError("Model not initialized")
            
            # Validate epoch (to be implemented by subclasses)
            metrics = self._validate_epoch_impl(model, data, epoch)
            
            # Update validation state
            self._training_state['validation_history'].append({
                'epoch': epoch,
                'metrics': metrics,
                'timestamp': time.time()
            })
            
            # Update best model if improved
            if self._is_better_model(metrics):
                self._training_state['best_epoch'] = epoch
                self._training_state['best_metrics'] = metrics
                self._training_state['early_stopping_counter'] = 0
                
                # Save best model
                self._save_best_model(model, epoch, metrics)
            else:
                self._training_state['early_stopping_counter'] += 1
            
            # Update performance stats
            epoch_time = time.time() - start_time
            self._update_performance_stats(True, epoch_time)
            
            self.logger.info(f"Epoch {epoch} validation: {metrics}")
            return metrics
            
        except Exception as e:
            self.logger.error(f"Epoch {epoch} validation failed: {e}")
            self._update_performance_stats(False, 0)
            raise
    
    def _validate_epoch_impl(self, model: Any, data: Any, epoch: int) -> Dict[str, float]:
        """Implement epoch validation logic."""
        try:
            # Extract validation data
            if isinstance(data, dict):
                X_val = data.get('X_val', data.get('X_train'))
                y_val = data.get('y_val', data.get('y_train'))
            else:
                # Assume data is a tuple or list
                X_val, y_val = data[0], data[1]
            
            if X_val is None or y_val is None:
                raise ValueError("Validation data must contain X_val and y_val")
            
            # Make predictions
            if hasattr(model, 'predict'):
                val_predictions = model.predict(X_val)
            else:
                raise ValueError(f"Model {type(model).__name__} does not have predict method")
            
            # Calculate validation metrics
            metrics = self._calculate_validation_metrics(y_val, val_predictions)
            
            # Add epoch-specific metrics
            metrics['epoch'] = epoch
            metrics['validation_loss'] = self._calculate_loss(y_val, val_predictions)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Epoch validation implementation failed: {e}")
            return {
                'epoch': epoch,
                'error': str(e),
                'validation_loss': float('inf')
            }
    
    def _check_early_stopping(self, metrics: Dict[str, float]) -> bool:
        """Check if early stopping should be triggered."""
        patience = self._training_state['early_stopping_patience']
        counter = self._training_state['early_stopping_counter']
        
        return patience > 0 and counter >= patience
    
    def _should_save_checkpoint(self, epoch: int) -> bool:
        """Check if a checkpoint should be saved."""
        checkpoint_frequency = self.training_config.get('checkpoint_frequency', 10)
        return epoch % checkpoint_frequency == 0
    
    def _save_checkpoint(self, model: Any, epoch: int, metrics: Dict[str, float]) -> None:
        """Save a model checkpoint."""
        try:
            checkpoint = {
                'epoch': epoch,
                'model_state': self._get_model_state(model),
                'metrics': metrics,
                'timestamp': time.time()
            }
            
            self._training_state['model_checkpoints'].append(checkpoint)
            self.set_ml_state('latest_checkpoint', checkpoint)
            
            self.logger.info(f"Checkpoint saved for epoch {epoch}")
            
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {e}")
    
    def _save_best_model(self, model: Any, epoch: int, metrics: Dict[str, float]) -> None:
        """Save the best model."""
        try:
            best_model = {
                'epoch': epoch,
                'model_state': self._get_model_state(model),
                'metrics': metrics,
                'timestamp': time.time()
            }
            
            self.set_ml_state('best_model', best_model)
            self.logger.info(f"Best model saved for epoch {epoch}")
            
        except Exception as e:
            self.logger.error(f"Failed to save best model: {e}")
    
    def _get_model_state(self, model: Any) -> Any:
        """Get model state for serialization."""
        # This would be implemented by subclasses based on model type
        return model
    
    def _is_better_model(self, metrics: Dict[str, float]) -> bool:
        """Check if current model is better than best model."""
        if not self._training_state['best_metrics']:
            return True
        
        best_metrics = self._training_state['best_metrics']
        
        # Use primary metric for comparison
        primary_metric = self.training_config.get('primary_metric', 'accuracy')
        
        if primary_metric in metrics and primary_metric in best_metrics:
            return metrics[primary_metric] > best_metrics[primary_metric]
        
        return False
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get comprehensive training summary."""
        return {
            'component_name': self.name,
            'training_state': self._training_state.copy(),
            'ml_state': self.get_ml_state('all', {}),
            'performance_stats': self.get_performance_stats(),
            'model_info': self._get_model_info(),
            'training_time': self.get_ml_state('total_training_time', 0),
            'best_epoch': self._training_state['best_epoch'],
            'best_metrics': self._training_state['best_metrics']
        }
    
    def _get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        model = self.get_ml_state('model')
        if model is None:
            return {'type': 'none', 'created': False}
        
        return {
            'type': type(model).__name__,
            'created': True,
            'checkpoints': len(self._training_state['model_checkpoints']),
            'best_model_saved': self.get_ml_state('best_model') is not None
        }
    
    def _get_validation_rules(self) -> Dict[str, Any]:
        """Get validation rules for this component."""
        return {
            'min_size': 100,
            'max_size': 1000000,
            'required_keys': ['X_train', 'y_train'],
            'data_types': ['dict', 'pandas.DataFrame'],
            'required_columns': ['X_train', 'y_train']
        }
    
    def _validate_component_specific(self, data: Any) -> Dict[str, Any]:
        """Validate data with component-specific rules."""
        errors = []
        warnings = []
        metadata = {}
        
        if isinstance(data, dict):
            # Check required keys
            required_keys = ['X_train', 'y_train']
            for key in required_keys:
                if key not in data:
                    errors.append(f"Missing required key: {key}")
            
            # Check data shapes
            if 'X_train' in data and 'y_train' in data:
                X_train = data['X_train']
                y_train = data['y_train']
                
                if hasattr(X_train, 'shape') and hasattr(y_train, 'shape'):
                    metadata['X_train_shape'] = X_train.shape
                    metadata['y_train_shape'] = y_train.shape
                    
                    if len(X_train) != len(y_train):
                        errors.append("X_train and y_train must have same number of samples")
                    
                    if len(X_train) < 100:
                        warnings.append("Training data is small, consider more data")
        
        return {'errors': errors, 'warnings': warnings, 'metadata': metadata}
    
    def get_dependencies(self) -> List[str]:
        """Get list of required dependencies."""
        return ['pandas', 'numpy', 'torch', 'sklearn']
    
    def get_processing_capabilities(self) -> Dict[str, Any]:
        """Get component processing capabilities."""
        return {
            'input_types': ['dict', 'pandas.DataFrame'],
            'output_types': ['dict'],
            'parallel_processing': False,
            'memory_efficient': True,
            'supports_checkpointing': True,
            'supports_validation': True,
            'supports_early_stopping': True,
            'supports_ensemble': self._ml_capabilities['supports_ensemble'],
            'supports_transfer_learning': self._ml_capabilities['supports_transfer_learning']
        }
    
    def get_required_config(self) -> List[str]:
        """Get required configuration parameters."""
        return ['model', 'training']
    
    def estimate_processing_time(self, data: Any) -> float:
        """Estimate processing time for given data."""
        base_time = self.get_config('base_processing_time', 5.0)
        
        # Size-based factor
        if isinstance(data, dict) and 'X_train' in data:
            data_size = len(data['X_train'])
        elif hasattr(data, '__len__'):
            data_size = len(data)
        else:
            data_size = 1000
        
        # Epochs factor
        epochs = self.training_config.get('epochs', 100)
        
        # Model complexity factor
        model_type = self.model_config.get('type', 'neural_network')
        complexity_factors = {
            'neural_network': 1.0,
            'tree_based': 0.5,
            'linear': 0.3,
            'ensemble': 2.0
        }
        complexity_factor = complexity_factors.get(model_type, 1.0)
        
        return base_time * (data_size / 1000) * epochs * complexity_factor
    
    def get_memory_requirements(self, data: Any) -> Dict[str, Any]:
        """Get memory requirements for processing data."""
        base_memory = 200  # MB
        
        # Calculate data memory usage
        if isinstance(data, dict) and 'X_train' in data:
            X_train = data['X_train']
            if hasattr(X_train, 'memory_usage'):
                data_memory = X_train.memory_usage(deep=True).sum() / (1024 * 1024)  # MB
            else:
                data_memory = 100  # Estimate
        else:
            data_memory = 100  # Estimate
        
        # Model memory (depends on model type)
        model_type = self.model_config.get('type', 'neural_network')
        model_memory = {
            'neural_network': 500,
            'tree_based': 100,
            'linear': 50,
            'ensemble': 1000
        }.get(model_type, 200)
        
        # Overhead factor
        overhead_factor = self.get_config('memory_overhead_factor', 2.0)
        
        estimated_memory = (base_memory + data_memory + model_memory) * overhead_factor
        peak_memory = estimated_memory * 1.5  # 50% buffer
        
        return {
            'estimated_memory_mb': estimated_memory,
            'peak_memory_mb': peak_memory,
            'data_memory_mb': data_memory,
            'model_memory_mb': model_memory,
            'base_memory_mb': base_memory
        }
    
    # Helper methods for the implemented abstract methods
    
    def _calculate_training_metrics(self, y_true, y_pred) -> Dict[str, float]:
        """Calculate training metrics."""
        try:
            import numpy as np
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score
            
            metrics = {}
            
            # Determine if classification or regression
            unique_values = len(np.unique(y_true))
            is_classification = unique_values <= 20  # Heuristic for classification
            
            if is_classification:
                # Classification metrics
                metrics['accuracy'] = accuracy_score(y_true, y_pred)
                try:
                    metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
                    metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
                    metrics['f1'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
                except:
                    metrics['precision'] = 0.0
                    metrics['recall'] = 0.0
                    metrics['f1'] = 0.0
            else:
                # Regression metrics
                metrics['mse'] = mean_squared_error(y_true, y_pred)
                metrics['rmse'] = np.sqrt(metrics['mse'])
                metrics['r2'] = r2_score(y_true, y_pred)
                
                # Additional regression metrics
                mae = np.mean(np.abs(y_true - y_pred))
                metrics['mae'] = mae
                
                # Mean absolute percentage error
                mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
                metrics['mape'] = mape
            
            return metrics
            
        except Exception as e:
            self.logger.warning(f"Failed to calculate training metrics: {e}")
            return {'error': str(e)}
    
    def _calculate_validation_metrics(self, y_true, y_pred) -> Dict[str, float]:
        """Calculate validation metrics."""
        try:
            import numpy as np
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score
            
            metrics = {}
            
            # Determine if classification or regression
            unique_values = len(np.unique(y_true))
            is_classification = unique_values <= 20  # Heuristic for classification
            
            if is_classification:
                # Classification metrics
                metrics['val_accuracy'] = accuracy_score(y_true, y_pred)
                try:
                    metrics['val_precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
                    metrics['val_recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
                    metrics['val_f1'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
                except:
                    metrics['val_precision'] = 0.0
                    metrics['val_recall'] = 0.0
                    metrics['val_f1'] = 0.0
            else:
                # Regression metrics
                metrics['val_mse'] = mean_squared_error(y_true, y_pred)
                metrics['val_rmse'] = np.sqrt(metrics['val_mse'])
                metrics['val_r2'] = r2_score(y_true, y_pred)
                
                # Additional regression metrics
                mae = np.mean(np.abs(y_true - y_pred))
                metrics['val_mae'] = mae
                
                # Mean absolute percentage error
                mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
                metrics['val_mape'] = mape
            
            return metrics
            
        except Exception as e:
            self.logger.warning(f"Failed to calculate validation metrics: {e}")
            return {'error': str(e)}
    
    def _calculate_loss(self, y_true, y_pred) -> float:
        """Calculate loss for the given predictions."""
        try:
            import numpy as np
            
            # Use mean squared error as default loss
            mse = np.mean((y_true - y_pred) ** 2)
            return float(mse)
            
        except Exception as e:
            self.logger.warning(f"Failed to calculate loss: {e}")
            return float('inf')
    
    def _update_performance_stats(self, success: bool, duration: float) -> None:
        """Update performance statistics."""
        try:
            if success:
                self._performance_stats['successful_operations'] = self._performance_stats.get('successful_operations', 0) + 1
                self._performance_stats['total_success_time'] = self._performance_stats.get('total_success_time', 0.0) + duration
            else:
                self._performance_stats['failed_operations'] = self._performance_stats.get('failed_operations', 0) + 1
                self._performance_stats['total_failure_time'] = self._performance_stats.get('total_failure_time', 0.0) + duration
            
            # Update averages
            total_ops = self._performance_stats.get('successful_operations', 0) + self._performance_stats.get('failed_operations', 0)
            if total_ops > 0:
                self._performance_stats['success_rate'] = self._performance_stats.get('successful_operations', 0) / total_ops
                self._performance_stats['avg_operation_time'] = (
                    self._performance_stats.get('total_success_time', 0.0) + 
                    self._performance_stats.get('total_failure_time', 0.0)
                ) / total_ops
            
        except Exception as e:
            self.logger.warning(f"Failed to update performance stats: {e}")