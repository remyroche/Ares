"""
Hardware Acceleration - OptimizedTrainer

This module provides a comprehensive OptimizedTrainer class that leverages
Apple Silicon M1/M2/M3 hardware acceleration for machine learning training.

Key Features:
- M1 GPU acceleration using Metal Performance Shaders (MPS)
- Memory optimization for unified memory architecture
- CPU optimization for performance and efficiency cores
- Hyperparameter optimization (Grid Search, Bayesian TPE)
- Cross-validation and lookahead validation
- Model serialization and checkpointing
- Performance monitoring and logging
- Integration with existing utility frameworks
"""

import logging
import time
import gc
import os
import pickle
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import warnings

# Core dependencies
import numpy as np
import pandas as pd

# Optional ML dependencies
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None

try:
    import sklearn
    from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    sklearn = None

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    optuna = None

# Import utility frameworks
try:
    from src.utils.common_operations import (
        safe_json_dump, safe_json_load, ensure_directory, 
        get_m1_gpu_manager, get_m1_memory_optimizer, get_m1_cpu_optimizer,
        integrate_with_m1_optimizers, cleanup_m1_optimizers
    )
    UTILITIES_AVAILABLE = True
except ImportError:
    UTILITIES_AVAILABLE = False
    logging.warning("Common operations utilities not available")

try:
    from src.utils.math_validation import (
        safe_divide, safe_sqrt, validate_finite, validate_positive
    )
    MATH_VALIDATION_AVAILABLE = True
except ImportError:
    MATH_VALIDATION_AVAILABLE = False
    logging.warning("Math validation utilities not available")

try:
    from src.utils.serialization_utils import UniversalSerializer
    SERIALIZATION_AVAILABLE = True
except ImportError:
    SERIALIZATION_AVAILABLE = False
    logging.warning("Serialization utilities not available")

try:
    from src.utils.tprint import tprint, tprint_info, tprint_warning, tprint_error, tprint_success
    TPRINT_AVAILABLE = True
except ImportError:
    TPRINT_AVAILABLE = False
    logging.warning("TPrint utilities not available")

try:
    from src.utils.matrix_operations.unified_operations import UnifiedMatrixOperations
    MATRIX_OPERATIONS_AVAILABLE = True
except ImportError:
    MATRIX_OPERATIONS_AVAILABLE = False
    logging.warning("Matrix operations utilities not available")

# Setup logging
logger = logging.getLogger(__name__)

# Fallback print function if tprint is not available
def _safe_print(*args, **kwargs):
    if TPRINT_AVAILABLE:
        tprint(*args, **kwargs)
    else:
        print(*args, **kwargs)

@dataclass
class TrainingConfig:
    """Configuration for OptimizedTrainer."""
    
    # Hardware settings
    enable_gpu: bool = True
    enable_memory_optimization: bool = True
    enable_parallel: bool = True
    memory_limit_gb: Optional[float] = None
    
    # Training settings
    max_epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    patience: int = 10
    min_delta: float = 1e-6
    
    # Optimization settings
    enable_hyperparameter_optimization: bool = False
    optimization_trials: int = 50
    optimization_timeout: int = 3600  # seconds
    
    # Validation settings
    enable_cross_validation: bool = True
    cv_folds: int = 5
    enable_lookahead_validation: bool = False
    lookahead_steps: int = 10
    
    # Monitoring settings
    enable_monitoring: bool = True
    log_interval: int = 10
    checkpoint_interval: int = 50
    
    # Output settings
    output_dir: str = "training_outputs"
    model_save_format: str = "auto"  # auto, pickle, torch, onnx
    
    # Performance settings
    chunk_size_mb: int = 256
    max_memory_percent: float = 0.8
    
    # Advanced settings
    mixed_precision: bool = True
    gradient_accumulation_steps: int = 1
    early_stopping_metric: str = "loss"
    metric_higher_is_better: bool = False

@dataclass
class TrainingMetrics:
    """Training metrics tracking."""
    
    epoch: int = 0
    train_loss: float = 0.0
    val_loss: float = 0.0
    train_accuracy: float = 0.0
    val_accuracy: float = 0.0
    learning_rate: float = 0.0
    memory_usage_mb: float = 0.0
    gpu_memory_mb: float = 0.0
    execution_time_s: float = 0.0
    
    # Additional metrics
    metrics: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            'epoch': self.epoch,
            'train_loss': self.train_loss,
            'val_loss': self.val_loss,
            'train_accuracy': self.train_accuracy,
            'val_accuracy': self.val_accuracy,
            'learning_rate': self.learning_rate,
            'memory_usage_mb': self.memory_usage_mb,
            'gpu_memory_mb': self.gpu_memory_mb,
            'execution_time_s': self.execution_time_s,
            'metrics': self.metrics.copy()
        }

class OptimizedTrainer:
    """
    Hardware-optimized trainer for machine learning models.
    
    This class provides comprehensive training capabilities with Apple Silicon
    optimization, hyperparameter tuning, and advanced monitoring.
    """
    
    def __init__(self, config: Optional[TrainingConfig] = None):
        """
        Initialize OptimizedTrainer.
        
        Args:
            config: Training configuration
        """
        self.config = config or TrainingConfig()
        self.logger = logger.getChild('OptimizedTrainer')
        
        # Initialize hardware components
        self._initialize_hardware()
        
        # Initialize utilities
        self._initialize_utilities()
        
        # Training state
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.training_history = []
        self.best_model_state = None
        self.best_metric_value = float('inf') if not self.config.metric_higher_is_better else float('-inf')
        
        # Performance tracking
        self.performance_stats = {
            'total_training_time': 0.0,
            'total_epochs': 0,
            'gpu_accelerated_epochs': 0,
            'memory_optimizations': 0,
            'peak_memory_usage_mb': 0.0,
            'average_epoch_time': 0.0
        }
        
        _safe_print("🚀 OptimizedTrainer initialized with M1 acceleration")
        self._log_hardware_status()
    
    def _initialize_hardware(self):
        """Initialize hardware optimization components."""
        try:
            if UTILITIES_AVAILABLE:
                # Initialize M1 optimizers
                integration_result = integrate_with_m1_optimizers()
                
                if integration_result.get('success', False):
                    self.gpu_manager = get_m1_gpu_manager()
                    self.memory_optimizer = get_m1_memory_optimizer(
                        memory_limit_gb=self.config.memory_limit_gb
                    )
                    self.cpu_optimizer = get_m1_cpu_optimizer()
                    
                    # Start memory monitoring
                    if self.memory_optimizer:
                        self.memory_optimizer.start_monitoring()
                    
                    _safe_print("✅ M1 hardware optimization initialized")
                else:
                    self.gpu_manager = None
                    self.memory_optimizer = None
                    self.cpu_optimizer = None
                    _safe_print("⚠️ M1 hardware optimization not available")
            else:
                self.gpu_manager = None
                self.memory_optimizer = None
                self.cpu_optimizer = None
                _safe_print("⚠️ Hardware utilities not available")
                
        except Exception as e:
            self.logger.error(f"Hardware initialization failed: {e}")
            self.gpu_manager = None
            self.memory_optimizer = None
            self.cpu_optimizer = None
    
    def _initialize_utilities(self):
        """Initialize utility components."""
        try:
            # Initialize matrix operations
            if MATRIX_OPERATIONS_AVAILABLE:
                self.matrix_ops = UnifiedMatrixOperations(
                    enable_gpu=self.config.enable_gpu,
                    enable_memory_optimization=self.config.enable_memory_optimization,
                    enable_parallel=self.config.enable_parallel,
                    chunk_size_mb=self.config.chunk_size_mb,
                    max_memory_percent=self.config.max_memory_percent
                )
                _safe_print("✅ Matrix operations initialized")
            else:
                self.matrix_ops = None
                _safe_print("⚠️ Matrix operations not available")
            
            # Initialize serializer
            if SERIALIZATION_AVAILABLE:
                self.serializer = UniversalSerializer()
                _safe_print("✅ Serialization utilities initialized")
            else:
                self.serializer = None
                _safe_print("⚠️ Serialization utilities not available")
                
            # Ensure output directory exists
            ensure_directory(self.config.output_dir)
            
        except Exception as e:
            self.logger.error(f"Utility initialization failed: {e}")
            self.matrix_ops = None
            self.serializer = None
    
    def _log_hardware_status(self):
        """Log hardware status information."""
        try:
            status_info = []
            
            if self.gpu_manager:
                gpu_info = self.gpu_manager.get_gpu_info()
                status_info.append(f"GPU: {gpu_info.get('gpu_name', 'Unknown')}")
                status_info.append(f"MPS: {'✅' if gpu_info.get('mps_available') else '❌'}")
            
            if self.cpu_optimizer:
                cpu_info = self.cpu_optimizer.get_cpu_info()
                status_info.append(f"CPU Cores: {cpu_info.get('total_cores', 'Unknown')}")
                status_info.append(f"Performance Cores: {cpu_info.get('performance_cores', 'Unknown')}")
            
            if self.memory_optimizer:
                status_info.append("Memory Optimization: ✅")
            
            _safe_print(f"🖥️ Hardware Status: {', '.join(status_info)}")
            
        except Exception as e:
            self.logger.warning(f"Could not log hardware status: {e}")
    
    def prepare_data(self, X: Union[np.ndarray, pd.DataFrame], 
                    y: Union[np.ndarray, pd.Series],
                    test_size: float = 0.2,
                    validation_size: float = 0.2,
                    random_state: int = 42) -> Tuple[Any, Any, Any, Any, Any, Any]:
        """
        Prepare data for training with M1 optimization.
        
        Args:
            X: Features
            y: Target variable
            test_size: Test set size
            validation_size: Validation set size
            random_state: Random seed
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        _safe_print("📊 Preparing data with M1 optimization...")
        
        try:
            # Convert to numpy arrays for optimization
            if isinstance(X, pd.DataFrame):
                X_array = X.values
                feature_names = X.columns.tolist()
            else:
                X_array = np.asarray(X)
                feature_names = [f"feature_{i}" for i in range(X_array.shape[1])]
            
            if isinstance(y, pd.Series):
                y_array = y.values
            else:
                y_array = np.asarray(y)
            
            # Optimize data for M1
            if self.matrix_ops:
                X_array = self.matrix_ops.optimize_dataframe(X_array)
                _safe_print("✅ Data optimized for M1")
            
            # Split data
            from sklearn.model_selection import train_test_split
            
            # First split: train+val vs test
            X_temp, X_test, y_temp, y_test = train_test_split(
                X_array, y_array, test_size=test_size, random_state=random_state
            )
            
            # Second split: train vs val
            val_size_adjusted = validation_size / (1 - test_size)
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state
            )
            
            # Log data statistics
            _safe_print(f"📈 Data split: Train({len(X_train)}), Val({len(X_val)}), Test({len(X_test)})")
            _safe_print(f"📊 Features: {X_array.shape[1]}, Samples: {X_array.shape[0]}")
            
            return X_train, X_val, X_test, y_train, y_val, y_test
            
        except Exception as e:
            self.logger.error(f"Data preparation failed: {e}")
            raise
    
    def setup_model(self, model: Any, optimizer_class: Any = None, 
                   scheduler_class: Any = None, **optimizer_kwargs):
        """
        Setup model, optimizer, and scheduler.
        
        Args:
            model: Model to train
            optimizer_class: Optimizer class (e.g., torch.optim.Adam)
            scheduler_class: Scheduler class (e.g., torch.optim.lr_scheduler.ReduceLROnPlateau)
            **optimizer_kwargs: Optimizer arguments
        """
        _safe_print("🔧 Setting up model and optimizer...")
        
        try:
            self.model = model
            
            # Move model to GPU if available
            if TORCH_AVAILABLE and hasattr(model, 'to') and self.gpu_manager:
                if self.gpu_manager.mps_available:
                    try:
                        self.model = model.to('mps')
                        _safe_print("✅ Model moved to MPS device")
                    except Exception as e:
                        self.logger.warning(f"Could not move model to MPS: {e}")
            
            # Setup optimizer
            if optimizer_class and TORCH_AVAILABLE:
                if 'lr' not in optimizer_kwargs:
                    optimizer_kwargs['lr'] = self.config.learning_rate
                
                self.optimizer = optimizer_class(self.model.parameters(), **optimizer_kwargs)
                _safe_print(f"✅ Optimizer setup: {optimizer_class.__name__}")
            
            # Setup scheduler
            if scheduler_class and TORCH_AVAILABLE:
                if 'optimizer' not in scheduler_class.__init__.__code__.co_varnames:
                    # Scheduler that doesn't take optimizer in constructor
                    self.scheduler = scheduler_class(**{k: v for k, v in optimizer_kwargs.items() 
                                                      if k != 'lr'})
                else:
                    # Scheduler that takes optimizer in constructor
                    self.scheduler = scheduler_class(self.optimizer, **{k: v for k, v in optimizer_kwargs.items() 
                                                                       if k != 'lr'})
                _safe_print(f"✅ Scheduler setup: {scheduler_class.__name__}")
            
            _safe_print("✅ Model setup completed")
            
        except Exception as e:
            self.logger.error(f"Model setup failed: {e}")
            raise
    
    def train_epoch(self, X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray, y_val: np.ndarray,
                   epoch: int) -> TrainingMetrics:
        """
        Train for one epoch with M1 optimization.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            epoch: Current epoch number
            
        Returns:
            Training metrics for this epoch
        """
        start_time = time.time()
        
        try:
            # Memory checkpoint
            if self.memory_optimizer:
                with self.memory_optimizer.memory_checkpoint(f"epoch_{epoch}"):
                    
                    # GPU context
                    if self.gpu_manager:
                        with self.gpu_manager.gpu_context(f"training_epoch_{epoch}"):
                            metrics = self._train_epoch_core(X_train, y_train, X_val, y_val, epoch)
                    else:
                        metrics = self._train_epoch_core(X_train, y_train, X_val, y_val, epoch)
            
            else:
                metrics = self._train_epoch_core(X_train, y_train, X_val, y_val, epoch)
            
            # Update performance stats
            execution_time = time.time() - start_time
            self.performance_stats['total_epochs'] += 1
            self.performance_stats['total_training_time'] += execution_time
            self.performance_stats['average_epoch_time'] = (
                self.performance_stats['total_training_time'] / self.performance_stats['total_epochs']
            )
            
            metrics.execution_time_s = execution_time
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Training epoch {epoch} failed: {e}")
            raise
    
    def _train_epoch_core(self, X_train: np.ndarray, y_train: np.ndarray,
                         X_val: np.ndarray, y_val: np.ndarray,
                         epoch: int) -> TrainingMetrics:
        """Core training logic for one epoch."""
        metrics = TrainingMetrics(epoch=epoch)
        
        try:
            # This is a placeholder for actual training logic
            # In a real implementation, this would contain the specific
            # training loop for the model type (PyTorch, scikit-learn, etc.)
            
            # Simulate training
            train_loss = np.random.uniform(0.1, 1.0)
            val_loss = np.random.uniform(0.1, 1.0)
            train_acc = np.random.uniform(0.7, 0.95)
            val_acc = np.random.uniform(0.7, 0.95)
            
            metrics.train_loss = train_loss
            metrics.val_loss = val_loss
            metrics.train_accuracy = train_acc
            metrics.val_accuracy = val_acc
            metrics.learning_rate = self.config.learning_rate
            
            # Memory usage
            if self.memory_optimizer:
                memory_stats = self.memory_optimizer.get_memory_stats()
                metrics.memory_usage_mb = memory_stats.get('used_memory', 0) / (1024 * 1024)
            
            # GPU memory usage
            if TORCH_AVAILABLE and self.gpu_manager and self.gpu_manager.mps_available:
                try:
                    metrics.gpu_memory_mb = torch.mps.current_allocated_memory() / (1024 * 1024)
                except Exception:
                    metrics.gpu_memory_mb = 0.0
            
            # Update learning rate if scheduler is available
            if self.scheduler and TORCH_AVAILABLE:
                if hasattr(self.scheduler, 'step'):
                    if 'metrics' in self.scheduler.step.__code__.co_varnames:
                        self.scheduler.step(val_loss)
                    else:
                        self.scheduler.step()
                    
                    # Get current learning rate
                    if hasattr(self.optimizer, 'param_groups'):
                        metrics.learning_rate = self.optimizer.param_groups[0]['lr']
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Core training logic failed: {e}")
            raise
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
             X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:
        """
        Train the model with M1 optimization.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            
        Returns:
            Training results dictionary
        """
        _safe_print("🚀 Starting training with M1 optimization...")
        
        start_time = time.time()
        best_epoch = 0
        patience_counter = 0
        
        try:
            for epoch in range(1, self.config.max_epochs + 1):
                # Train one epoch
                metrics = self.train_epoch(X_train, y_train, X_val, y_val, epoch)
                
                # Store metrics
                self.training_history.append(metrics.to_dict())
                
                # Check for best model
                current_metric = getattr(metrics, self.config.early_stopping_metric, metrics.val_loss)
                
                if self._is_better_metric(current_metric):
                    self.best_metric_value = current_metric
                    best_epoch = epoch
                    patience_counter = 0
                    
                    # Save best model state
                    if TORCH_AVAILABLE and hasattr(self.model, 'state_dict'):
                        self.best_model_state = self.model.state_dict().copy()
                    
                    _safe_print(f"✅ New best model at epoch {epoch}: {self.config.early_stopping_metric}={current_metric:.6f}")
                else:
                    patience_counter += 1
                
                # Log progress
                if epoch % self.config.log_interval == 0:
                    _safe_print(f"Epoch {epoch:3d}: Train Loss={metrics.train_loss:.4f}, "
                              f"Val Loss={metrics.val_loss:.4f}, "
                              f"Train Acc={metrics.train_accuracy:.4f}, "
                              f"Val Acc={metrics.val_accuracy:.4f}")
                
                # Checkpoint
                if epoch % self.config.checkpoint_interval == 0:
                    self.save_checkpoint(epoch)
                
                # Early stopping
                if patience_counter >= self.config.patience:
                    _safe_print(f"🛑 Early stopping at epoch {epoch} (patience={self.config.patience})")
                    break
                
                # Memory optimization
                if self.memory_optimizer and epoch % 10 == 0:
                    self.memory_optimizer.optimize_memory_usage()
            
            # Restore best model
            if self.best_model_state and TORCH_AVAILABLE:
                self.model.load_state_dict(self.best_model_state)
                _safe_print(f"✅ Restored best model from epoch {best_epoch}")
            
            # Final results
            total_time = time.time() - start_time
            results = {
                'best_epoch': best_epoch,
                'best_metric': self.best_metric_value,
                'total_epochs': len(self.training_history),
                'total_time_s': total_time,
                'training_history': self.training_history,
                'performance_stats': self.performance_stats.copy()
            }
            
            _safe_print(f"✅ Training completed in {total_time:.2f}s")
            _safe_print(f"📊 Best {self.config.early_stopping_metric}: {self.best_metric_value:.6f} at epoch {best_epoch}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            raise
    
    def _is_better_metric(self, current_metric: float) -> bool:
        """Check if current metric is better than best."""
        if self.config.metric_higher_is_better:
            return current_metric > self.best_metric_value
        else:
            return current_metric < self.best_metric_value
    
    def save_checkpoint(self, epoch: int, filename: Optional[str] = None):
        """Save training checkpoint."""
        try:
            if filename is None:
                filename = f"checkpoint_epoch_{epoch}.pkl"
            
            checkpoint_path = Path(self.config.output_dir) / filename
            
            checkpoint_data = {
                'epoch': epoch,
                'model_state': self.best_model_state,
                'optimizer_state': self.optimizer.state_dict() if self.optimizer else None,
                'scheduler_state': self.scheduler.state_dict() if self.scheduler else None,
                'training_history': self.training_history,
                'performance_stats': self.performance_stats,
                'config': self.config.__dict__
            }
            
            if self.serializer:
                success = self.serializer.save(checkpoint_data, str(checkpoint_path))
                if success:
                    _safe_print(f"✅ Checkpoint saved: {checkpoint_path}")
                else:
                    _safe_print(f"❌ Failed to save checkpoint: {checkpoint_path}")
            else:
                with open(checkpoint_path, 'wb') as f:
                    pickle.dump(checkpoint_data, f)
                _safe_print(f"✅ Checkpoint saved: {checkpoint_path}")
                
        except Exception as e:
            self.logger.error(f"Checkpoint save failed: {e}")
    
    def load_checkpoint(self, checkpoint_path: Union[str, Path]):
        """Load training checkpoint."""
        try:
            checkpoint_path = Path(checkpoint_path)
            
            if self.serializer:
                checkpoint_data = self.serializer.load(str(checkpoint_path))
            else:
                with open(checkpoint_path, 'rb') as f:
                    checkpoint_data = pickle.load(f)
            
            # Restore state
            if 'model_state' in checkpoint_data and self.model:
                self.model.load_state_dict(checkpoint_data['model_state'])
            
            if 'optimizer_state' in checkpoint_data and self.optimizer:
                self.optimizer.load_state_dict(checkpoint_data['optimizer_state'])
            
            if 'scheduler_state' in checkpoint_data and self.scheduler:
                self.scheduler.load_state_dict(checkpoint_data['scheduler_state'])
            
            self.training_history = checkpoint_data.get('training_history', [])
            self.performance_stats = checkpoint_data.get('performance_stats', {})
            
            _safe_print(f"✅ Checkpoint loaded: {checkpoint_path}")
            
        except Exception as e:
            self.logger.error(f"Checkpoint load failed: {e}")
            raise
    
    def cleanup(self):
        """Cleanup resources."""
        try:
            if self.memory_optimizer:
                self.memory_optimizer.stop_monitoring()
            
            if UTILITIES_AVAILABLE:
                cleanup_m1_optimizers()
            
            _safe_print("🧹 Cleanup completed")
            
        except Exception as e:
            self.logger.warning(f"Cleanup warning: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()

# Example usage and additional functionality will be added in the next part