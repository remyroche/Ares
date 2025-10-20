"""
GPU Acceleration Utilities for ML Common

This module provides GPU acceleration utilities specifically designed for ML operations
with fallback to CPU processing when GPU is not available.
"""

import logging
import time
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
from functools import wraps
import pandas as pd
import numpy as np

# Hardware optimization imports
from ..hardware.enhanced_gpu_manager import (
    get_enhanced_gpu_manager, EnhancedM1GPUManager, GPUOperationType
)
from ..hardware.adaptive_optimization_engine import (
    get_adaptive_optimization_engine, AdaptiveOptimizationEngine
)

logger = logging.getLogger(__name__)

class GPUAccelerationUtils:
    """
    GPU acceleration utilities for ML operations.
    """

    def __init__(self, enable_gpu: bool = True):
        """Initialize GPU acceleration utilities."""
        self.enable_gpu = enable_gpu
        self.gpu_manager = get_enhanced_gpu_manager() if enable_gpu else None
        self.adaptive_engine = get_adaptive_optimization_engine() if enable_gpu else None

    def is_gpu_available(self) -> bool:
        """Check if GPU is available and suitable for operations."""
        if not self.enable_gpu or not self.gpu_manager:
            return False
        return self.gpu_manager.is_gpu_available()

    def should_use_gpu(self, data_size: int, operation_type: str) -> bool:
        """Determine if GPU should be used based on data characteristics."""
        if not self.is_gpu_available():
            return False
        
        # Get optimal strategy from adaptive engine
        if self.adaptive_engine:
            try:
                strategy = self.adaptive_engine.get_optimal_strategy(
                    operation_type,
                    {
                        'memory_pressure': self._get_memory_pressure(),
                        'data_size': data_size
                    }
                )
                return strategy.get('use_gpu', False)
            except Exception as e:
                logger.warning(f"Failed to get GPU strategy: {e}")
        
        # Fallback: use GPU for large datasets
        return data_size > 10000

    def accelerate_matrix_operations(self, func: Callable, *args, **kwargs):
        """Accelerate matrix operations with GPU if suitable."""
        if not self.should_use_gpu(self._estimate_data_size(args), 'matrix_operations'):
            return func(*args, **kwargs)
        
        try:
            return self.gpu_manager.execute_gpu_operation(
                func, GPUOperationType.MATRIX_MULTIPLICATION, *args, **kwargs
            )
        except Exception as e:
            logger.warning(f"GPU matrix operations failed: {e}, falling back to CPU")
            return func(*args, **kwargs)

    def accelerate_feature_engineering(self, df: pd.DataFrame, feature_funcs: List[Callable], **kwargs):
        """Accelerate feature engineering with GPU if suitable."""
        if not self.should_use_gpu(len(df), 'feature_engineering'):
            return self._cpu_feature_engineering(df, feature_funcs, **kwargs)
        
        try:
            # Convert DataFrame to GPU format
            gpu_data = self.gpu_manager.prepare_data_for_gpu(df)
            
            results = []
            for func in feature_funcs:
                # Execute on GPU
                result = self.gpu_manager.execute_gpu_operation(
                    func, GPUOperationType.MATRIX_MULTIPLICATION, gpu_data, **kwargs
                )
                # Convert back to DataFrame
                result_df = self.gpu_manager.convert_from_gpu(result, df.index, df.columns)
                results.append(result_df)
            
            return pd.concat(results, axis=1)
        except Exception as e:
            logger.warning(f"GPU feature engineering failed: {e}, falling back to CPU")
            return self._cpu_feature_engineering(df, feature_funcs, **kwargs)

    def accelerate_rolling_operations(self, df: pd.DataFrame, window_sizes: List[int], operation: str = 'mean'):
        """Accelerate rolling operations with GPU if suitable."""
        if not self.should_use_gpu(len(df), 'rolling_operations'):
            return self._cpu_rolling_operations(df, window_sizes, operation)
        
        try:
            # Convert DataFrame to GPU format
            gpu_data = self.gpu_manager.prepare_data_for_gpu(df)
            
            results = []
            for window_size in window_sizes:
                # Execute rolling operation on GPU
                result = self.gpu_manager.rolling_operation(
                    gpu_data, operation, window_size
                )
                # Convert back to DataFrame
                result_df = self.gpu_manager.convert_from_gpu(result, df.index, df.columns)
                results.append(result_df)
            
            return pd.concat(results, axis=1)
        except Exception as e:
            logger.warning(f"GPU rolling operations failed: {e}, falling back to CPU")
            return self._cpu_rolling_operations(df, window_sizes, operation)

    def accelerate_hyperparameter_optimization(self, model, X, y, param_grid, **kwargs):
        """Accelerate hyperparameter optimization with GPU if suitable."""
        if not self.should_use_gpu(X.shape[0] if hasattr(X, 'shape') else len(X), 'hyperparameter_optimization'):
            return self._cpu_hyperparameter_optimization(model, X, y, param_grid, **kwargs)
        
        try:
            # Use GPU-accelerated training for each parameter combination
            from sklearn.model_selection import ParameterGrid
            from sklearn.model_selection import cross_val_score
            
            best_score = -np.inf
            best_params = None
            best_model = None
            
            for params in ParameterGrid(param_grid):
                # Create model with parameters
                model_instance = model.set_params(**params)
                
                # Use GPU acceleration for training
                if hasattr(model_instance, 'fit'):
                    # Train on GPU if possible
                    X_gpu = self.gpu_manager.prepare_data_for_gpu(X)
                    y_gpu = self.gpu_manager.prepare_data_for_gpu(y)
                    
                    # Train model
                    model_instance.fit(X_gpu, y_gpu)
                    
                    # Evaluate with cross-validation
                    scores = cross_val_score(model_instance, X, y, cv=5)
                    mean_score = scores.mean()
                    
                    if mean_score > best_score:
                        best_score = mean_score
                        best_params = params
                        best_model = model_instance
            
            return best_model, best_params, best_score
        except Exception as e:
            logger.warning(f"GPU hyperparameter optimization failed: {e}, falling back to CPU")
            return self._cpu_hyperparameter_optimization(model, X, y, param_grid, **kwargs)

    def _cpu_feature_engineering(self, df: pd.DataFrame, feature_funcs: List[Callable], **kwargs):
        """CPU-based feature engineering fallback."""
        results = []
        for func in feature_funcs:
            result = func(df, **kwargs)
            results.append(result)
        return pd.concat(results, axis=1)

    def _cpu_rolling_operations(self, df: pd.DataFrame, window_sizes: List[int], operation: str):
        """CPU-based rolling operations fallback."""
        results = []
        for window_size in window_sizes:
            result = df.copy()
            for col in df.select_dtypes(include=[np.number]).columns:
                if operation == 'mean':
                    result[f'{col}_rolling_{window_size}'] = df[col].rolling(window_size).mean()
                elif operation == 'std':
                    result[f'{col}_rolling_{window_size}_std'] = df[col].rolling(window_size).std()
                elif operation == 'min':
                    result[f'{col}_rolling_{window_size}_min'] = df[col].rolling(window_size).min()
                elif operation == 'max':
                    result[f'{col}_rolling_{window_size}_max'] = df[col].rolling(window_size).max()
            results.append(result)
        return pd.concat(results, axis=1)

    def _cpu_hyperparameter_optimization(self, model, X, y, param_grid, **kwargs):
        """CPU-based hyperparameter optimization fallback."""
        from sklearn.model_selection import GridSearchCV
        grid_search = GridSearchCV(model, param_grid, **kwargs)
        return grid_search.fit(X, y)

    def _get_memory_pressure(self) -> float:
        """Get current memory pressure."""
        try:
            import psutil
            memory = psutil.virtual_memory()
            return memory.percent / 100.0
        except:
            return 0.5

    def _estimate_data_size(self, args: tuple) -> int:
        """Estimate total data size in elements."""
        total_size = 0
        for arg in args:
            if hasattr(arg, 'size'):
                total_size += arg.size
            elif hasattr(arg, '__len__'):
                total_size += len(arg)
        return total_size

# GPU acceleration decorators
def gpu_accelerated(operation_type: str = 'matrix_operations'):
    """Decorator for GPU-accelerated operations."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            gpu_utils = GPUAccelerationUtils()
            if operation_type == 'matrix_operations':
                return gpu_utils.accelerate_matrix_operations(func, *args, **kwargs)
            elif operation_type == 'feature_engineering':
                return gpu_utils.accelerate_feature_engineering(*args, **kwargs)
            elif operation_type == 'rolling_operations':
                return gpu_utils.accelerate_rolling_operations(*args, **kwargs)
            elif operation_type == 'hyperparameter_optimization':
                return gpu_utils.accelerate_hyperparameter_optimization(*args, **kwargs)
            else:
                return func(*args, **kwargs)
        return wrapper
    return decorator

def adaptive_gpu_acceleration():
    """Decorator for adaptive GPU acceleration based on data characteristics."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            gpu_utils = GPUAccelerationUtils()
            
            # Estimate data size
            data_size = gpu_utils._estimate_data_size(args)
            
            # Determine if GPU should be used
            if gpu_utils.should_use_gpu(data_size, func.__name__):
                try:
                    return gpu_utils.accelerate_matrix_operations(func, *args, **kwargs)
                except Exception as e:
                    logger.warning(f"GPU acceleration failed: {e}, falling back to CPU")
                    return func(*args, **kwargs)
            else:
                return func(*args, **kwargs)
        return wrapper
    return decorator

# Global GPU acceleration utils instance
_global_gpu_utils: Optional[GPUAccelerationUtils] = None

def get_gpu_acceleration_utils() -> GPUAccelerationUtils:
    """Get the global GPU acceleration utils instance."""
    global _global_gpu_utils
    if _global_gpu_utils is None:
        _global_gpu_utils = GPUAccelerationUtils()
    return _global_gpu_utils