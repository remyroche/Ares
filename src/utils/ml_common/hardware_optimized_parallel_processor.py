"""
Hardware-Optimized Parallel Processor for ML Common

This module provides hardware-aware parallel processing specifically designed
for ML operations with full integration of the hardware optimization system.
"""

import logging
import time
from typing import Any, Callable, Dict, List, Optional, Union
from functools import wraps
import pandas as pd
import numpy as np

# Hardware optimization imports
from ..hardware.integrated_hardware_manager import (
    get_integrated_hardware_manager, IntegratedHardwareManager, WorkloadType
)
from ..hardware.unified_hardware_manager import (
    get_unified_hardware_manager, UnifiedHardwareManager, OptimizationLevel
)
from ..hardware.adaptive_optimization_engine import (
    get_adaptive_optimization_engine, AdaptiveOptimizationEngine, OptimizationTarget
)
from ..hardware.advanced_memory_manager import (
    get_advanced_memory_manager, AdvancedMemoryManager
)
from ..hardware.advanced_memory_optimizer import (
    MemoryStrategy
)
from ..hardware.enhanced_gpu_manager import (
    get_enhanced_gpu_manager, EnhancedM1GPUManager, GPUOperationType
)

logger = logging.getLogger(__name__)

class HardwareOptimizedMLProcessor:
    """
    Hardware-optimized processor for ML operations with adaptive optimization.
    """

    def __init__(self, enable_hardware_optimization: bool = True):
        """Initialize the hardware-optimized ML processor."""
        self.enable_hardware_optimization = enable_hardware_optimization
        
        if self.enable_hardware_optimization:
            self.hardware_manager = get_integrated_hardware_manager()
            self.unified_manager = get_unified_hardware_manager()
            self.adaptive_engine = get_adaptive_optimization_engine()
            self.memory_manager = get_advanced_memory_manager()
            self.gpu_manager = get_enhanced_gpu_manager()
        else:
            self.hardware_manager = None
            self.unified_manager = None
            self.adaptive_engine = None
            self.memory_manager = None
            self.gpu_manager = None

    def process_ml_training(self, model, X, y, **kwargs):
        """Process ML training with hardware optimization."""
        if not self.enable_hardware_optimization:
            return model.fit(X, y, **kwargs)
        
        # Get optimal strategy for training
        strategy = self.adaptive_engine.get_optimal_strategy(
            'training',
            {
                'memory_pressure': self._get_memory_pressure(),
                'data_size': X.shape[0] if hasattr(X, 'shape') else len(X),
                'feature_count': X.shape[1] if hasattr(X, 'shape') else 0
            }
        )
        
        # Apply hardware optimization
        with self.unified_manager.optimization_context(WorkloadType.ML_TRAINING, OptimizationLevel.AGGRESSIVE):
            # Use memory optimization
            with self.memory_manager.memory_context(MemoryStrategy.ADAPTIVE):
                # Optimize data types for memory efficiency
                X_optimized = self._optimize_data_types(X)
                y_optimized = self._optimize_data_types(y)
                
                # Train model
                return model.fit(X_optimized, y_optimized, **kwargs)

    def process_feature_engineering(self, df: pd.DataFrame, feature_funcs: List[Callable], **kwargs):
        """Process feature engineering with hardware optimization."""
        if not self.enable_hardware_optimization:
            return self._standard_feature_engineering(df, feature_funcs, **kwargs)
        
        # Get optimal strategy
        strategy = self.adaptive_engine.get_optimal_strategy(
            'feature_engineering',
            {
                'memory_pressure': self._get_memory_pressure(),
                'data_size': len(df),
                'feature_count': len(df.columns)
            }
        )
        
        # Apply hardware optimization
        with self.unified_manager.optimization_context(WorkloadType.FEATURE_ENGINEERING, OptimizationLevel.AGGRESSIVE):
            # Use memory optimization
            with self.memory_manager.memory_context(MemoryStrategy.ADAPTIVE):
                # Optimize DataFrame for memory efficiency
                df_optimized = self._optimize_dataframe(df)
                
                # Process features with GPU acceleration if suitable
                if strategy.get('use_gpu', False) and self.gpu_manager.is_gpu_available():
                    return self._gpu_feature_engineering(df_optimized, feature_funcs, **kwargs)
                else:
                    return self._cpu_feature_engineering(df_optimized, feature_funcs, **kwargs)

    def process_hyperparameter_optimization(self, model, X, y, param_grid, **kwargs):
        """Process hyperparameter optimization with hardware optimization."""
        if not self.enable_hardware_optimization:
            return self._standard_hpo(model, X, y, param_grid, **kwargs)
        
        # Get optimal strategy
        strategy = self.adaptive_engine.get_optimal_strategy(
            'hyperparameter_optimization',
            {
                'memory_pressure': self._get_memory_pressure(),
                'data_size': X.shape[0] if hasattr(X, 'shape') else len(X),
                'param_count': len(param_grid)
            }
        )
        
        # Apply hardware optimization
        with self.unified_manager.optimization_context(WorkloadType.ML_TRAINING, OptimizationLevel.AGGRESSIVE):
            # Use memory optimization
            with self.memory_manager.memory_context(MemoryStrategy.ADAPTIVE):
                # Optimize data for HPO
                X_optimized = self._optimize_data_types(X)
                y_optimized = self._optimize_data_types(y)
                
                # Use parallel processing for HPO
                return self._parallel_hpo(model, X_optimized, y_optimized, param_grid, strategy, **kwargs)

    def _get_memory_pressure(self) -> float:
        """Get current memory pressure."""
        try:
            import psutil
            memory = psutil.virtual_memory()
            return memory.percent / 100.0
        except:
            return 0.5

    def _optimize_data_types(self, data):
        """Optimize data types for memory efficiency."""
        if isinstance(data, pd.DataFrame):
            return self._optimize_dataframe(data)
        elif isinstance(data, np.ndarray):
            return self._optimize_numpy_array(data)
        else:
            return data

    def _optimize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame for memory efficiency."""
        try:
            # Use the memory manager's optimization
            optimized_df, _ = self.memory_manager.optimize_dataframe(df)
            return optimized_df
        except Exception as e:
            logger.warning(f"DataFrame optimization failed: {e}")
            return df

    def _optimize_numpy_array(self, arr: np.ndarray) -> np.ndarray:
        """Optimize NumPy array for memory efficiency."""
        try:
            # Use the memory manager's optimization
            optimized_arr, _ = self.memory_manager.optimize_numpy_array(arr)
            return optimized_arr
        except Exception as e:
            logger.warning(f"NumPy array optimization failed: {e}")
            return arr

    def _standard_feature_engineering(self, df: pd.DataFrame, feature_funcs: List[Callable], **kwargs):
        """Standard feature engineering without hardware optimization."""
        results = []
        for func in feature_funcs:
            result = func(df, **kwargs)
            results.append(result)
        return pd.concat(results, axis=1)

    def _cpu_feature_engineering(self, df: pd.DataFrame, feature_funcs: List[Callable], **kwargs):
        """CPU-based feature engineering with memory optimization."""
        results = []
        for func in feature_funcs:
            # Apply memory optimization for each function
            with self.memory_manager.memory_context(MemoryStrategy.ADAPTIVE):
                result = func(df, **kwargs)
                results.append(result)
        return pd.concat(results, axis=1)

    def _gpu_feature_engineering(self, df: pd.DataFrame, feature_funcs: List[Callable], **kwargs):
        """GPU-accelerated feature engineering."""
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

    def _standard_hpo(self, model, X, y, param_grid, **kwargs):
        """Standard hyperparameter optimization."""
        from sklearn.model_selection import GridSearchCV
        grid_search = GridSearchCV(model, param_grid, **kwargs)
        return grid_search.fit(X, y)

    def _parallel_hpo(self, model, X, y, param_grid, strategy, **kwargs):
        """Parallel hyperparameter optimization with hardware optimization."""
        from sklearn.model_selection import GridSearchCV
        from concurrent.futures import ThreadPoolExecutor
        
        # Use strategy-recommended number of threads
        n_jobs = strategy.get('num_threads', 4)
        
        grid_search = GridSearchCV(model, param_grid, n_jobs=n_jobs, **kwargs)
        return grid_search.fit(X, y)

# Hardware-aware decorators for ML operations
def ml_training_optimized(enable_gpu: bool = True):
    """Decorator for hardware-optimized ML training."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            processor = HardwareOptimizedMLProcessor()
            return processor.process_ml_training(func, *args, **kwargs)
        return wrapper
    return decorator

def feature_engineering_optimized(enable_gpu: bool = True):
    """Decorator for hardware-optimized feature engineering."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            processor = HardwareOptimizedMLProcessor()
            # Extract DataFrame from args
            df = None
            for arg in args:
                if isinstance(arg, pd.DataFrame):
                    df = arg
                    break
            if df is None:
                return func(*args, **kwargs)
            
            return processor.process_feature_engineering(df, [func], **kwargs)
        return wrapper
    return decorator

def hpo_optimized(enable_gpu: bool = True):
    """Decorator for hardware-optimized hyperparameter optimization."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            processor = HardwareOptimizedMLProcessor()
            return processor.process_hyperparameter_optimization(func, *args, **kwargs)
        return wrapper
    return decorator

# Global processor instance
_global_ml_processor: Optional[HardwareOptimizedMLProcessor] = None

def get_hardware_optimized_ml_processor() -> HardwareOptimizedMLProcessor:
    """Get the global hardware-optimized ML processor instance."""
    global _global_ml_processor
    if _global_ml_processor is None:
        _global_ml_processor = HardwareOptimizedMLProcessor()
    return _global_ml_processor