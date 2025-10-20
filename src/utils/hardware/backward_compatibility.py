"""
Backward Compatibility Layer for Hardware Optimizations.

This module ensures that existing code continues to work while providing access
to new hardware optimization features through a unified interface.
"""

import logging
import warnings
from typing import Any, Dict, List, Optional, Callable, Union
from functools import wraps
import numpy as np
import pandas as pd

# Import all hardware modules
from .vectorbt_gpu_accelerator import (
    get_vectorbt_gpu_accelerator, gpu_vectorbt_optimization,
    get_vectorbt_gpu_performance_metrics, VectorBTOperationType
)
from .enhanced_cpu_optimizer import (
    get_enhanced_cpu_optimizer, cpu_optimized_feature_correlation,
    get_enhanced_cpu_performance_metrics, CPUIntensity, PowerMode
)
from .enhanced_unified_memory_manager import (
    get_enhanced_unified_memory_manager, unified_memory_feature_processing,
    get_enhanced_unified_memory_stats, MemoryComponent, MemoryAccessPattern
)
from .adaptive_optimization_engine import (
    get_adaptive_optimization_engine, adaptive_feature_selection,
    get_adaptive_optimization_metrics, OptimizationStrategy, WorkloadCategory
)

logger = logging.getLogger(__name__)

class HardwareOptimizationManager:
    """Unified hardware optimization manager with backward compatibility."""
    
    def __init__(self):
        self.logger = logger.getChild('HardwareOptimizationManager')
        
        # Initialize all hardware components
        self.vectorbt_gpu = get_vectorbt_gpu_accelerator()
        self.enhanced_cpu = get_enhanced_cpu_optimizer()
        self.unified_memory = get_enhanced_unified_memory_manager()
        self.adaptive_engine = get_adaptive_optimization_engine()
        
        # Backward compatibility mappings
        self._setup_backward_compatibility()
        
        self.logger.info("🔧 Hardware Optimization Manager initialized with backward compatibility")
    
    def _setup_backward_compatibility(self):
        """Setup backward compatibility mappings."""
        # Map old function names to new implementations
        self.compatibility_mappings = {
            'gpu_vectorbt_optimization': self._gpu_vectorbt_optimization_compat,
            'cpu_optimized_feature_correlation': self._cpu_optimized_feature_correlation_compat,
            'unified_memory_feature_processing': self._unified_memory_feature_processing_compat,
            'adaptive_feature_selection': self._adaptive_feature_selection_compat
        }
    
    def _gpu_vectorbt_optimization_compat(self, price_data: np.ndarray, features: Dict[str, Any]) -> Dict[str, Any]:
        """Backward compatible GPU VectorBT optimization."""
        try:
            # Use new GPU accelerator
            return gpu_vectorbt_optimization(price_data, features)
        except Exception as e:
            self.logger.warning(f"GPU optimization failed, falling back to CPU: {e}")
            # Fallback to CPU implementation
            return self._cpu_vectorbt_fallback(price_data, features)
    
    def _cpu_optimized_feature_correlation_compat(self, data: np.ndarray) -> np.ndarray:
        """Backward compatible CPU-optimized feature correlation."""
        try:
            # Use enhanced CPU optimizer
            return cpu_optimized_feature_correlation(data)
        except Exception as e:
            self.logger.warning(f"Enhanced CPU optimization failed, using basic implementation: {e}")
            # Fallback to basic NumPy implementation
            return np.corrcoef(data.T)
    
    def _unified_memory_feature_processing_compat(self, data: Any, operation_type: str = 'feature_selection', 
                                                component: str = 'gpu') -> Any:
        """Backward compatible unified memory feature processing."""
        try:
            # Use enhanced unified memory manager
            return unified_memory_feature_processing(data, operation_type, component)
        except Exception as e:
            self.logger.warning(f"Unified memory optimization failed, using basic processing: {e}")
            # Fallback to basic processing
            return data
    
    def _adaptive_feature_selection_compat(self, data: Any, learn_from_execution: bool = True) -> Any:
        """Backward compatible adaptive feature selection."""
        try:
            # Use adaptive optimization engine
            return adaptive_feature_selection(data, learn_from_execution)
        except Exception as e:
            self.logger.warning(f"Adaptive optimization failed, using basic processing: {e}")
            # Fallback to basic processing
            return data
    
    def _cpu_vectorbt_fallback(self, price_data: np.ndarray, features: Dict[str, Any]) -> Dict[str, Any]:
        """CPU fallback for VectorBT operations."""
        # Basic portfolio analysis
        if 'weights' in features:
            weights = features['weights']
        else:
            weights = np.ones(price_data.shape[1]) / price_data.shape[1]
        
        portfolio_returns = np.sum(price_data * weights, axis=1)
        
        return {
            'mean_return': np.mean(portfolio_returns),
            'volatility': np.std(portfolio_returns),
            'sharpe_ratio': np.mean(portfolio_returns) / np.std(portfolio_returns) if np.std(portfolio_returns) > 0 else 0,
            'var_95': np.percentile(portfolio_returns, 5),
            'portfolio_returns': portfolio_returns
        }
    
    def get_comprehensive_metrics(self) -> Dict[str, Any]:
        """Get comprehensive hardware optimization metrics."""
        return {
            'vectorbt_gpu_metrics': get_vectorbt_gpu_performance_metrics(),
            'enhanced_cpu_metrics': get_enhanced_cpu_performance_metrics(),
            'unified_memory_metrics': get_enhanced_unified_memory_stats(),
            'adaptive_optimization_metrics': get_adaptive_optimization_metrics(),
            'backward_compatibility': {
                'mappings_available': len(self.compatibility_mappings),
                'fallback_implementations': True
            }
        }

# Global instance
_hardware_manager: Optional[HardwareOptimizationManager] = None

def get_hardware_optimization_manager() -> HardwareOptimizationManager:
    """Get or create the global hardware optimization manager."""
    global _hardware_manager
    
    if _hardware_manager is None:
        _hardware_manager = HardwareOptimizationManager()
    
    return _hardware_manager

# Backward compatibility functions
def gpu_vectorbt_optimization(price_data: np.ndarray, features: Dict[str, Any]) -> Dict[str, Any]:
    """
    GPU-accelerated VectorBT optimization with backward compatibility.
    
    This function provides GPU acceleration for VectorBT operations while maintaining
    backward compatibility with existing code.
    
    Args:
        price_data: Price data array
        features: Dictionary of features and parameters
        
    Returns:
        Dictionary containing optimization results
    """
    manager = get_hardware_optimization_manager()
    return manager._gpu_vectorbt_optimization_compat(price_data, features)

def cpu_optimized_feature_correlation(data: np.ndarray) -> np.ndarray:
    """
    CPU-optimized feature correlation with backward compatibility.
    
    This function provides enhanced CPU optimization for feature correlation
    while maintaining backward compatibility.
    
    Args:
        data: Input data array
        
    Returns:
        Correlation matrix
    """
    manager = get_hardware_optimization_manager()
    return manager._cpu_optimized_feature_correlation_compat(data)

def unified_memory_feature_processing(data: Any, operation_type: str = 'feature_selection', 
                                    component: str = 'gpu') -> Any:
    """
    Unified memory feature processing with backward compatibility.
    
    This function provides unified memory optimization for feature processing
    while maintaining backward compatibility.
    
    Args:
        data: Input data
        operation_type: Type of operation
        component: Target component (cpu, gpu, etc.)
        
    Returns:
        Processed data
    """
    manager = get_hardware_optimization_manager()
    return manager._unified_memory_feature_processing_compat(data, operation_type, component)

def adaptive_feature_selection(data: Any, learn_from_execution: bool = True) -> Any:
    """
    Adaptive feature selection with backward compatibility.
    
    This function provides adaptive optimization for feature selection
    while maintaining backward compatibility.
    
    Args:
        data: Input data
        learn_from_execution: Whether to learn from execution patterns
        
    Returns:
        Selected features
    """
    manager = get_hardware_optimization_manager()
    return manager._adaptive_feature_selection_compat(data, learn_from_execution)

# Decorator functions for easy integration
def gpu_accelerated(operation_type: str = "matrix_multiplication"):
    """
    Decorator for GPU acceleration with backward compatibility.
    
    This decorator provides GPU acceleration for functions while maintaining
    backward compatibility with existing code.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                # Try GPU acceleration
                if operation_type == "matrix_multiplication" and len(args) >= 2:
                    return gpu_vectorbt_optimization(args[0], {"operation": "matrix_multiply"})
                else:
                    # Fallback to original function
                    return func(*args, **kwargs)
            except Exception as e:
                logger.warning(f"GPU acceleration failed: {e}, using CPU fallback")
                return func(*args, **kwargs)
        return wrapper
    return decorator

def optimize_cpu_execution(workload_type: str = "cpu_intensive"):
    """
    Decorator for CPU optimization with backward compatibility.
    
    This decorator provides CPU optimization for functions while maintaining
    backward compatibility.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                # Use enhanced CPU optimization
                if len(args) >= 1 and isinstance(args[0], np.ndarray):
                    return cpu_optimized_feature_correlation(args[0])
                else:
                    return func(*args, **kwargs)
            except Exception as e:
                logger.warning(f"CPU optimization failed: {e}, using basic implementation")
                return func(*args, **kwargs)
        return wrapper
    return decorator

def unified_memory_optimized(operation_type: str = 'general', component: str = 'gpu'):
    """
    Decorator for unified memory optimization with backward compatibility.
    
    This decorator provides unified memory optimization for functions while
    maintaining backward compatibility.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                # Use unified memory optimization
                if len(args) >= 1:
                    optimized_data = unified_memory_feature_processing(args[0], operation_type, component)
                    return func(optimized_data, *args[1:], **kwargs)
                else:
                    return func(*args, **kwargs)
            except Exception as e:
                logger.warning(f"Unified memory optimization failed: {e}, using basic processing")
                return func(*args, **kwargs)
        return wrapper
    return decorator

def adaptive_optimization(learn_from_execution: bool = True):
    """
    Decorator for adaptive optimization with backward compatibility.
    
    This decorator provides adaptive optimization for functions while
    maintaining backward compatibility.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                # Use adaptive optimization
                if len(args) >= 1:
                    optimized_data = adaptive_feature_selection(args[0], learn_from_execution)
                    return func(optimized_data, *args[1:], **kwargs)
                else:
                    return func(*args, **kwargs)
            except Exception as e:
                logger.warning(f"Adaptive optimization failed: {e}, using basic processing")
                return func(*args, **kwargs)
        return wrapper
    return decorator

def smart_cache(ttl: int = 3600, max_size: int = 1000, compression: bool = True):
    """
    Decorator for smart caching with backward compatibility.
    
    This decorator provides intelligent caching for functions while
    maintaining backward compatibility.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Simple caching implementation
            cache_key = f"{func.__name__}_{hash(str(args))}_{hash(str(kwargs))}"
            
            # Check cache (simplified)
            if hasattr(wrapper, '_cache'):
                if cache_key in wrapper._cache:
                    return wrapper._cache[cache_key]
            else:
                wrapper._cache = {}
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Store in cache
            wrapper._cache[cache_key] = result
            
            # Simple cache size management
            if len(wrapper._cache) > max_size:
                # Remove oldest entry (simplified)
                oldest_key = next(iter(wrapper._cache))
                del wrapper._cache[oldest_key]
            
            return result
        return wrapper
    return decorator

def performance_tracked(metrics: List[str] = None):
    """
    Decorator for performance tracking with backward compatibility.
    
    This decorator provides performance tracking for functions while
    maintaining backward compatibility.
    """
    if metrics is None:
        metrics = ['execution_time', 'memory_usage', 'cpu_utilization']
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            import time
            import psutil
            
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            try:
                result = func(*args, **kwargs)
                
                end_time = time.time()
                end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
                
                # Track metrics
                execution_time = end_time - start_time
                memory_usage = end_memory - start_memory
                cpu_utilization = psutil.cpu_percent()
                
                # Log performance metrics
                logger.info(f"Performance metrics for {func.__name__}:")
                logger.info(f"  Execution time: {execution_time:.3f}s")
                logger.info(f"  Memory usage: {memory_usage:.1f}MB")
                logger.info(f"  CPU utilization: {cpu_utilization:.1f}%")
                
                return result
                
            except Exception as e:
                logger.error(f"Performance tracking error for {func.__name__}: {e}")
                raise
        
        return wrapper
    return decorator

def comprehensive_memory_optimization(int64_to_int32: bool = True, 
                                    float64_to_float32: bool = True,
                                    object_to_category: bool = True,
                                    compression_ratio: float = 0.7):
    """
    Decorator for comprehensive memory optimization with backward compatibility.
    
    This decorator provides comprehensive memory optimization for functions while
    maintaining backward compatibility.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Optimize inputs
            optimized_args = []
            for arg in args:
                if isinstance(arg, pd.DataFrame):
                    optimized_df = arg.copy()
                    
                    if int64_to_int32:
                        for col in optimized_df.select_dtypes(include=['int64']):
                            optimized_df[col] = optimized_df[col].astype('int32')
                    
                    if float64_to_float32:
                        for col in optimized_df.select_dtypes(include=['float64']):
                            optimized_df[col] = optimized_df[col].astype('float32')
                    
                    if object_to_category:
                        for col in optimized_df.select_dtypes(include=['object']):
                            if optimized_df[col].nunique() / len(optimized_df) < 0.5:
                                optimized_df[col] = optimized_df[col].astype('category')
                    
                    optimized_args.append(optimized_df)
                elif isinstance(arg, np.ndarray):
                    if int64_to_int32 and arg.dtype == np.int64:
                        optimized_args.append(arg.astype(np.int32))
                    elif float64_to_float32 and arg.dtype == np.float64:
                        optimized_args.append(arg.astype(np.float32))
                    else:
                        optimized_args.append(arg)
                else:
                    optimized_args.append(arg)
            
            # Execute function
            result = func(*optimized_args, **kwargs)
            
            # Optimize output
            if isinstance(result, pd.DataFrame):
                optimized_result = result.copy()
                
                if int64_to_int32:
                    for col in optimized_result.select_dtypes(include=['int64']):
                        optimized_result[col] = optimized_result[col].astype('int32')
                
                if float64_to_float32:
                    for col in optimized_result.select_dtypes(include=['float64']):
                        optimized_result[col] = optimized_result[col].astype('float32')
                
                if object_to_category:
                    for col in optimized_result.select_dtypes(include=['object']):
                        if optimized_result[col].nunique() / len(optimized_result) < 0.5:
                            optimized_result[col] = optimized_result[col].astype('category')
                
                return optimized_result
            elif isinstance(result, np.ndarray):
                if int64_to_int32 and result.dtype == np.int64:
                    return result.astype(np.int32)
                elif float64_to_float32 and result.dtype == np.float64:
                    return result.astype(np.float32)
            
            return result
        
        return wrapper
    return decorator

# Utility functions
def get_hardware_optimization_status() -> Dict[str, Any]:
    """Get comprehensive hardware optimization status."""
    manager = get_hardware_optimization_manager()
    return manager.get_comprehensive_metrics()

def clear_optimization_caches():
    """Clear all optimization caches."""
    try:
        # Clear GPU caches
        vectorbt_gpu = get_vectorbt_gpu_accelerator()
        vectorbt_gpu.clear_memory()
        
        # Clear memory caches
        unified_memory = get_enhanced_unified_memory_manager()
        unified_memory.cleanup_all()
        
        logger.info("🧹 All optimization caches cleared")
    except Exception as e:
        logger.warning(f"Failed to clear caches: {e}")

def initialize_optimization_system():
    """Initialize the complete optimization system."""
    try:
        manager = get_hardware_optimization_manager()
        logger.info("🚀 Hardware optimization system initialized successfully")
        return manager
    except Exception as e:
        logger.error(f"Failed to initialize optimization system: {e}")
        return None

# Deprecation warnings for old function names
def _deprecated_function_warning(old_name: str, new_name: str):
    """Issue deprecation warning for old function names."""
    warnings.warn(
        f"{old_name} is deprecated. Use {new_name} instead.",
        DeprecationWarning,
        stacklevel=3
    )

# Export all functions for backward compatibility
__all__ = [
    # Core functions
    'gpu_vectorbt_optimization',
    'cpu_optimized_feature_correlation', 
    'unified_memory_feature_processing',
    'adaptive_feature_selection',
    
    # Decorators
    'gpu_accelerated',
    'optimize_cpu_execution',
    'unified_memory_optimized',
    'adaptive_optimization',
    'smart_cache',
    'performance_tracked',
    'comprehensive_memory_optimization',
    
    # Utility functions
    'get_hardware_optimization_status',
    'clear_optimization_caches',
    'initialize_optimization_system',
    
    # Manager
    'get_hardware_optimization_manager'
]