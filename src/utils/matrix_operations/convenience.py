"""
Convenience Functions - Unified Implementation

This module provides convenient wrapper functions for common matrix operations
with backwards compatibility and easy access to unified functionality.
"""

from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import logging

# Conditional imports for optional dependencies
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None

try:
    from scipy import sparse
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    sparse = None

# Import the unified modules
from .unified_operations import (
    get_unified_matrix_operations,
    safe_matrix_multiply as _safe_matrix_multiply,
    safe_correlation_matrix as _safe_correlation_matrix,
    safe_matrix_inverse as _safe_matrix_inverse
)

# Import VectorBT optimizations
try:
    from .vectorbt_optimizations import (
        get_vectorbt_optimized_operations,
        vectorbt_matrix_multiply,
        vectorbt_correlation_matrix,
        vectorbt_trading_indicators,
        vectorbt_rolling_features,
        vectorbt_batch_processing
    )
    VECTORBT_OPTIMIZATIONS_AVAILABLE = True
except ImportError:
    VECTORBT_OPTIMIZATIONS_AVAILABLE = False
    vectorbt_matrix_multiply = None
    vectorbt_correlation_matrix = None
    vectorbt_trading_indicators = None
    vectorbt_rolling_features = None
    vectorbt_batch_processing = None

from .vectorized_core import (
    get_vectorized_processing_core,
    optimize_dataframe as _optimize_dataframe,
    vectorized_rolling_features as _vectorized_rolling_features,
    matrix_correlation_analysis as _matrix_correlation_analysis,
    OptimizedPipelineExecutor,
    PipelineExecutionMode,
    PipelineExecutionResult
)

from .batch_operations import (
    get_batch_matrix_processor,
    batch_matrix_multiply as _batch_matrix_multiply,
    batch_feature_transformation as _batch_feature_transformation,
    batch_correlation_analysis as _batch_correlation_analysis
)

from .enhanced_operations import (
    OperationComplexity,
    get_enhanced_matrix_operations,
    gpu_matrix_multiply as _gpu_matrix_multiply,
    correlation_matrix_gpu as _correlation_matrix_gpu,
    eigendecomposition_gpu as _eigendecomposition_gpu,
    svd_gpu as _svd_gpu,
    optimize_batch_size as _optimize_batch_size,
    record_batch_performance as _record_batch_performance,
    get_batch_optimization_stats as _get_batch_optimization_stats
)

# Matrix operations convenience functions
def safe_matrix_multiply(A: 'np.ndarray', B: 'np.ndarray') -> 'np.ndarray':
    """Safe matrix multiplication with VectorBT optimization and validation."""
    # Try VectorBT optimization first if available
    if VECTORBT_OPTIMIZATIONS_AVAILABLE and vectorbt_matrix_multiply:
        try:
            return vectorbt_matrix_multiply(A, B)
        except Exception as e:
            logger.warning(f"⚠️ VectorBT matrix multiplication failed: {e}, falling back to standard method")
    
    return _safe_matrix_multiply(A, B)

def safe_correlation_matrix(data: Union['np.ndarray', 'pd.DataFrame']) -> 'np.ndarray':
    """Safe correlation matrix computation with VectorBT optimization."""
    # Try VectorBT optimization first if available
    if VECTORBT_OPTIMIZATIONS_AVAILABLE and vectorbt_correlation_matrix:
        try:
            return vectorbt_correlation_matrix(data)
        except Exception as e:
            logger.warning(f"⚠️ VectorBT correlation matrix failed: {e}, falling back to standard method")
    
    return _safe_correlation_matrix(data)

def safe_matrix_inverse(matrix: 'np.ndarray') -> 'np.ndarray':
    """Safe matrix inversion."""
    return _safe_matrix_inverse(matrix)

def gpu_matrix_multiply(a: 'np.ndarray', b: 'np.ndarray') -> 'np.ndarray':
    """GPU-accelerated matrix multiplication."""
    return _gpu_matrix_multiply(a, b)

def correlation_matrix_gpu(data: Union['pd.DataFrame', 'np.ndarray']) -> 'np.ndarray':
    """GPU-accelerated correlation matrix."""
    return _correlation_matrix_gpu(data)

def eigendecomposition_gpu(matrix: 'np.ndarray') -> Tuple['np.ndarray', 'np.ndarray']:
    """GPU-accelerated eigendecomposition."""
    return _eigendecomposition_gpu(matrix)

def svd_gpu(matrix: 'np.ndarray', k: Optional[int] = None) -> Tuple['np.ndarray', 'np.ndarray', 'np.ndarray']:
    """GPU-accelerated SVD."""
    return _svd_gpu(matrix, k)

# Vectorized operations convenience functions
def optimize_dataframe(df: 'pd.DataFrame') -> 'pd.DataFrame':
    """Optimize DataFrame for processing."""
    return _optimize_dataframe(df)

def vectorized_rolling_features(data: 'pd.DataFrame',
                              windows: List[int] = None,
                              features: List[str] = None) -> 'pd.DataFrame':
    """Create vectorized rolling features with VectorBT optimization."""
    # Try VectorBT optimization first if available
    if VECTORBT_OPTIMIZATIONS_AVAILABLE and vectorbt_rolling_features:
        try:
            return vectorbt_rolling_features(data, windows, features)
        except Exception as e:
            logger.warning(f"⚠️ VectorBT rolling features failed: {e}, falling back to standard method")
    
    return _vectorized_rolling_features(data, windows, features)

def matrix_correlation_analysis(data: 'pd.DataFrame',
                              method: str = 'pearson') -> Tuple['np.ndarray', 'pd.DataFrame']:
    """Compute matrix-based correlation analysis."""
    return _matrix_correlation_analysis(data, method)

def parallel_feature_engineering(data: 'pd.DataFrame',
                               feature_functions: List[Callable[['pd.DataFrame'], 'pd.Series']],
                               max_workers: Optional[int] = None) -> 'pd.DataFrame':
    """Parallel feature engineering."""
    core = get_vectorized_processing_core()
    return core.parallel_feature_engineering(data, feature_functions, max_workers)

# Batch operations convenience functions
def batch_matrix_multiply(matrices_a: List['np.ndarray'], matrices_b: List['np.ndarray']) -> List['np.ndarray']:
    """Convenience function for batch matrix multiplication with VectorBT optimization."""
    # Try VectorBT optimization first if available
    if VECTORBT_OPTIMIZATIONS_AVAILABLE and vectorbt_batch_processing:
        try:
            return vectorbt_batch_processing(matrices_a, 'batch_matrix_multiply', matrices_b=matrices_b)
        except Exception as e:
            logger.warning(f"⚠️ VectorBT batch processing failed: {e}, falling back to standard method")
    
    return _batch_matrix_multiply(matrices_a, matrices_b)

def batch_feature_transformation(data: Union['np.ndarray', 'pd.DataFrame'], 
                               transformations: List[Dict[str, Any]]) -> Union['np.ndarray', 'pd.DataFrame']:
    """Convenience function for batch feature transformation."""
    return _batch_feature_transformation(data, transformations)

def batch_correlation_analysis(data: Union['np.ndarray', 'pd.DataFrame'], 
                             method: str = 'pearson') -> Tuple['np.ndarray', 'np.ndarray']:
    """Convenience function for batch correlation analysis."""
    return _batch_correlation_analysis(data, method)

# Sparse matrix operations convenience functions
def sparse_matrix_multiply(a: Union['sparse.spmatrix', 'np.ndarray'],
                          b: Union['sparse.spmatrix', 'np.ndarray'],
                          format: str = 'csr') -> 'sparse.spmatrix':
    """Sparse matrix multiplication with GPU acceleration."""
    ops = get_enhanced_matrix_operations()
    return ops.sparse_matrix_multiply(a, b, format)

def sparse_svd(matrix: 'sparse.spmatrix', k: Optional[int] = None,
              solver: str = 'arpack') -> Tuple['np.ndarray', 'np.ndarray', 'np.ndarray']:
    """Sparse SVD decomposition."""
    ops = get_enhanced_matrix_operations()
    return ops.sparse_svd(matrix, k, solver)

def sparse_eigen(matrix: 'sparse.spmatrix', k: int = 10,
                which: str = 'LM') -> Tuple['np.ndarray', 'np.ndarray']:
    """Sparse eigenvalue decomposition."""
    ops = get_enhanced_matrix_operations()
    return ops.sparse_eigen(matrix, k, which)

def create_sparse_matrix(matrix: 'np.ndarray',
                        sparsity_threshold: float = 0.1) -> Union['sparse.spmatrix', 'np.ndarray']:
    """Create sparse matrix from dense matrix if beneficial."""
    ops = get_enhanced_matrix_operations()
    return ops.create_sparse_from_dense(matrix, sparsity_threshold)

def sparse_solve(a: 'sparse.spmatrix', b: 'np.ndarray',
                solver: str = 'spsolve') -> 'np.ndarray':
    """Solve sparse linear system."""
    ops = get_enhanced_matrix_operations()
    return ops.sparse_solve_linear(a, b, solver)

# Pipeline operations convenience functions
def create_ml_pipeline(stages_config: List[Dict[str, Any]]) -> 'OptimizedPipelineExecutor':
    """Create an optimized ML processing pipeline."""
    core = get_vectorized_processing_core()
    return core.create_optimized_pipeline(stages_config)

def execute_ml_pipeline(data: 'pd.DataFrame',
                       pipeline_config: List[Dict[str, Any]],
                       execution_mode: 'PipelineExecutionMode' = None) -> 'PipelineExecutionResult':
    """Execute a complete ML processing pipeline with optimization."""
    core = get_vectorized_processing_core()
    if execution_mode is None:
        from .vectorized_core import PipelineExecutionMode
        execution_mode = PipelineExecutionMode.HYBRID
    return core.execute_ml_pipeline(data, pipeline_config, execution_mode)

def optimize_pipeline_config(pipeline_config: List[Dict[str, Any]],
                           data_sample: 'pd.DataFrame') -> Dict[str, Any]:
    """Analyze and optimize pipeline execution strategy."""
    core = get_vectorized_processing_core()
    return core.optimize_pipeline_execution(pipeline_config, data_sample)

def get_pipeline_executor() -> 'OptimizedPipelineExecutor':
    """Get the global pipeline executor instance."""
    core = get_vectorized_processing_core()
    return core.pipeline_executor

# Optimization utilities convenience functions
def optimize_batch_size(operation_name: str, data_shape: Tuple[int, ...],
                       complexity: 'OperationComplexity' = None,
                       available_memory_mb: Optional[float] = None) -> int:
    """Optimize batch size for matrix operations."""
    if complexity is None:
        from .enhanced_operations import OperationComplexity
        complexity = OperationComplexity.MEDIUM
    return _optimize_batch_size(operation_name, data_shape, complexity, available_memory_mb)

def record_batch_performance(operation_name: str, batch_size: int, execution_time: float,
                           memory_usage: float, data_processed: int):
    """Record performance metrics for batch optimization learning."""
    return _record_batch_performance(operation_name, batch_size, execution_time, memory_usage, data_processed)

def get_batch_optimization_stats() -> Dict[str, Any]:
    """Get batch optimization statistics."""
    return _get_batch_optimization_stats()

# Trading indicators convenience functions
def compute_trading_indicators(data: 'pd.DataFrame', 
                              config: Optional[Dict[str, Any]] = None,
                              use_hardware_optimization: bool = True) -> 'pd.DataFrame':
    """Compute comprehensive trading indicators with VectorBT and hardware optimization."""
    # Try VectorBT optimization first if available
    if VECTORBT_OPTIMIZATIONS_AVAILABLE and vectorbt_trading_indicators:
        try:
            return vectorbt_trading_indicators(data, config)
        except Exception as e:
            logger.warning(f"⚠️ VectorBT trading indicators failed: {e}, falling back to standard method")
    
    core = get_vectorized_processing_core()
    return core.compute_trading_indicators(data, config)

def compute_moving_averages(data: 'pd.DataFrame', 
                           sma_periods: List[int] = None,
                           ema_periods: List[int] = None) -> 'pd.DataFrame':
    """Compute moving averages with custom periods."""
    if sma_periods is None:
        sma_periods = [9, 21, 50, 200]
    if ema_periods is None:
        ema_periods = [12, 26, 50]
    
    config = {
        'sma_periods': sma_periods,
        'ema_periods': ema_periods
    }
    
    core = get_vectorized_processing_core()
    return core._compute_moving_averages(data, config)

def compute_momentum_indicators(data: 'pd.DataFrame',
                               rsi_period: int = 14,
                               macd_fast: int = 12,
                               macd_slow: int = 26,
                               macd_signal: int = 9) -> 'pd.DataFrame':
    """Compute momentum indicators with custom parameters."""
    config = {
        'rsi_period': rsi_period,
        'macd_fast': macd_fast,
        'macd_slow': macd_slow,
        'macd_signal': macd_signal
    }
    
    core = get_vectorized_processing_core()
    return core._compute_momentum_indicators(data, config)

def compute_volatility_indicators(data: 'pd.DataFrame',
                                 bb_period: int = 20,
                                 bb_std: float = 2.0,
                                 atr_period: int = 14) -> 'pd.DataFrame':
    """Compute volatility indicators with custom parameters."""
    config = {
        'bb_period': bb_period,
        'bb_std': bb_std,
        'atr_period': atr_period
    }
    
    core = get_vectorized_processing_core()
    return core._compute_volatility_indicators(data, config)

def compute_volume_indicators(data: 'pd.DataFrame',
                             volume_sma_period: int = 20,
                             obv_smooth: int = 10) -> 'pd.DataFrame':
    """Compute volume-based indicators with custom parameters."""
    config = {
        'volume_sma_period': volume_sma_period,
        'obv_smooth': obv_smooth
    }
    
    core = get_vectorized_processing_core()
    return core._compute_volume_indicators(data, config)

def compute_trend_indicators(data: 'pd.DataFrame',
                            adx_period: int = 14) -> 'pd.DataFrame':
    """Compute trend indicators with custom parameters."""
    config = {
        'adx_period': adx_period
    }
    
    core = get_vectorized_processing_core()
    return core._compute_trend_indicators(data, config)

def compute_oscillator_indicators(data: 'pd.DataFrame',
                                 stoch_k: int = 14,
                                 stoch_d: int = 3,
                                 williams_period: int = 14,
                                 cci_period: int = 20) -> 'pd.DataFrame':
    """Compute oscillator indicators with custom parameters."""
    config = {
        'stoch_k': stoch_k,
        'stoch_d': stoch_d,
        'williams_period': williams_period,
        'cci_period': cci_period
    }
    
    core = get_vectorized_processing_core()
    return core._compute_oscillator_indicators(data, config)

def compute_pattern_indicators(data: 'pd.DataFrame') -> 'pd.DataFrame':
    """Compute pattern recognition indicators."""
    core = get_vectorized_processing_core()
    return core._compute_pattern_indicators(data, {})

# Hardware optimization convenience functions
def get_hardware_performance_report() -> Optional[Dict[str, Any]]:
    """Get comprehensive hardware performance report."""
    core = get_vectorized_processing_core()
    return core.get_hardware_performance_report()

def optimize_matrix_operation_with_hardware(data: Union['np.ndarray', 'pd.DataFrame'], 
                                          operation_func: Callable,
                                          *args, **kwargs) -> Any:
    """Optimize a matrix operation using available hardware."""
    try:
        from .hardware_integration import optimize_matrix_operation
        return optimize_matrix_operation(data, operation_func, *args, **kwargs)
    except ImportError:
        # Fallback to standard operation
        return operation_func(data, *args, **kwargs)

def cleanup_hardware_resources():
    """Cleanup hardware resources."""
    core = get_vectorized_processing_core()
    core.cleanup_hardware_resources()

def get_processing_performance_stats() -> Dict[str, Any]:
    """Get comprehensive processing performance statistics."""
    core = get_vectorized_processing_core()
    return core.get_processing_stats()

# Additional convenience functions for common operations
def matrix_multiply(a: 'np.ndarray', b: 'np.ndarray', use_gpu: bool = True) -> 'np.ndarray':
    """Convenient matrix multiplication with GPU option."""
    if use_gpu:
        try:
            return gpu_matrix_multiply(a, b)
        except Exception:
            # Fallback to safe CPU multiplication
            return safe_matrix_multiply(a, b)
    else:
        return safe_matrix_multiply(a, b)

def correlation_matrix(data: Union['pd.DataFrame', 'np.ndarray'], 
                     method: str = 'pearson', use_gpu: bool = True) -> 'np.ndarray':
    """Convenient correlation matrix computation with GPU option."""
    if use_gpu:
        try:
            return correlation_matrix_gpu(data)
        except Exception:
            # Fallback to safe CPU correlation
            return safe_correlation_matrix(data)
    else:
        return safe_correlation_matrix(data)

def matrix_inverse(matrix: 'np.ndarray', use_gpu: bool = True) -> 'np.ndarray':
    """Convenient matrix inversion with GPU option."""
    if use_gpu:
        try:
            ops = get_enhanced_matrix_operations()
            return ops.matrix_inverse(matrix, use_gpu=True)
        except Exception:
            # Fallback to safe CPU inversion
            return safe_matrix_inverse(matrix)
    else:
        return safe_matrix_inverse(matrix)

def eigendecomposition(matrix: 'np.ndarray', use_gpu: bool = True) -> Tuple['np.ndarray', 'np.ndarray']:
    """Convenient eigendecomposition with GPU option."""
    if use_gpu:
        try:
            return eigendecomposition_gpu(matrix)
        except Exception:
            # Fallback to CPU eigendecomposition
            ops = get_unified_matrix_operations()
            return ops.eigendecomposition(matrix)
    else:
        ops = get_unified_matrix_operations()
        return ops.eigendecomposition(matrix)

def svd_decomposition(matrix: 'np.ndarray', k: Optional[int] = None, 
                     use_gpu: bool = True) -> Tuple['np.ndarray', 'np.ndarray', 'np.ndarray']:
    """Convenient SVD decomposition with GPU option."""
    if use_gpu:
        try:
            return svd_gpu(matrix, k)
        except Exception:
            # Fallback to CPU SVD
            ops = get_unified_matrix_operations()
            return ops.svd_decomposition(matrix, k)
    else:
        ops = get_unified_matrix_operations()
        return ops.svd_decomposition(matrix, k)

def batch_process(data: Union['np.ndarray', 'pd.DataFrame'], operation: str, **kwargs) -> Any:
    """Convenient batch processing with automatic optimization."""
    ops = get_unified_matrix_operations()
    return ops.batch_process(data, operation, **kwargs)

def optimize_memory_usage() -> Dict[str, Any]:
    """Convenient memory optimization."""
    ops = get_unified_matrix_operations()
    return ops.optimize_memory_usage()

def get_performance_stats() -> Dict[str, Any]:
    """Get comprehensive performance statistics from all components."""
    stats = {}
    
    # Get stats from unified operations
    try:
        ops = get_unified_matrix_operations()
        stats['unified_operations'] = ops.get_performance_stats()
        stats['hardware_info'] = ops.get_hardware_info()
    except Exception as e:
        stats['unified_operations_error'] = str(e)
    
    # Get stats from vectorized core
    try:
        core = get_vectorized_processing_core()
        stats['vectorized_core'] = core.get_processing_stats()
    except Exception as e:
        stats['vectorized_core_error'] = str(e)
    
    # Get stats from enhanced operations
    try:
        enhanced_ops = get_enhanced_matrix_operations()
        stats['enhanced_operations'] = enhanced_ops.get_performance_stats()
    except Exception as e:
        stats['enhanced_operations_error'] = str(e)
    
    # Get batch optimization stats
    try:
        stats['batch_optimization'] = get_batch_optimization_stats()
    except Exception as e:
        stats['batch_optimization_error'] = str(e)
    
    return stats

def get_system_info() -> Dict[str, Any]:
    """Get comprehensive system information."""
    import psutil
    import platform
    
    info = {
        'platform': platform.platform(),
        'python_version': platform.python_version(),
        'cpu_count': psutil.cpu_count(),
        'memory_total_gb': psutil.virtual_memory().total / (1024**3),
        'memory_available_gb': psutil.virtual_memory().available / (1024**3),
        'memory_percent': psutil.virtual_memory().percent
    }
    
    # Add GPU info if available
    try:
        import torch
        if torch.cuda.is_available():
            info['cuda_available'] = True
            info['cuda_device_count'] = torch.cuda.device_count()
        elif torch.backends.mps.is_available():
            info['mps_available'] = True
        else:
            info['gpu_available'] = False
    except ImportError:
        info['torch_available'] = False
    
    # Add unified operations hardware info
    try:
        ops = get_unified_matrix_operations()
        info['unified_operations_hardware'] = ops.get_hardware_info()
    except Exception as e:
        info['unified_operations_hardware_error'] = str(e)
    
    return info

# Backwards compatibility aliases
def m1_matrix_multiply(A: 'np.ndarray', B: 'np.ndarray') -> 'np.ndarray':
    """Legacy M1 matrix multiplication function."""
    import logging
    logger = logging.getLogger(__name__)
    logger.warning("⚠️ m1_matrix_multiply() is deprecated. Use matrix_multiply() instead.")
    return matrix_multiply(A, B)

class MatrixConvenience:
    """Convenience class for common matrix operations."""

    @staticmethod
    def multiply(a: 'np.ndarray', b: 'np.ndarray', use_gpu: bool = True) -> 'np.ndarray':
        """Convenient matrix multiplication with hardware optimization."""
        return matrix_multiply(a, b, use_gpu=use_gpu)

    @staticmethod
    def correlation(data: Union['pd.DataFrame', 'np.ndarray']) -> 'np.ndarray':
        """Convenient correlation matrix computation."""
        return correlation_matrix(data)

    @staticmethod
    def inverse(matrix: 'np.ndarray', use_gpu: bool = True) -> 'np.ndarray':
        """Convenient matrix inverse with hardware optimization."""
        return matrix_inverse(matrix, use_gpu=use_gpu)

    @staticmethod
    def eigendecomposition(matrix: 'np.ndarray', use_gpu: bool = True) -> Tuple['np.ndarray', 'np.ndarray']:
        """Convenient eigendecomposition with hardware optimization."""
        return eigendecomposition(matrix, use_gpu=use_gpu)

    @staticmethod
    def svd(matrix: 'np.ndarray', k: Optional[int] = None, use_gpu: bool = True) -> Tuple['np.ndarray', 'np.ndarray', 'np.ndarray']:
        """Convenient SVD with hardware optimization."""
        return svd_decomposition(matrix, k, use_gpu=use_gpu)


def get_enhanced_matrix_operations():
    """Legacy function for backward compatibility."""
    logger = logging.getLogger(__name__)
    logger.warning("⚠️ get_enhanced_matrix_operations() is deprecated. Use get_unified_matrix_operations() instead.")
    return get_unified_matrix_operations()


def safe_matrix_operations(operation_func, *args, **kwargs):
    """
    Safe wrapper for matrix operations with error handling.
    
    Args:
        operation_func: The matrix operation function to execute
        *args: Arguments to pass to the operation function
        **kwargs: Keyword arguments to pass to the operation function
        
    Returns:
        Result of the operation or None if it fails
    """
    try:
        return operation_func(*args, **kwargs)
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.warning(f"Matrix operation failed: {e}")
        return None


def validate_matrix_properties(matrix, **kwargs):
    """
    Validate matrix properties for safe operations.
    
    Args:
        matrix: The matrix to validate
        **kwargs: Additional validation parameters
        
    Returns:
        bool: True if matrix is valid, False otherwise
    """
    try:
        if matrix is None:
            return False
        if hasattr(matrix, 'shape') and len(matrix.shape) == 0:
            return False
        return True
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.warning(f"Matrix validation failed: {e}")
        return False


def optimize_matrix_computations(matrix, **kwargs):
    """
    Optimize matrix computations for better performance.
    
    Args:
        matrix: The matrix to optimize
        **kwargs: Additional optimization parameters
        
    Returns:
        The optimized matrix or the original matrix if optimization fails
    """
    try:
        # For now, just return the matrix as-is
        # In a real implementation, this would apply optimizations
        return matrix
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.warning(f"Matrix optimization failed: {e}")
        return matrix