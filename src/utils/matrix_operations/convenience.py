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

# Import the unified modules with lazy loading to avoid circular imports
try:
    from .unified_operations import (
        get_unified_matrix_operations,
        safe_matrix_multiply as _safe_matrix_multiply,
        safe_matrix_inverse as _safe_matrix_inverse
    )
    UNIFIED_OPERATIONS_IMPORTED = True
except ImportError as e:
    UNIFIED_OPERATIONS_IMPORTED = False
    _safe_matrix_multiply = None
    _safe_matrix_inverse = None
    get_unified_matrix_operations = None

# Import correlation matrix function locally to avoid circular import
def _safe_correlation_matrix(*args, **kwargs):
    """Local wrapper to avoid circular import."""
    try:
        from .unified_operations import safe_correlation_matrix
        return safe_correlation_matrix(*args, **kwargs)
    except ImportError:
        # Fallback implementation
        import numpy as np
        if NUMPY_AVAILABLE and len(args) > 0:
            data = args[0]
            try:
                if hasattr(data, 'corr'):
                    return data.corr()
                elif isinstance(data, np.ndarray):
                    # Handle different array shapes
                    if data.ndim == 1:
                        # Single array - return correlation with itself (1.0)
                        return np.array([[1.0]])
                    elif data.ndim == 2:
                        if data.shape[0] == 1 or data.shape[1] == 1:
                            # Single row or column - return correlation with itself
                            return np.array([[1.0]])
                        else:
                            # Multiple variables - compute correlation matrix
                            return np.corrcoef(data)
                    else:
                        raise ValueError(f"Unsupported array shape: {data.shape}")
                else:
                    raise ValueError(f"Unsupported data type: {type(data)}")
            except Exception as e:
                # If all else fails, return identity matrix as fallback
                if hasattr(data, 'shape') and len(data.shape) >= 2:
                    n_vars = data.shape[1] if data.shape[0] > 1 else 1
                    return np.eye(n_vars)
                else:
                    return np.array([[1.0]])
        else:
            # No numpy available - return simple fallback
            return np.array([[1.0]])

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

# Lazy imports to avoid circular dependencies
def _get_vectorized_core():
    try:
        from .vectorized_core import get_vectorized_processing_core
        return get_vectorized_processing_core()
    except ImportError:
        return None

def _optimize_dataframe(df):
    try:
        from .vectorized_core import optimize_dataframe as _optimize_dataframe_impl
        return _optimize_dataframe_impl(df)
    except ImportError:
        return df

def _vectorized_rolling_features(data, windows, features):
    try:
        from .vectorized_core import vectorized_rolling_features as _vectorized_rolling_features_impl
        return _vectorized_rolling_features_impl(data, windows, features)
    except ImportError:
        return data

def _matrix_correlation_analysis(data, method):
    try:
        from .vectorized_core import matrix_correlation_analysis as _matrix_correlation_analysis_impl
        return _matrix_correlation_analysis_impl(data, method)
    except ImportError:
        return None, None

# Lazy imports for batch operations
def _get_batch_matrix_processor():
    try:
        from .batch_operations import get_batch_matrix_processor
        return get_batch_matrix_processor()
    except ImportError:
        return None

def _batch_matrix_multiply(matrices_a, matrices_b):
    try:
        from .batch_operations import batch_matrix_multiply
        return batch_matrix_multiply(matrices_a, matrices_b)
    except ImportError:
        return []

def _batch_feature_transformation(data, transformations):
    try:
        from .batch_operations import batch_feature_transformation
        return batch_feature_transformation(data, transformations)
    except ImportError:
        return data

def _batch_correlation_analysis(data, method):
    try:
        from .batch_operations import batch_correlation_analysis
        return batch_correlation_analysis(data, method)
    except ImportError:
        return None, None

# Lazy imports for enhanced operations
def _get_enhanced_matrix_operations():
    try:
        from .enhanced_operations import get_enhanced_matrix_operations
        return get_enhanced_matrix_operations()
    except ImportError:
        return None

def _gpu_matrix_multiply(a, b):
    try:
        from .enhanced_operations import gpu_matrix_multiply
        return gpu_matrix_multiply(a, b)
    except ImportError:
        return a @ b

def _correlation_matrix_gpu(data):
    try:
        from .enhanced_operations import correlation_matrix_gpu
        return correlation_matrix_gpu(data)
    except ImportError:
        return None

def _eigendecomposition_gpu(matrix):
    try:
        from .enhanced_operations import eigendecomposition_gpu
        return eigendecomposition_gpu(matrix)
    except ImportError:
        return None, None

def _svd_gpu(matrix, k):
    try:
        from .enhanced_operations import svd_gpu
        return svd_gpu(matrix, k)
    except ImportError:
        return None, None, None

def _optimize_batch_size(operation_name, data_shape, complexity, available_memory_mb):
    try:
        from .enhanced_operations import optimize_batch_size
        return optimize_batch_size(operation_name, data_shape, complexity, available_memory_mb)
    except ImportError:
        return 1000

def _record_batch_performance(operation_name, batch_size, execution_time, memory_usage, data_processed):
    try:
        from .enhanced_operations import record_batch_performance
        return record_batch_performance(operation_name, batch_size, execution_time, memory_usage, data_processed)
    except ImportError:
        return None

def _get_batch_optimization_stats():
    try:
        from .enhanced_operations import get_batch_optimization_stats
        return get_batch_optimization_stats()
    except ImportError:
        return {}

# Matrix operations convenience functions
def safe_matrix_multiply(A: 'np.ndarray', B: 'np.ndarray') -> 'np.ndarray':
    """Safe matrix multiplication with VectorBT optimization and validation."""
    # Try VectorBT optimization first if available
    if VECTORBT_OPTIMIZATIONS_AVAILABLE and vectorbt_matrix_multiply:
        try:
            return vectorbt_matrix_multiply(A, B)
        except Exception as e:
            logger.warning(f"⚠️ VectorBT matrix multiplication failed: {e}, falling back to standard method")

    # Fallback to unified operations if available
    if UNIFIED_OPERATIONS_IMPORTED and _safe_matrix_multiply:
        return _safe_matrix_multiply(A, B)

    # Final fallback to basic numpy implementation
    if NUMPY_AVAILABLE:
        return np.dot(A, B)
    else:
        raise ImportError("Neither unified operations nor numpy available for matrix multiplication")

def safe_correlation_matrix(data: Union['np.ndarray', 'pd.DataFrame']) -> 'np.ndarray':
    """Safe correlation matrix computation with VectorBT optimization."""
    # Try VectorBT optimization first if available
    if VECTORBT_OPTIMIZATIONS_AVAILABLE and vectorbt_correlation_matrix:
        try:
            return vectorbt_correlation_matrix(data)
        except Exception as e:
            logger.warning(f"⚠️ VectorBT correlation matrix failed: {e}, falling back to standard method")

    # Fallback to local implementation (avoids circular import)
    try:
        return _safe_correlation_matrix(data)
    except Exception as e:
        logger.warning(f"⚠️ Local correlation matrix failed: {e}, falling back to standard method")

    # Final fallback to basic numpy implementation
    if NUMPY_AVAILABLE:
        return np.corrcoef(data.T)
    else:
        raise ImportError("Neither unified operations nor numpy available for correlation matrix computation")

def safe_matrix_inverse(matrix: 'np.ndarray') -> 'np.ndarray':
    """Safe matrix inversion."""
    # Fallback to unified operations if available
    if UNIFIED_OPERATIONS_IMPORTED and _safe_matrix_inverse:
        return _safe_matrix_inverse(matrix)

    # Final fallback to basic numpy implementation
    if NUMPY_AVAILABLE:
        return np.linalg.inv(matrix)
    else:
        raise ImportError("Neither unified operations nor numpy available for matrix inversion")

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
    core = _get_vectorized_core()
    if core:
        return core.parallel_feature_engineering(data, feature_functions, max_workers)
    return data

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
    ops = _get_enhanced_matrix_operations()
    if ops:
        return ops.sparse_matrix_multiply(a, b, format)
    return None

def sparse_svd(matrix: 'sparse.spmatrix', k: Optional[int] = None,
              solver: str = 'arpack') -> Tuple['np.ndarray', 'np.ndarray', 'np.ndarray']:
    """Sparse SVD decomposition."""
    ops = _get_enhanced_matrix_operations()
    if ops:
        return ops.sparse_svd(matrix, k, solver)
    return None, None, None

def sparse_eigen(matrix: 'sparse.spmatrix', k: int = 10,
                which: str = 'LM') -> Tuple['np.ndarray', 'np.ndarray']:
    """Sparse eigenvalue decomposition."""
    ops = _get_enhanced_matrix_operations()
    if ops:
        return ops.sparse_eigen(matrix, k, which)
    return None, None

def create_sparse_matrix(matrix: 'np.ndarray',
                        sparsity_threshold: float = 0.1) -> Union['sparse.spmatrix', 'np.ndarray']:
    """Create sparse matrix from dense matrix if beneficial."""
    ops = _get_enhanced_matrix_operations()
    if ops:
        return ops.create_sparse_from_dense(matrix, sparsity_threshold)
    return matrix

def sparse_solve(a: 'sparse.spmatrix', b: 'np.ndarray',
                solver: str = 'spsolve') -> 'np.ndarray':
    """Solve sparse linear system."""
    ops = _get_enhanced_matrix_operations()
    if ops:
        return ops.sparse_solve_linear(a, b, solver)
    return None

# Pipeline operations convenience functions
def create_ml_pipeline(stages_config: List[Dict[str, Any]]) -> 'OptimizedPipelineExecutor':
    """Create an optimized ML processing pipeline."""
    core = _get_vectorized_core()
    if core:
        return core.create_optimized_pipeline(stages_config)
    return None

def execute_ml_pipeline(data: 'pd.DataFrame',
                       pipeline_config: List[Dict[str, Any]],
                       execution_mode: 'PipelineExecutionMode' = None) -> 'PipelineExecutionResult':
    """Execute a complete ML processing pipeline with optimization."""
    core = _get_vectorized_core()
    if core:
        if execution_mode is None:
            from .vectorized_core import PipelineExecutionMode
            execution_mode = PipelineExecutionMode.HYBRID
        return core.execute_ml_pipeline(data, pipeline_config, execution_mode)
    return None

def optimize_pipeline_config(pipeline_config: List[Dict[str, Any]],
                           data_sample: 'pd.DataFrame') -> Dict[str, Any]:
    """Analyze and optimize pipeline execution strategy."""
    core = _get_vectorized_core()
    if core:
        return core.optimize_pipeline_execution(pipeline_config, data_sample)
    return {}

def get_pipeline_executor() -> 'OptimizedPipelineExecutor':
    """Get the global pipeline executor instance."""
    core = _get_vectorized_core()
    if core:
        return core.pipeline_executor
    return None

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

    core = _get_vectorized_core()
    if core:
        return core.compute_trading_indicators(data, config)
    return data

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

    core = _get_vectorized_core()
    if core:
        return core._compute_moving_averages(data, config)
    return data

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

    core = _get_vectorized_core()
    if core:
        return core._compute_momentum_indicators(data, config)
    return data

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

    core = _get_vectorized_core()
    if core:
        return core._compute_volatility_indicators(data, config)
    return data

def compute_volume_indicators(data: 'pd.DataFrame',
                             volume_sma_period: int = 20,
                             obv_smooth: int = 10) -> 'pd.DataFrame':
    """Compute volume-based indicators with custom parameters."""
    config = {
        'volume_sma_period': volume_sma_period,
        'obv_smooth': obv_smooth
    }

    core = _get_vectorized_core()
    if core:
        return core._compute_volume_indicators(data, config)
    return data

def compute_trend_indicators(data: 'pd.DataFrame',
                            adx_period: int = 14) -> 'pd.DataFrame':
    """Compute trend indicators with custom parameters."""
    config = {
        'adx_period': adx_period
    }

    core = _get_vectorized_core()
    if core:
        return core._compute_trend_indicators(data, config)
    return data

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

    core = _get_vectorized_core()
    if core:
        return core._compute_oscillator_indicators(data, config)
    return data

def compute_pattern_indicators(data: 'pd.DataFrame') -> 'pd.DataFrame':
    """Compute pattern recognition indicators."""
    core = _get_vectorized_core()
    if core:
        return core._compute_pattern_indicators(data, {})
    return data

# Hardware optimization convenience functions
def get_hardware_performance_report() -> Optional[Dict[str, Any]]:
    """Get comprehensive hardware performance report."""
    core = _get_vectorized_core()
    if core:
        return core.get_hardware_performance_report()
    return None

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
    core = _get_vectorized_core()
    if core:
        core.cleanup_hardware_resources()

def get_processing_performance_stats() -> Dict[str, Any]:
    """Get comprehensive processing performance statistics."""
    core = _get_vectorized_core()
    if core:
        return core.get_processing_stats()
    return {}

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
            ops = _get_enhanced_matrix_operations()
            if ops:
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
        core = _get_vectorized_core()
        if core:
            stats['vectorized_core'] = core.get_processing_stats()
    except Exception as e:
        stats['vectorized_core_error'] = str(e)

    # Get stats from enhanced operations
    try:
        enhanced_ops = _get_enhanced_matrix_operations()
        if enhanced_ops:
            stats['enhanced_operations'] = enhanced_ops.get_performance_stats()
    except Exception as e:
        stats['enhanced_operations_error'] = str(e)

    # Get batch optimization stats
    try:
        stats['batch_optimization'] = _get_batch_optimization_stats()
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

def safe_matrix_operations(operation: str, *args, **kwargs):
    """
    Unified safe matrix operations interface.

    Args:
        operation: The operation to perform ('multiply', 'correlation', 'inverse')
        *args: Arguments for the operation
        **kwargs: Keyword arguments for the operation

    Returns:
        Result of the matrix operation
    """
    if operation == 'multiply':
        return safe_matrix_multiply(*args, **kwargs)
    elif operation == 'correlation':
        return safe_correlation_matrix(*args, **kwargs)
    elif operation == 'inverse':
        return safe_matrix_inverse(*args, **kwargs)
    else:
        raise ValueError(f"Unknown operation: {operation}. Supported operations: 'multiply', 'correlation', 'inverse'")

def validate_matrix_properties(matrix: 'np.ndarray') -> Dict[str, Any]:
    """
    Validate matrix properties for machine learning operations.

    Args:
        matrix: Input matrix to validate

    Returns:
        Dictionary containing validation results
    """
    if not NUMPY_AVAILABLE:
        raise ImportError("NumPy is required for matrix validation")

    validation_results = {
        'is_finite': np.all(np.isfinite(matrix)),
        'has_nan': np.any(np.isnan(matrix)),
        'has_inf': np.any(np.isinf(matrix)),
        'shape': matrix.shape,
        'dtype': matrix.dtype,
        'memory_usage': matrix.nbytes,
        'is_square': matrix.shape[0] == matrix.shape[1] if len(matrix.shape) == 2 else False,
        'is_symmetric': False,
        'condition_number': None,
        'rank': None
    }

    if len(matrix.shape) == 2:
        validation_results['is_symmetric'] = np.allclose(matrix, matrix.T)
        try:
            validation_results['condition_number'] = np.linalg.cond(matrix)
            validation_results['rank'] = np.linalg.matrix_rank(matrix)
        except np.linalg.LinAlgError:
            validation_results['condition_number'] = float('inf')
            validation_results['rank'] = 0

    return validation_results

def optimize_matrix_computations(matrix: 'np.ndarray', operation: str = 'multiply') -> Dict[str, Any]:
    """
    Optimize matrix computations based on matrix properties.

    Args:
        matrix: Input matrix
        operation: Type of operation to optimize for

    Returns:
        Dictionary containing optimization recommendations
    """
    if not NUMPY_AVAILABLE:
        raise ImportError("NumPy is required for matrix optimization")

    validation = validate_matrix_properties(matrix)

    optimization_results = {
        'use_sparse': False,
        'use_gpu': False,
        'chunk_size': None,
        'memory_efficient': False,
        'recommended_dtype': matrix.dtype,
        'warnings': []
    }

    # Check for sparsity
    if len(matrix.shape) == 2:
        sparsity = np.count_nonzero(matrix) / matrix.size
        if sparsity < 0.1:  # Less than 10% non-zero elements
            optimization_results['use_sparse'] = True
            optimization_results['warnings'].append("Matrix is sparse - consider using sparse operations")

    # Check memory usage
    if validation['memory_usage'] > 100 * 1024 * 1024:  # 100MB
        optimization_results['memory_efficient'] = True
        optimization_results['chunk_size'] = min(1000, matrix.shape[0] // 4)
        optimization_results['warnings'].append("Large matrix detected - consider chunked processing")

    # Check condition number
    if validation['condition_number'] and validation['condition_number'] > 1e12:
        optimization_results['warnings'].append("Matrix is ill-conditioned - numerical stability issues possible")

    # Check for NaN/Inf
    if validation['has_nan'] or validation['has_inf']:
        optimization_results['warnings'].append("Matrix contains NaN or Inf values - clean data before processing")

    return optimization_results

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
