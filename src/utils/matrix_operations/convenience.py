"""
Convenience Functions - Unified Implementation

This module provides convenient wrapper functions for common matrix operations
with backwards compatibility and easy access to unified functionality.
"""

from typing import Any, Dict, List, Optional, Tuple, Union, Callable

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

from .vectorized_core import (
    get_vectorized_processing_core,
    optimize_dataframe as _optimize_dataframe,
    vectorized_rolling_features as _vectorized_rolling_features,
    matrix_correlation_analysis as _matrix_correlation_analysis
)

from .batch_operations import (
    get_batch_matrix_processor,
    batch_matrix_multiply as _batch_matrix_multiply,
    batch_feature_transformation as _batch_feature_transformation,
    batch_correlation_analysis as _batch_correlation_analysis
)

from .enhanced_operations import (
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
    """Safe matrix multiplication with validation."""
    return _safe_matrix_multiply(A, B)

def safe_correlation_matrix(data: Union['np.ndarray', 'pd.DataFrame']) -> 'np.ndarray':
    """Safe correlation matrix computation."""
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
    """Create vectorized rolling features."""
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
    """Convenience function for batch matrix multiplication."""
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

def get_enhanced_matrix_operations():
    """Legacy function for backward compatibility."""
    import logging
    logger = logging.getLogger(__name__)
    logger.warning("⚠️ get_enhanced_matrix_operations() is deprecated. Use get_unified_matrix_operations() instead.")
    return get_unified_matrix_operations()