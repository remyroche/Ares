from __future__ import annotations

"""Enhanced Matrix Operations with M1 Optimization Integration.

This module provides comprehensive matrix operations optimized for M1/M2/M3 Macs,
integrating GPU acceleration, memory optimization, CPU optimization, and vectorized
processing for maximum performance in machine learning workflows.

Key Features:
- M1 GPU acceleration with Metal Performance Shaders (MPS)
- Intelligent memory management and optimization
- CPU parallel processing optimization
- Vectorized processing pipelines
- Comprehensive error handling and recovery
- Performance monitoring and adaptive optimization

Usage:
>>> from src.utils.ml_common.matrix_operations import get_enhanced_matrix_operations
>>> ops = get_enhanced_matrix_operations()
>>> result = ops.matrix_multiply(A, B)

>>> # M1-optimized operations
>>> from src.utils.ml_common.matrix_operations import m1_matrix_multiply, m1_batch_process
>>> result = m1_matrix_multiply(A, B)
>>> batch_results = m1_batch_process(data, operation_type="matrix_mult")
"""

import numpy as np
import pandas as pd
import torch
from typing import Any, Dict, List, Optional, Tuple, Union, Callable, Iterator
from contextlib import contextmanager
import logging
import time
import gc

# Import M1 optimization utilities
try:
    from ..m1_gpu_utils import get_m1_gpu_manager, M1GPUManager, m1_tensor_multiply, m1_batch_process as m1_batch_process_gpu
    from ..m1_memory_optimizer import get_m1_memory_optimizer, M1MemoryOptimizer
    from ..m1_cpu_optimizer import get_m1_cpu_optimizer, M1CPUOptimizer, parallel_map
    from ..vectorized_processing_core import get_vectorized_processing_core, VectorizedProcessingCore
    M1_UTILS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"M1 optimization utilities not available: {e}")
    M1_UTILS_AVAILABLE = False

# Import enhanced matrix operations as base
try:
    from ..feature_engineering.enhanced_matrix_operations import *  # type: ignore  # noqa: F401,F403
except Exception as exc:  # pragma: no cover – must never fail silently
    raise ImportError(
        "Unable to import 'enhanced_matrix_operations' – ensure utilities package is intact"
    ) from exc

warn(
    "`src.utils.ml_common.matrix_operations` is the new canonical import path for the"
    " enhanced matrix-operation helpers (formerly Step07).  Please update your imports.",
    category=DeprecationWarning,
    stacklevel=2,
)

logger = logging.getLogger(__name__)

class M1EnhancedMatrixOperations:
    """Enhanced matrix operations with comprehensive M1 optimization integration."""

    def __init__(self, 
                 use_gpu: bool = True, 
                 memory_efficient: bool = True,
                 enable_parallel_processing: bool = True,
                 chunk_size: int = 10000, 
                 dtype: torch.dtype = torch.float32,
                 enable_dynamic_batch: bool = True,
                 enable_performance_monitoring: bool = True):
        """Initialize M1-enhanced matrix operations.

        Args:
            use_gpu: Whether to use GPU acceleration
            memory_efficient: Whether to use memory-efficient operations
            enable_parallel_processing: Whether to enable parallel processing
            chunk_size: Chunk size for large matrices
            dtype: Default data type for tensors
            enable_dynamic_batch: Whether to use dynamic batch optimization
            enable_performance_monitoring: Whether to enable performance monitoring
        """
        self.use_gpu = use_gpu
        self.memory_efficient = memory_efficient
        self.enable_parallel_processing = enable_parallel_processing
        self.chunk_size = chunk_size
        self.dtype = dtype
        self.enable_dynamic_batch = enable_dynamic_batch
        self.enable_performance_monitoring = enable_performance_monitoring

        # Initialize M1 optimization components
        self._init_m1_components()
        
        # Initialize base operations if available
        if BASE_OPERATIONS_AVAILABLE:
            self.base_ops = get_base_enhanced_ops()
        else:
            self.base_ops = None

        # Performance tracking
        self.performance_stats = {
            'total_operations': 0,
            'gpu_operations': 0,
            'cpu_operations': 0,
            'memory_optimizations': 0,
            'average_execution_time': 0.0,
            'peak_memory_usage': 0.0
        }

        self.logger = logger.getChild('M1EnhancedMatrixOperations')
        self.logger.info(f"🔧 M1 Enhanced Matrix Operations initialized (GPU: {self.use_gpu}, Parallel: {self.enable_parallel_processing})")

    def _init_m1_components(self):
        """Initialize M1 optimization components."""
        if not M1_UTILS_AVAILABLE:
            self.logger.warning("⚠️ M1 utilities not available, using fallback implementations")
            self.gpu_manager = None
            self.memory_optimizer = None
            self.cpu_optimizer = None
            self.vectorized_core = None
            return

        try:
            # Initialize M1 GPU manager
            if self.use_gpu:
                self.gpu_manager = get_m1_gpu_manager()
                self.logger.info(f"🎯 M1 GPU Manager initialized: {self.gpu_manager.device}")
            else:
                self.gpu_manager = None

            # Initialize M1 memory optimizer
            if self.memory_efficient:
                self.memory_optimizer = get_m1_memory_optimizer()
                self.logger.info("🧠 M1 Memory Optimizer initialized")

            # Initialize M1 CPU optimizer
            if self.enable_parallel_processing:
                self.cpu_optimizer = get_m1_cpu_optimizer()
                self.logger.info(f"⚡ M1 CPU Optimizer initialized: {self.cpu_optimizer.max_workers} workers")

            # Initialize vectorized processing core
            self.vectorized_core = get_vectorized_processing_core()
            self.logger.info("🔄 Vectorized Processing Core initialized")

        except Exception as e:
            self.logger.error(f"Failed to initialize M1 components: {e}")
            self.gpu_manager = None
            self.memory_optimizer = None
            self.cpu_optimizer = None
            self.vectorized_core = None

    @contextmanager
    def operation_context(self, operation_name: str = "matrix_op"):
        """Context manager for matrix operations with comprehensive monitoring."""
        start_time = time.time()
        start_memory = 0.0

        # Memory tracking
        if self.memory_optimizer:
            start_memory = self.memory_optimizer.get_memory_usage()['rss_gb']
            with self.memory_optimizer.memory_checkpoint(operation_name):
                try:
                    yield
                finally:
                    self._log_operation_performance(operation_name, start_time, start_memory)
        else:
            try:
                yield
            finally:
                self._log_operation_performance(operation_name, start_time, start_memory)

    def _log_operation_performance(self, operation_name: str, start_time: float, start_memory: float):
        """Log operation performance and update statistics."""
        execution_time = time.time() - start_time
        
        # Update performance stats
        self.performance_stats['total_operations'] += 1
        self.performance_stats['average_execution_time'] = (
            (self.performance_stats['average_execution_time'] * (self.performance_stats['total_operations'] - 1)) +
            execution_time
        ) / self.performance_stats['total_operations']

        # Memory tracking
        if self.memory_optimizer:
            end_memory = self.memory_optimizer.get_memory_usage()['rss_gb']
            memory_delta = end_memory - start_memory
            if abs(memory_delta) > 0.1:  # Significant memory change
                self.performance_stats['memory_optimizations'] += 1

        # Log performance
        if execution_time >= 0.2:
            self.logger.info(f"✅ {operation_name} completed in {execution_time*1000:.0f} ms")
        else:
            self.logger.debug(f"⚡ {operation_name} completed in {execution_time:.4f}s")

    @with_error_handling("m1_matrix_multiply")
    @with_gpu_fallback("m1_matrix_multiply")
    def matrix_multiply(self, a: Union[np.ndarray, torch.Tensor],
                       b: Union[np.ndarray, torch.Tensor],
                       use_gpu: Optional[bool] = None) -> Union[np.ndarray, torch.Tensor]:
        """M1-optimized matrix multiplication with intelligent device selection."""
        use_gpu = use_gpu if use_gpu is not None else self.use_gpu

        with self.operation_context("matrix_multiply"):
            # Convert to tensors if needed
            if not isinstance(a, torch.Tensor):
                a = torch.from_numpy(np.asarray(a, dtype=np.float32))
            if not isinstance(b, torch.Tensor):
                b = torch.from_numpy(np.asarray(b, dtype=np.float32))

            # Determine optimal device and precision
            if use_gpu and self.gpu_manager:
                # Use M1 GPU manager for intelligent device selection
                data_size = a.numel() + b.numel()
                should_use_gpu = self.gpu_manager.should_use_gpu(
                    data_size, "matrix_mult", dtype=a.dtype, shape=tuple(a.shape)
                )

                if should_use_gpu:
                    # GPU-accelerated multiplication
                    a_gpu = self.gpu_manager.to_device(a, "matrix_mult")
                    b_gpu = self.gpu_manager.to_device(b, "matrix_mult")
                    
                    with self.gpu_manager.gpu_context("matrix_multiply"):
                        result = torch.matmul(a_gpu, b_gpu)
                    
                    self.performance_stats['gpu_operations'] += 1
                    
                    # Convert back to CPU if input was numpy
                    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
                        return result.cpu().numpy()
                    return result

            # CPU fallback
            result = torch.matmul(a, b)
            self.performance_stats['cpu_operations'] += 1

            # Convert back to numpy if input was numpy
            if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
                return result.numpy()
            return result

    @with_error_handling("m1_batch_matrix_multiply")
    def batch_matrix_multiply(self, matrices_a: List[np.ndarray],
                            matrices_b: List[np.ndarray],
                            batch_size: Optional[int] = None) -> List[np.ndarray]:
        """M1-optimized batch matrix multiplication with parallel processing."""
        with self.operation_context("batch_matrix_multiply"):
            if not matrices_a or not matrices_b:
                return []

            # Determine optimal batch size
            if batch_size is None:
                if self.cpu_optimizer:
                    # Use M1 CPU optimizer for optimal batch size
                    sample_shape = matrices_a[0].shape
                    batch_size = self.cpu_optimizer.get_optimal_workers_for_task("cpu_bound")
                else:
                    batch_size = self.chunk_size

            # Use M1 batch processing if available
            if self.gpu_manager and self.use_gpu:
                try:
                    # Convert to tensors
                    a_tensors = [torch.from_numpy(a.astype(np.float32)) for a in matrices_a]
                    b_tensors = [torch.from_numpy(b.astype(np.float32)) for b in matrices_b]
                    
                    # Stack for batch processing
                    a_batch = torch.stack(a_tensors)
                    b_batch = torch.stack(b_tensors)
                    
                    # Use M1 batch processing
                    results = m1_batch_process_gpu(
                        a_batch,
                        batch_size=batch_size,
                        op=lambda x: torch.bmm(x, b_batch[:x.shape[0]]),
                        operation_type="matrix_mult",
                        return_cpu=True
                    )
                    
                    return [result.numpy() for result in results]
                    
                except Exception as e:
                    self.logger.warning(f"M1 batch processing failed: {e}, falling back to sequential")

            # Fallback to sequential processing
            results = []
            for i in range(0, len(matrices_a), batch_size):
                end_idx = min(i + batch_size, len(matrices_a))
                batch_a = matrices_a[i:end_idx]
                batch_b = matrices_b[i:end_idx]
                
                batch_results = []
                for a, b in zip(batch_a, batch_b):
                    result = self.matrix_multiply(a, b)
                    batch_results.append(result)
                
                results.extend(batch_results)

            return results

    @with_error_handling("m1_correlation_matrix")
    def correlation_matrix(self, data: Union[pd.DataFrame, np.ndarray],
                         method: str = 'pearson') -> np.ndarray:
        """M1-optimized correlation matrix computation."""
        with self.operation_context("correlation_matrix"):
            if isinstance(data, pd.DataFrame):
                # Use vectorized processing core for optimization
                if self.vectorized_core:
                    corr_matrix, _ = self.vectorized_core.matrix_correlation_analysis(data, method)
                    return corr_matrix
                else:
                    # Fallback to pandas
                    return data.corr(method=method).values
            else:
                # NumPy array
                if method == 'pearson':
                    return np.corrcoef(data.T)
                else:
                    # Convert to DataFrame for other methods
                    df = pd.DataFrame(data.T)
                    return df.corr(method=method).values

    @with_error_handling("m1_eigendecomposition")
    def eigendecomposition(self, matrix: np.ndarray,
                          use_gpu: Optional[bool] = None) -> Tuple[np.ndarray, np.ndarray]:
        """M1-optimized eigendecomposition with GPU acceleration."""
        use_gpu = use_gpu if use_gpu is not None else self.use_gpu

        with self.operation_context("eigendecomposition"):
            if use_gpu and self.gpu_manager:
                try:
                    # Convert to tensor and move to GPU
                    matrix_tensor = torch.from_numpy(matrix.astype(np.float32))
                    matrix_tensor = self.gpu_manager.to_device(matrix_tensor, "general")
                    
                    with self.gpu_manager.gpu_context("eigendecomposition"):
                        eigenvalues, eigenvectors = torch.linalg.eigh(matrix_tensor)
                    
                    return eigenvalues.cpu().numpy(), eigenvectors.cpu().numpy()
                    
                except Exception as e:
                    self.logger.warning(f"GPU eigendecomposition failed: {e}, using CPU")
                    return np.linalg.eigh(matrix)
            else:
                return np.linalg.eigh(matrix)

    @with_error_handling("m1_svd_decomposition")
    def svd_decomposition(self, matrix: np.ndarray,
                         k: Optional[int] = None,
                         use_gpu: Optional[bool] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """M1-optimized SVD decomposition with GPU acceleration."""
        use_gpu = use_gpu if use_gpu is not None else self.use_gpu

        with self.operation_context("svd_decomposition"):
            if use_gpu and self.gpu_manager:
                try:
                    # Convert to tensor and move to GPU
                    matrix_tensor = torch.from_numpy(matrix.astype(np.float32))
                    matrix_tensor = self.gpu_manager.to_device(matrix_tensor, "general")
                    
                    with self.gpu_manager.gpu_context("svd_decomposition"):
                        U, S, V = torch.linalg.svd(matrix_tensor)
                    
                    # Truncate if k is specified
                    if k is not None:
                        U = U[:, :k]
                        S = S[:k]
                        V = V[:k, :]
                    
                    return U.cpu().numpy(), S.cpu().numpy(), V.cpu().numpy()
                    
                except Exception as e:
                    self.logger.warning(f"GPU SVD failed: {e}, using CPU")
                    U, S, V = np.linalg.svd(matrix, full_matrices=False)
                    
                    if k is not None:
                        U = U[:, :k]
                        S = S[:k]
                        V = V[:k, :]
                    return U, S, V
            else:
                U, S, V = np.linalg.svd(matrix, full_matrices=False)
                
                if k is not None:
                    U = U[:, :k]
                    S = S[:k]
                    V = V[:k, :]
                return U, S, V

    @with_error_handling("m1_parallel_operations")
    def parallel_matrix_operations(self, matrices: List[np.ndarray],
                                 operation: str = "eigen",
                                 max_workers: Optional[int] = None) -> List[Any]:
        """Parallel matrix operations using M1 CPU optimization."""
        with self.operation_context("parallel_matrix_operations"):
            if not matrices:
                return []

            if not self.cpu_optimizer or not self.enable_parallel_processing:
                # Sequential fallback
                return [self._apply_operation(matrix, operation) for matrix in matrices]

            # Determine optimal number of workers
            if max_workers is None:
                max_workers = self.cpu_optimizer.get_optimal_workers_for_task("cpu_bound")

            # Use M1 parallel processing
            try:
                results = self.cpu_optimizer.parallel_process(
                    matrices,
                    lambda matrix: self._apply_operation(matrix, operation),
                    task_type="cpu_bound"
                )
                return results
            except Exception as e:
                self.logger.warning(f"Parallel processing failed: {e}, falling back to sequential")
                return [self._apply_operation(matrix, operation) for matrix in matrices]

    def _apply_operation(self, matrix: np.ndarray, operation: str) -> Any:
        """Apply a specific operation to a matrix."""
        if operation == "eigen":
            return self.eigendecomposition(matrix)
        elif operation == "svd":
            return self.svd_decomposition(matrix)
        elif operation == "inverse":
            return np.linalg.inv(matrix)
        elif operation == "determinant":
            return np.linalg.det(matrix)
        elif operation == "norm":
            return np.linalg.norm(matrix)
        else:
            raise ValueError(f"Unsupported operation: {operation}")

    @with_memory_optimization("m1_memory_cleanup")
    def optimize_memory(self) -> Dict[str, Any]:
        """Comprehensive memory optimization using M1 memory optimizer."""
        if self.memory_optimizer:
            return self.memory_optimizer.optimize_memory()
        else:
            # Fallback memory cleanup
            gc.collect()
            return {'gc_collected': gc.collect(), 'memory_freed_mb': 0}

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        stats = {
            'm1_enhanced_operations': self.performance_stats.copy(),
            'gpu_enabled': self.use_gpu and self.gpu_manager is not None,
            'memory_optimization_enabled': self.memory_efficient and self.memory_optimizer is not None,
            'parallel_processing_enabled': self.enable_parallel_processing and self.cpu_optimizer is not None,
            'vectorized_processing_enabled': self.vectorized_core is not None
        }

        # Add M1 component stats
        if self.gpu_manager:
            stats['gpu_device'] = str(self.gpu_manager.device)
            stats['gpu_memory_info'] = self.gpu_manager.memory_info

        if self.memory_optimizer:
            stats['memory_report'] = self.memory_optimizer.get_memory_report()

        if self.cpu_optimizer:
            stats['cpu_report'] = self.cpu_optimizer.get_cpu_usage_report()

        if self.vectorized_core:
            stats['vectorized_stats'] = self.vectorized_core.get_processing_stats()

        return stats

    def reset_performance_stats(self):
        """Reset performance statistics."""
        self.performance_stats = {
            'total_operations': 0,
            'gpu_operations': 0,
            'cpu_operations': 0,
            'memory_optimizations': 0,
            'average_execution_time': 0.0,
            'peak_memory_usage': 0.0
        }
        self.logger.info("📊 Performance statistics reset")

# Global instance
_m1_enhanced_matrix_ops = None

def get_enhanced_matrix_operations() -> M1EnhancedMatrixOperations:
    """Get global M1-enhanced matrix operations instance."""
    global _m1_enhanced_matrix_ops
    if _m1_enhanced_matrix_ops is None:
        _m1_enhanced_matrix_ops = M1EnhancedMatrixOperations()
    return _m1_enhanced_matrix_ops

# Convenience functions for M1-optimized operations
def m1_matrix_multiply(a: Union[np.ndarray, torch.Tensor], 
                      b: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """M1-optimized matrix multiplication."""
    ops = get_enhanced_matrix_operations()
    return ops.matrix_multiply(a, b)

def m1_batch_process(data: Union[np.ndarray, torch.Tensor],
                    batch_size: Optional[int] = None,
                    operation_type: str = "general") -> Union[np.ndarray, torch.Tensor, Iterator]:
    """M1-optimized batch processing."""
    ops = get_enhanced_matrix_operations()
    if hasattr(ops, 'gpu_manager') and ops.gpu_manager:
        return m1_batch_process_gpu(data, batch_size=batch_size, operation_type=operation_type)
    else:
        # CPU fallback
        if isinstance(data, torch.Tensor):
            return data
        else:
            return torch.from_numpy(data)

def m1_correlation_matrix(data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
    """M1-optimized correlation matrix."""
    ops = get_enhanced_matrix_operations()
    return ops.correlation_matrix(data)

def m1_eigendecomposition(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """M1-optimized eigendecomposition."""
    ops = get_enhanced_matrix_operations()
    return ops.eigendecomposition(matrix)

def m1_svd_decomposition(matrix: np.ndarray, k: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """M1-optimized SVD decomposition."""
    ops = get_enhanced_matrix_operations()
    return ops.svd_decomposition(matrix, k)

def m1_parallel_operations(matrices: List[np.ndarray], 
                          operation: str = "eigen") -> List[Any]:
    """M1-optimized parallel matrix operations."""
    ops = get_enhanced_matrix_operations()
    return ops.parallel_matrix_operations(matrices, operation)

def m1_optimize_memory() -> Dict[str, Any]:
    """M1-optimized memory cleanup."""
    ops = get_enhanced_matrix_operations()
    return ops.optimize_memory()

def get_m1_performance_stats() -> Dict[str, Any]:
    """Get M1 performance statistics."""
    ops = get_enhanced_matrix_operations()
    return ops.get_performance_stats()

# Backward compatibility - re-export from base operations if available
if BASE_OPERATIONS_AVAILABLE:
    # Re-export commonly used functions from base operations
    from ..enhanced_matrix_operations import (
        gpu_matrix_multiply, correlation_matrix_gpu, eigendecomposition_gpu, svd_gpu,
        optimize_batch_size, record_batch_performance, get_batch_optimization_stats,
        sparse_matrix_multiply, sparse_svd, sparse_eigen, create_sparse_matrix, sparse_solve,
        register_custom_matrix_operation, execute_custom_matrix_operation, list_custom_matrix_operations
    )

# Export all public functions and classes
__all__ = [
    # Main classes
    'M1EnhancedMatrixOperations', 'get_enhanced_matrix_operations',
    
    # M1-optimized convenience functions
    'm1_matrix_multiply', 'm1_batch_process', 'm1_correlation_matrix',
    'm1_eigendecomposition', 'm1_svd_decomposition', 'm1_parallel_operations',
    'm1_optimize_memory', 'get_m1_performance_stats',
    
    # Error handling and optimization
    'with_error_handling', 'with_gpu_fallback', 'with_memory_optimization',
    'DynamicBatchOptimizer', 'BatchOptimizationStrategy', 'OperationComplexity',
    'ErrorHandler', 'OptimizationError', 'GPUError', 'MemoryError', 'MatrixOperationError'
]

# Add base operation exports if available
if BASE_OPERATIONS_AVAILABLE:
    __all__.extend([
        'gpu_matrix_multiply', 'correlation_matrix_gpu', 'eigendecomposition_gpu', 'svd_gpu',
        'optimize_batch_size', 'record_batch_performance', 'get_batch_optimization_stats',
        'sparse_matrix_multiply', 'sparse_svd', 'sparse_eigen', 'create_sparse_matrix', 'sparse_solve',
        'register_custom_matrix_operation', 'execute_custom_matrix_operation', 'list_custom_matrix_operations'
    ])