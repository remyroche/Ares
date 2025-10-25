"""
Unified Matrix Operations - Core Implementation

This module provides the core unified matrix operations that consolidate
functionality from multiple scattered sources while maintaining backwards compatibility.

Key Features:
- Single source of truth for matrix operations
- Apple Silicon M1/M2/M3 optimization
- Memory management and GPU acceleration
- Comprehensive error handling
- Backwards compatibility with existing code
"""

import logging
import time
import gc
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor

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
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None

# Import existing utility frameworks with fallback handling
# Note: Using lazy imports to avoid circular dependencies
UTILITIES_AVAILABLE = False

# Lazy imports to avoid circular dependencies - these will be imported when needed
VECTORIZED_CORE_AVAILABLE = True
VECTORBT_OPTIMIZATIONS_AVAILABLE = True
VECTORBT_ROLLING_AVAILABLE = True
UNIFIED_VECTORIZATION_AVAILABLE = True

# Lazy import functions
def _get_vectorized_processing_core():
    """Lazy import vectorized processing core."""
    try:
        from .vectorized_core import get_vectorized_processing_core
        return get_vectorized_processing_core()
    except ImportError:
        return None

def _get_vectorbt_optimized_operations():
    """Lazy import VectorBT optimized operations."""
    try:
        from .vectorbt_optimizations import get_vectorbt_optimized_operations
        return get_vectorbt_optimized_operations()
    except ImportError:
        return None

def _get_vectorbt_rolling_optimizer():
    """Lazy import VectorBT rolling optimizer."""
    try:
        from src.feature_generation.utils.vectorbt_rolling_optimizer import get_vectorbt_rolling_optimizer
        return get_vectorbt_rolling_optimizer()
    except ImportError:
        return None

def _get_unified_vectorization_manager():
    """Lazy import unified vectorization manager."""
    try:
        from src.feature_generation.utils.unified_vectorization_manager import get_unified_vectorization_manager
        return get_unified_vectorization_manager()
    except ImportError:
        return None

# Import hardware optimizations
try:
    from ..hardware.m1_gpu_utils import M1GPUManager
    from ..hardware.m1_memory_optimizer import M1MemoryOptimizer
    from ..hardware.m1_cpu_optimizer import M1CPUOptimizer
    HARDWARE_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Hardware optimizations not available: {e}")
    HARDWARE_AVAILABLE = False

# Import pandas for ewm operations
try:
    PANDAS_AVAILABLE = True
except ImportError:
    logging.warning("Pandas not available - ewm operations will be limited")
    PANDAS_AVAILABLE = False

# Fallback for safe_correlation if not available - deferred to avoid circular imports
_safe_correlation_func = None
_utilities_cache = {}

def _get_safe_correlation():
    """Get safe_correlation function from utilities."""
    global _safe_correlation_func
    if _safe_correlation_func is None:
        utilities = _get_utilities()
        _safe_correlation_func = utilities.get('safe_correlation', None)
    return _safe_correlation_func

def _get_utilities():
    """Lazy import of utilities to avoid circular dependencies."""
    global _utilities_cache
    if _utilities_cache:
        return _utilities_cache
    try:
        # Use lazy imports to avoid circular dependency
        import importlib
        common_ops = importlib.import_module('src.utils.common_operations')
        math_val = importlib.import_module('src.utils.math_validation')

        get_m1_gpu_manager = getattr(common_ops, 'get_m1_gpu_manager', None)
        get_m1_memory_optimizer = getattr(common_ops, 'get_m1_memory_optimizer', None)
        get_m1_cpu_optimizer = getattr(common_ops, 'get_m1_cpu_optimizer', None)
        safe_divide = getattr(math_val, 'safe_divide', None)
        safe_sqrt = getattr(math_val, 'safe_sqrt', None)
        safe_correlation = getattr(math_val, 'safe_correlation', None)

        if None in [get_m1_gpu_manager, get_m1_memory_optimizer, get_m1_cpu_optimizer, safe_divide, safe_sqrt, safe_correlation]:
            raise ImportError("Required utilities not found")

        _utilities_cache = {
            'get_m1_gpu_manager': get_m1_gpu_manager,
            'get_m1_memory_optimizer': get_m1_memory_optimizer,
            'get_m1_cpu_optimizer': get_m1_cpu_optimizer,
            'safe_divide': safe_divide,
            'safe_sqrt': safe_sqrt,
            'safe_correlation': safe_correlation
        }
        UTILITIES_AVAILABLE = True
        return _utilities_cache
    except ImportError as e:
        logging.debug(f"Some utilities not available: {e}")
        return {}

logger = logging.getLogger(__name__)

class UnifiedMatrixOperations:
    """
    Unified matrix operations optimized for Apple Silicon Macs.

    This class provides a consolidated interface for matrix operations that
    intelligently uses available hardware acceleration and existing utilities.
    """

    def __init__(self,
                 enable_gpu: bool = True,
                 enable_memory_optimization: bool = True,
                 enable_parallel: bool = True,
                 chunk_size_mb: int = 256,
                 max_memory_percent: float = 0.7):
        """
        Initialize unified matrix operations.

        Args:
            enable_gpu: Whether to enable GPU acceleration
            enable_memory_optimization: Whether to enable memory optimization
            enable_parallel: Whether to enable parallel processing
            chunk_size_mb: Chunk size in MB for large matrices
            max_memory_percent: Maximum memory usage percentage
        """
        self.logger = logger.getChild('UnifiedMatrixOperations')

        # Configuration
        self.enable_gpu = enable_gpu and UTILITIES_AVAILABLE
        self.enable_memory_optimization = enable_memory_optimization and UTILITIES_AVAILABLE
        self.enable_parallel = enable_parallel and UTILITIES_AVAILABLE
        self.chunk_size_mb = chunk_size_mb
        self.max_memory_percent = max_memory_percent

        # Initialize components
        self._initialize_components()

        # Initialize VectorBT managers
        self._initialize_vectorbt_managers()

        # Initialize math validator for safe operations
        if UTILITIES_AVAILABLE:
            try:
                from ..ml_common.math_validation import MathValidator
                self.math_validator = MathValidator()
                self.logger.debug("✅ Math Validator initialized")
            except ImportError as e:
                self.logger.warning(f"⚠️ Math Validator import failed: {e}")
                self.math_validator = None
                self.logger.info("ℹ️ Math Validator not available")
        else:
            self.math_validator = None

        # Performance tracking
        self.performance_stats = {
            'total_operations': 0,
            'gpu_operations': 0,
            'cpu_operations': 0,
            'parallel_operations': 0,
            'memory_optimized_operations': 0,
            'vectorization_operations': 0,
            'rolling_operations': 0,
            'vectorbt_operations': 0,
            'average_execution_time': 0.0,
            'peak_memory_usage_mb': 0.0
        }

        self.logger.debug("✅ Unified Matrix Operations initialized")
        # Only log configuration on first initialization to reduce verbosity
        if not hasattr(self.__class__, '_config_logged'):
            self.logger.info(f"📊 GPU: {self.enable_gpu}, Memory Opt: {self.enable_memory_optimization}, Parallel: {self.enable_parallel}")
            self.__class__._config_logged = True

    def _initialize_components(self):
        """Initialize all required components."""
        # Lazy load utilities
        utilities = _get_utilities()
        if not utilities:
            self.logger.debug("⚠️ Some utilities not available - using fallback implementations")
            self.gpu_manager = None
            self.memory_optimizer = None
            self.cpu_optimizer = None
            self.vectorized_core = None
            return

        try:
            # Initialize M1 GPU manager
            if self.enable_gpu:
                self.gpu_manager = utilities.get('get_m1_gpu_manager', lambda: None)()
                if self.gpu_manager:
                    self.logger.debug("✅ M1 GPU Manager initialized")
                else:
                    self.logger.info("ℹ️ M1 GPU Manager not available")
            else:
                self.gpu_manager = None

            # Initialize M1 memory optimizer
            if self.enable_memory_optimization:
                self.memory_optimizer = utilities.get('get_m1_memory_optimizer', lambda: None)()
                if self.memory_optimizer:
                    self.logger.debug("✅ M1 Memory Optimizer initialized")
                else:
                    self.logger.info("ℹ️ M1 Memory Optimizer not available")
            else:
                self.memory_optimizer = None

            # Initialize M1 CPU optimizer
            if self.enable_parallel:
                self.cpu_optimizer = utilities.get('get_m1_cpu_optimizer', lambda: None)()
                if self.cpu_optimizer:
                    self.logger.debug("✅ M1 CPU Optimizer initialized")
                else:
                    self.logger.info("ℹ️ M1 CPU Optimizer not available")
            else:
                self.cpu_optimizer = None

            # Initialize vectorized processing core
            if VECTORIZED_CORE_AVAILABLE:
                try:
                    self.vectorized_core = _get_vectorized_processing_core()
                    if self.vectorized_core:
                        self.logger.debug("✅ Vectorized Processing Core initialized")
                    else:
                        self.logger.info("ℹ️ Vectorized Processing Core not available")
                except Exception as e:
                    self.logger.warning(f"⚠️ Vectorized Processing Core not available: {e}")
                    self.vectorized_core = None
            else:
                self.vectorized_core = None

        except Exception as e:
            self.logger.error(f"❌ Error initializing components: {e}")
            self.gpu_manager = None
            self.memory_optimizer = None
            self.cpu_optimizer = None
            self.vectorized_core = None

    def _initialize_vectorbt_managers(self):
        """Initialize VectorBT managers for optimized operations."""
        try:
            # Initialize VectorBTRollingOptimizer
            if VECTORBT_ROLLING_AVAILABLE:
                self.rolling_optimizer = _get_vectorbt_rolling_optimizer()
                if self.rolling_optimizer:
                    # Reduced verbosity - only log once per session
                    if not hasattr(UnifiedMatrixOperations, '_logged_rolling_init'):
                        self.logger.debug("✅ VectorBTRollingOptimizer initialized")
                        UnifiedMatrixOperations._logged_rolling_init = True
                else:
                    self.logger.info("ℹ️ VectorBTRollingOptimizer not available")
            else:
                self.rolling_optimizer = None

            # Initialize UnifiedVectorizationManager
            if UNIFIED_VECTORIZATION_AVAILABLE:
                self.vectorization_manager = _get_unified_vectorization_manager()
                if self.vectorization_manager:
                    self.logger.debug("✅ UnifiedVectorizationManager initialized")
                else:
                    self.logger.info("ℹ️ UnifiedVectorizationManager not available")
            else:
                self.vectorization_manager = None

        except Exception as e:
            # Use module-level logger as fallback if self.logger is not available
            try:
                self.logger.error(f"❌ Error initializing VectorBT managers: {e}")
            except AttributeError:
                logging.error(f"❌ Error initializing VectorBT managers: {e}")
            self.rolling_optimizer = None
            self.vectorization_manager = None

    def matrix_multiply(self, A: 'np.ndarray', B: 'np.ndarray') -> 'np.ndarray':
        """
        Optimized matrix multiplication with automatic hardware selection and VectorBT optimization.

        Args:
            A: First matrix
            B: Second matrix

        Returns:
            Result of matrix multiplication
        """
        start_time = time.time()

        # Validate inputs
        if A.shape[1] != B.shape[0]:
            raise ValueError(f"Matrix dimensions incompatible: {A.shape} @ {B.shape}")

        # Try UnifiedVectorizationManager first if available
        if self.vectorization_manager and self._should_use_vectorization_manager(A, B):
            try:
                result = self.vectorization_manager.matrix_multiply(A, B)
                self.performance_stats['vectorization_operations'] = self.performance_stats.get('vectorization_operations', 0) + 1
                self.logger.debug("✅ UnifiedVectorizationManager matrix multiplication completed")
            except Exception as e:
                self.logger.warning(f"⚠️ UnifiedVectorizationManager matrix multiplication failed: {e}, falling back to VectorBT")
                result = self._fallback_matrix_multiply(A, B)
        else:
            result = self._fallback_matrix_multiply(A, B)

        # Update performance stats
        execution_time = time.time() - start_time
        self.performance_stats['total_operations'] += 1
        self.performance_stats['average_execution_time'] = (
            (self.performance_stats['average_execution_time'] *
             (self.performance_stats['total_operations'] - 1)) + execution_time
        ) / self.performance_stats['total_operations']

        return result

    def _should_use_vectorization_manager(self, A: 'np.ndarray', B: 'np.ndarray') -> bool:
        """Determine if UnifiedVectorizationManager should be used for the operation."""
        if not self.vectorization_manager:
            return False

        # Use UnifiedVectorizationManager for medium to large matrices
        total_elements = A.size + B.size
        return total_elements > 5000  # 5K elements threshold

    def _should_use_vectorization_manager_for_correlation(self, data: Union['np.ndarray', 'pd.DataFrame']) -> bool:
        """Determine if UnifiedVectorizationManager should be used for correlation operations."""
        if not self.vectorization_manager:
            return False

        # Convert to numpy array for size check
        if isinstance(data, pd.DataFrame):
            data_array = data.values
        else:
            data_array = data

        # Use UnifiedVectorizationManager for medium to large datasets
        return data_array.size > 10000  # 10K elements threshold

    def _should_use_vectorization_manager_for_batch(self, data: Union['np.ndarray', 'pd.DataFrame'], operation: str) -> bool:
        """Determine if UnifiedVectorizationManager should be used for batch operations."""
        if not self.vectorization_manager:
            return False

        # Convert to numpy array for size check
        if isinstance(data, pd.DataFrame):
            data_array = data.values
        else:
            data_array = data

        # Use UnifiedVectorizationManager for supported operations and medium to large datasets
        supported_operations = ['correlation', 'rolling_features', 'trading_indicators', 'matrix_multiply', 'feature_engineering']
        return operation in supported_operations and data_array.size > 5000  # 5K elements threshold

    def _should_use_vectorbt(self, A: 'np.ndarray', B: 'np.ndarray') -> bool:
        """Determine if VectorBT should be used for the operation."""
        if not VECTORBT_OPTIMIZATIONS_AVAILABLE:
            return False

        # Use VectorBT for medium to large matrices
        total_elements = A.size + B.size
        return total_elements > 10000  # 10K elements threshold

    def _fallback_matrix_multiply(self, A: 'np.ndarray', B: 'np.ndarray') -> 'np.ndarray':
        """Fallback matrix multiplication using VectorBT or standard methods."""
        # Try VectorBT optimization first if available
        if VECTORBT_OPTIMIZATIONS_AVAILABLE and self._should_use_vectorbt(A, B):
            try:
                vectorbt_ops = _get_vectorbt_optimized_operations()
                result = vectorbt_ops.matrix_multiply(A, B)
                self.performance_stats['vectorbt_operations'] = self.performance_stats.get('vectorbt_operations', 0) + 1
                self.logger.debug("✅ VectorBT matrix multiplication completed")
                return result
            except Exception as e:
                self.logger.warning(f"⚠️ VectorBT matrix multiplication failed: {e}, falling back to standard method")

        # Fall back to standard implementation
        return self._standard_matrix_multiply(A, B)

    def _standard_matrix_multiply(self, A: 'np.ndarray', B: 'np.ndarray') -> 'np.ndarray':
        """Standard matrix multiplication with hardware selection."""
        # Choose optimal method based on size and available hardware
        if self._should_use_gpu(A, B):
            result = self._gpu_matrix_multiply(A, B)
            self.performance_stats['gpu_operations'] += 1
        elif self._should_use_parallel(A, B):
            result = self._parallel_matrix_multiply(A, B)
            self.performance_stats['parallel_operations'] += 1
        else:
            result = self._cpu_matrix_multiply(A, B)
            self.performance_stats['cpu_operations'] += 1

        return result

    def _should_use_gpu(self, A: 'np.ndarray', B: 'np.ndarray') -> bool:
        """Determine if GPU should be used for the operation."""
        if not self.enable_gpu or self.gpu_manager is None:
            return False

        # Use GPU for large matrices (> 1000x1000 elements total)
        total_elements = A.size + B.size
        return total_elements > 1000000  # 1M elements threshold

    def _should_use_parallel(self, A: 'np.ndarray', B: 'np.ndarray') -> bool:
        """Determine if parallel processing should be used."""
        if not self.enable_parallel or self.cpu_optimizer is None:
            return False

        # Use parallel for medium-large matrices
        total_elements = A.size + B.size
        return 100000 < total_elements <= 1000000

    def _gpu_matrix_multiply(self, A: 'np.ndarray', B: 'np.ndarray') -> 'np.ndarray':
        """GPU-accelerated matrix multiplication."""
        try:
            import torch

            # Convert to tensors
            A_tensor = torch.from_numpy(A.astype(np.float32 if np is not None else float))
            B_tensor = torch.from_numpy(B.astype(np.float32 if np is not None else float))

            # Move to MPS (Apple Silicon GPU)
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                A_tensor = A_tensor.to('mps')
                B_tensor = B_tensor.to('mps')

                # Perform multiplication
                result_tensor = torch.matmul(A_tensor, B_tensor)

                # Move back to CPU
                result = result_tensor.cpu().numpy()

                self.logger.debug("✅ GPU matrix multiplication completed")
                return result
            else:
                raise RuntimeError("MPS not available")

        except Exception as e:
            self.logger.warning(f"⚠️ GPU matrix multiplication failed: {e}")
            return self._cpu_matrix_multiply(A, B)

    def _parallel_matrix_multiply(self, A: 'np.ndarray', B: 'np.ndarray') -> 'np.ndarray':
        """Parallel matrix multiplication."""
        try:
            # Use numpy's optimized BLAS operations (already parallelized)
            result = np.dot(A, B)
            self.logger.debug("✅ Parallel matrix multiplication completed")
            return result

        except Exception as e:
            self.logger.warning(f"⚠️ Parallel matrix multiplication failed: {e}")
            return self._cpu_matrix_multiply(A, B)

    def _cpu_matrix_multiply(self, A: 'np.ndarray', B: 'np.ndarray') -> 'np.ndarray':
        """CPU matrix multiplication with memory optimization."""
        try:
            # Use numpy's optimized operations
            result = A @ B
            self.logger.debug("✅ CPU matrix multiplication completed")
            return result

        except Exception as e:
            self.logger.error(f"❌ CPU matrix multiplication failed: {e}")
            raise

    def safe_correlation_matrix(self, data: Union['np.ndarray', 'pd.DataFrame'],
                               method: str = 'pearson') -> 'np.ndarray':
        """
        Safe correlation matrix computation using VectorBT optimization and existing math validation utilities.

        Args:
            data: Input data matrix
            method: Correlation method ('pearson', 'spearman', 'kendall')

        Returns:
            Correlation matrix with safe operations
        """
        # Try UnifiedVectorizationManager first if available
        if self.vectorization_manager and self._should_use_vectorization_manager_for_correlation(data):
            try:
                result = self.vectorization_manager.correlation_matrix(data, method)
                self.performance_stats['vectorization_operations'] = self.performance_stats.get('vectorization_operations', 0) + 1
                self.logger.debug("✅ UnifiedVectorizationManager correlation matrix completed")
                return result
            except Exception as e:
                self.logger.warning(f"⚠️ UnifiedVectorizationManager correlation matrix failed: {e}, falling back to VectorBT")

        # Try VectorBTRollingOptimizer if available
        if self.rolling_optimizer:
            try:
                result = self.rolling_optimizer.correlation_matrix(data, method)
                self.performance_stats['vectorbt_rolling_operations'] = self.performance_stats.get('vectorbt_rolling_operations', 0) + 1
                self.logger.debug("✅ VectorBTRollingOptimizer correlation matrix completed")
                return result
            except Exception as e:
                self.logger.warning(f"⚠️ VectorBTRollingOptimizer correlation matrix failed: {e}, falling back to standard method")

        # Fallback to standard implementation
        if isinstance(data, pd.DataFrame):
            data = data.values

        if data.ndim != 2:
            raise ValueError("Data must be 2-dimensional")

        n_features = data.shape[1]
        correlation_matrix = np.zeros((n_features, n_features))

        # Use existing safe correlation function
        for i in range(n_features):
            for j in range(n_features):
                if i == j:
                    correlation_matrix[i, j] = 1.0
                else:
                    if _safe_correlation_func is not None:
                        corr = safe_correlation_func(data[:, i], data[:, j])
                    else:
                        # Fallback to numpy correlation
                        corr = np.corrcoef(data[:, i], data[:, j])[0, 1]
                    correlation_matrix[i, j] = corr
                    correlation_matrix[j, i] = corr  # Symmetric

        return correlation_matrix

    def matrix_inverse(self, matrix: 'np.ndarray') -> 'np.ndarray':
        """
        Safe matrix inversion with conditioning checks.

        Args:
            matrix: Square matrix to invert

        Returns:
            Inverse of the matrix
        """
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Matrix must be square for inversion")

        # Check condition number for numerical stability
        try:
            cond = np.linalg.cond(matrix)
            if cond > 1e12:  # Very ill-conditioned
                self.logger.warning(f"⚠️ Matrix is ill-conditioned (cond={cond:.2e})")
        except Exception as e:
            self.logger.warning(f"⚠️ Could not check condition number: {e}")

        try:
            # Use numpy's optimized inverse
            inverse = np.linalg.inv(matrix)

            # Verify the result
            product = matrix @ inverse
            identity_error = np.max(np.abs(product - np.eye(matrix.shape[0])))

            if identity_error > 1e-6:
                self.logger.warning(f"⚠️ Matrix inversion may be inaccurate (error={identity_error:.2e})")

            return inverse

        except np.linalg.LinAlgError as e:
            self.logger.error(f"❌ Matrix inversion failed: {e}")
            raise

    def eigendecomposition(self, matrix: 'np.ndarray') -> Tuple['np.ndarray', 'np.ndarray']:
        """
        Eigenvalue decomposition with safety checks.

        Args:
            matrix: Square matrix for decomposition

        Returns:
            Tuple of (eigenvalues, eigenvectors)
        """
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Matrix must be square for eigendecomposition")

        try:
            eigenvalues, eigenvectors = np.linalg.eig(matrix)

            # Sort by absolute eigenvalue magnitude
            idx = np.argsort(np.abs(eigenvalues))[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]

            return eigenvalues, eigenvectors

        except np.linalg.LinAlgError as e:
            self.logger.error(f"❌ Eigendecomposition failed: {e}")
            raise

    def svd_decomposition(self, matrix: 'np.ndarray', k: Optional[int] = None) -> Tuple['np.ndarray', 'np.ndarray', 'np.ndarray']:
        """
        Singular value decomposition with optional dimensionality reduction.

        Args:
            matrix: Input matrix
            k: Number of singular values/vectors to keep (optional)

        Returns:
            Tuple of (U, s, Vh)
        """
        try:
            U, s, Vh = np.linalg.svd(matrix, full_matrices=False)

            if k is not None and k < len(s):
                # Keep only top k components
                U = U[:, :k]
                s = s[:k]
                Vh = Vh[:k, :]

            return U, s, Vh

        except np.linalg.LinAlgError as e:
            self.logger.error(f"❌ SVD decomposition failed: {e}")
            raise

    def batch_process(self, data: Union['np.ndarray', 'pd.DataFrame'],
                     operation: str, **kwargs) -> Any:
        """
        Batch processing with automatic memory management and VectorBT optimization.

        Args:
            data: Input data
            operation: Operation to perform
            **kwargs: Additional arguments

        Returns:
            Processed result
        """
        # Try UnifiedVectorizationManager first if available
        if self.vectorization_manager and self._should_use_vectorization_manager_for_batch(data, operation):
            try:
                result = self.vectorization_manager.batch_process(data, operation, **kwargs)
                self.performance_stats['vectorization_operations'] = self.performance_stats.get('vectorization_operations', 0) + 1
                self.logger.debug("✅ UnifiedVectorizationManager batch processing completed")
                return result
            except Exception as e:
                self.logger.warning(f"⚠️ UnifiedVectorizationManager batch processing failed: {e}, falling back to standard method")

        if isinstance(data, pd.DataFrame):
            data = data.values

        # Check memory usage
        memory_mb = data.nbytes / (1024 * 1024)

        if memory_mb > self.chunk_size_mb:
            self.logger.info(f"📊 Large dataset ({memory_mb:.1f}MB) - using chunked processing")
            return self._chunked_batch_process(data, operation, **kwargs)
        else:
            return self._direct_batch_process(data, operation, **kwargs)

    def _chunked_batch_process(self, data: 'np.ndarray', operation: str, **kwargs) -> Any:
        """Process data in chunks for memory efficiency."""
        try:
            chunk_size = int(self.chunk_size_mb * 1024 * 1024 / data.dtype.itemsize)
        except AttributeError:
            # Fallback if numpy is not available
            chunk_size = 1000
        results = []

        for i in range(0, len(data), chunk_size):
            end_idx = min(i + chunk_size, len(data))
            chunk = data[i:end_idx]

            result = self._direct_batch_process(chunk, operation, **kwargs)
            results.append(result)

        # Combine results based on operation type
        if operation in ['correlation', 'covariance']:
            # Average correlation/covariance matrices
            return np.mean(results, axis=0)
        elif operation == 'mean':
            return np.concatenate(results).mean()
        elif operation == 'std':
            # For std, we need to compute across all chunks
            all_data = np.concatenate(results) if len(results[0].shape) == 1 else np.vstack(results)
            return np.std(all_data, axis=0)
        elif operation == 'ewm_mean':
            # For ewm_mean, we can't simply concatenate results as it depends on temporal order
            # Process the entire dataset as one chunk
            self.logger.info("⚠️ ewm_mean requires processing entire dataset - switching to direct processing")
            return self._direct_batch_process(data, operation, **kwargs)
        elif operation == 'safe_divide':
            # For safe_divide, we need to handle the numerator and denominator properly
            # This is complex for chunked processing, so we'll process the entire dataset
            self.logger.info("⚠️ safe_divide requires processing entire dataset - switching to direct processing")
            return self._direct_batch_process(data, operation, **kwargs)
        else:
            return np.concatenate(results) if len(results[0].shape) == 1 else np.vstack(results)

    def _direct_batch_process(self, data: 'np.ndarray', operation: str, **kwargs) -> Any:
        """Direct batch processing for smaller datasets."""
        if operation == 'correlation':
            return self.safe_correlation_matrix(data)
        elif operation == 'covariance':
            return np.cov(data.T)
        elif operation == 'mean':
            return np.mean(data, axis=0)
        elif operation == 'std':
            return np.std(data, axis=0)
        elif operation == 'normalize':
            mean = np.mean(data, axis=0)
            std = np.std(data, axis=0)
            # Use safe_divide if available, otherwise use numpy operations
            utilities = _get_utilities()
            safe_divide_func = utilities.get('safe_divide')
            if safe_divide_func:
                return (data - mean) / safe_divide_func(std, np.ones_like(std), 1.0)
            else:
                # Fallback to numpy division with epsilon
                return (data - mean) / np.where(std == 0, 1.0, std)
        elif operation == 'safe_divide':
            # Safe division operation: safe_divide(numerator, denominator, default)
            numerator = kwargs.get('numerator', data)
            denominator = kwargs.get('denominator', np.ones_like(data))
            default_value = kwargs.get('default_value', 0.0)

            # Ensure both numerator and denominator are numpy arrays
            numerator = np.asarray(numerator)
            denominator = np.asarray(denominator)

            # Broadcast denominator to match numerator shape if needed
            if denominator.ndim == 1 and numerator.ndim == 2:
                denominator = denominator.reshape(1, -1).repeat(numerator.shape[0], axis=0)
            elif denominator.ndim == 0 and numerator.ndim > 0:
                denominator = np.full_like(numerator, denominator)

            if hasattr(self, 'math_validator') and self.math_validator:
                return self.math_validator.safe_divide(numerator, denominator, default_value)
            else:
                utilities = _get_utilities()
                safe_divide_func = utilities.get('safe_divide')
                if safe_divide_func:
                    return safe_divide_func(numerator, denominator, default_value)
                else:
                    # Fallback to numpy division with epsilon
                    return np.divide(numerator, denominator,
                                   out=np.full_like(numerator, default_value),
                                   where=(denominator != 0))
        elif operation == 'ewm_mean':
            # Exponential weighted moving average
            if not PANDAS_AVAILABLE:
                raise ValueError("ewm_mean operation requires pandas")

            # Convert to DataFrame for ewm operations
            df = pd.DataFrame(data)

            # Get ewm parameters
            span = kwargs.get('span', 20)
            adjust = kwargs.get('adjust', True)
            axis = kwargs.get('axis', 0)

            # Apply ewm mean along specified axis
            if axis == 0:
                result = df.ewm(span=span, adjust=adjust).mean().values
            else:
                result = df.T.ewm(span=span, adjust=adjust).mean().T.values

            return result
        else:
            raise ValueError(f"Unknown operation: {operation}")

    def optimize_memory_usage(self) -> Dict[str, Any]:
        """
        Optimize memory usage using existing memory optimizer.

        Returns:
            Dictionary with memory optimization results
        """
        if not self.enable_memory_optimization or self.memory_optimizer is None:
            self.logger.info("ℹ️ Memory optimization not available")
            return {'status': 'not_available'}

        try:
            # Use existing M1 memory optimizer
            if hasattr(self.memory_optimizer, 'optimize_memory'):
                result = self.memory_optimizer.optimize_memory()
                self.performance_stats['memory_optimized_operations'] += 1
                return result
            else:
                # Fallback memory cleanup
                gc.collect()
                return {'status': 'fallback_gc', 'freed_mb': 0}

        except Exception as e:
            self.logger.warning(f"⚠️ Memory optimization failed: {e}")
            return {'status': 'failed', 'error': str(e)}

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        return self.performance_stats.copy()

    def kmeans_plus_plus_init(self, data: 'np.ndarray', n_components: int, random_state: Optional[int] = None) -> 'np.ndarray':
        """
        K-means++ initialization for cluster centers.

        Args:
            data: Input data matrix (n_samples, n_features)
            n_components: Number of cluster centers to initialize
            random_state: Random seed for reproducibility

        Returns:
            Initial cluster centers (n_components, n_features)
        """
        if random_state is not None:
            np.random.seed(random_state)

        n_samples, n_features = data.shape

        if n_components >= n_samples:
            # If we have more components than samples, use all samples as centers
            return data[np.random.choice(n_samples, n_components, replace=True)]

        # Initialize first center randomly
        centers = np.zeros((n_components, n_features))
        centers[0] = data[np.random.randint(n_samples)]

        # Initialize remaining centers using k-means++ algorithm
        for i in range(1, n_components):
            # Calculate distances to nearest center for each point
            distances = np.full(n_samples, np.inf)
            for j in range(i):
                dist_to_center = np.sum((data - centers[j])**2, axis=1)
                distances = np.minimum(distances, dist_to_center)

            # Choose next center with probability proportional to squared distance
            probabilities = distances / np.sum(distances)
            centers[i] = data[np.random.choice(n_samples, p=probabilities)]

        return centers

    def normalize_matrix(self, data: 'np.ndarray', method: str = 'zscore') -> 'np.ndarray':
        """
        Normalize matrix data using specified method.

        Args:
            data: Input data matrix (n_samples, n_features)
            method: Normalization method ('zscore', 'minmax', 'robust')

        Returns:
            Normalized data matrix
        """
        if data.ndim != 2:
            raise ValueError("Data must be 2-dimensional")

        if method == 'zscore':
            # Z-score normalization: (x - mean) / std
            mean = np.mean(data, axis=0)
            std = np.std(data, axis=0)
            # Avoid division by zero
            std = np.where(std == 0, 1.0, std)
            return (data - mean) / std

        elif method == 'minmax':
            # Min-max normalization: (x - min) / (max - min)
            data_min = np.min(data, axis=0)
            data_max = np.max(data, axis=0)
            # Avoid division by zero
            data_range = np.where(data_max == data_min, 1.0, data_max - data_min)
            return (data - data_min) / data_range

        elif method == 'robust':
            # Robust normalization using median and IQR
            median = np.median(data, axis=0)
            q75, q25 = np.percentile(data, [75, 25], axis=0)
            iqr = q75 - q25
            # Avoid division by zero
            iqr = np.where(iqr == 0, 1.0, iqr)
            return (data - median) / iqr

        else:
            raise ValueError(f"Unknown normalization method: {method}")

    def initialize_covariances(self, data: 'np.ndarray', means: 'np.ndarray', covariance_type: str = 'full') -> 'np.ndarray':
        """
        Initialize covariance matrices for HMM components.

        Args:
            data: Input data matrix (n_samples, n_features)
            means: Component means (n_components, n_features)
            covariance_type: Type of covariance ('full', 'diag', 'spherical', 'tied')

        Returns:
            Initialized covariance matrices
        """
        n_samples, n_features = data.shape
        n_components = means.shape[0]

        if covariance_type == 'full':
            # Full covariance matrices
            covariances = np.zeros((n_components, n_features, n_features))
            for i in range(n_components):
                # Calculate covariance for this component
                centered_data = data - means[i]
                covariances[i] = np.cov(centered_data.T)
                # Add regularization to ensure positive definiteness
                covariances[i] += np.eye(n_features) * 1e-6

        elif covariance_type == 'diag':
            # Diagonal covariance matrices
            covariances = np.zeros((n_components, n_features))
            for i in range(n_components):
                centered_data = data - means[i]
                covariances[i] = np.var(centered_data, axis=0)
                # Add regularization
                covariances[i] += 1e-6

        elif covariance_type == 'spherical':
            # Spherical covariance (same variance for all features)
            covariances = np.zeros(n_components)
            for i in range(n_components):
                centered_data = data - means[i]
                # Use average variance across features
                covariances[i] = np.mean(np.var(centered_data, axis=0)) + 1e-6

        elif covariance_type == 'tied':
            # Tied covariance (same for all components)
            # Calculate global covariance
            global_mean = np.mean(data, axis=0)
            centered_data = data - global_mean
            covariances = np.cov(centered_data.T)
            # Add regularization
            covariances += np.eye(n_features) * 1e-6
            # Return as single matrix for tied case
            return covariances

        else:
            raise ValueError(f"Unknown covariance type: {covariance_type}")

        return covariances

    def get_hardware_info(self) -> Dict[str, Any]:
        """Get hardware capability information."""
        info = {
            'gpu_available': self.enable_gpu and self.gpu_manager is not None,
            'memory_optimizer_available': self.enable_memory_optimization and self.memory_optimizer is not None,
            'cpu_optimizer_available': self.enable_parallel and self.cpu_optimizer is not None,
            'vectorized_core_available': self.vectorized_core is not None,
            'rolling_optimizer_available': self.rolling_optimizer is not None,
            'vectorization_manager_available': self.vectorization_manager is not None
        }

        # Add GPU info if available
        if self.gpu_manager and hasattr(self.gpu_manager, 'get_gpu_info'):
            info['gpu_info'] = self.gpu_manager.get_gpu_info()

        # Add VectorBT manager info if available
        if self.rolling_optimizer and hasattr(self.rolling_optimizer, 'get_performance_stats'):
            info['rolling_optimizer_stats'] = self.rolling_optimizer.get_performance_stats()

        if self.vectorization_manager and hasattr(self.vectorization_manager, 'get_performance_stats'):
            info['vectorization_manager_stats'] = self.vectorization_manager.get_performance_stats()

        return info

    def optimize_dataframe(self, df: Union['pd.DataFrame', 'np.ndarray'],
                          operations: Optional[List[str]] = None) -> Union['pd.DataFrame', 'np.ndarray']:
        """
        Optimize dataframe operations using available hardware acceleration.

        Args:
            df: Input dataframe or numpy array
            operations: List of operations to perform (optional)

        Returns:
            Optimized dataframe or array
        """
        self.logger.info("🔧 Starting dataframe optimization...")

        start_time = time.time()

        try:
            # Convert to appropriate format for processing
            if isinstance(df, pd.DataFrame):
                data = df.values
                is_dataframe = True
                columns = df.columns
                index = df.index
            else:
                data = df.copy()
                is_dataframe = False
                columns = None
                index = None

            # Default operations if none specified
            if operations is None:
                operations = ['memory_optimization', 'dtype_optimization', 'nan_handling']

            # Apply memory optimization
            if 'memory_optimization' in operations:
                self.logger.debug("🧠 Applying memory optimization...")
                memory_stats = self.optimize_memory_usage()
                if memory_stats.get('status') == 'success':
                    self.logger.debug("✅ Memory optimization completed")

            # Apply dtype optimization
            if 'dtype_optimization' in operations and is_dataframe:
                self.logger.debug("🔢 Applying dtype optimization...")
                # Optimize numeric dtypes
                numeric_columns = df.select_dtypes(include=[np.number]).columns
                for col in numeric_columns:
                    if df[col].dtype == 'float64':
                        # Check if we can downcast to float32
                        if df[col].abs().max() < np.finfo(np.float32).max:
                            df[col] = df[col].astype(np.float32)
                    elif df[col].dtype == 'int64':
                        # Check if we can downcast to int32
                        if df[col].min() >= np.iinfo(np.int32).min and df[col].max() <= np.iinfo(np.int32).max:
                            df[col] = df[col].astype(np.int32)

                # Update data array
                data = df.values

            # Apply NaN handling
            if 'nan_handling' in operations:
                self.logger.debug("🔍 Handling NaN values...")
                if np.isnan(data).any():
                    # Use safe operations for NaN handling
                    nan_count = np.isnan(data).sum()
                    self.logger.debug(f"Found {nan_count} NaN values")

                    # Replace NaN with median for numeric data
                    if data.dtype.kind in 'biufc':  # numeric types
                        for col_idx in range(data.shape[1]):
                            col_data = data[:, col_idx]
                            if np.isnan(col_data).any():
                                median_val = np.nanmedian(col_data)
                                if not np.isnan(median_val):
                                    data[:, col_idx] = np.where(np.isnan(col_data), median_val, col_data)

            # Apply vectorized operations if available
            if 'vectorized_operations' in operations and self.vectorized_core:
                self.logger.debug("⚡ Applying vectorized optimizations...")
                try:
                    # Use vectorized processing for large datasets
                    if hasattr(self.vectorized_core, 'optimize_array'):
                        data = self.vectorized_core.optimize_array(data)
                except Exception as e:
                    self.logger.warning(f"⚠️ Vectorized optimization failed: {e}")

            # Apply parallel processing for large datasets
            if 'parallel_processing' in operations and data.size > 100000:
                self.logger.debug("🔄 Applying parallel processing optimizations...")
                if self.enable_parallel and self.cpu_optimizer:
                    try:
                        # Use parallel processing for large operations
                        if hasattr(self.cpu_optimizer, 'optimize_large_array'):
                            data = self.cpu_optimizer.optimize_large_array(data)
                    except Exception as e:
                        self.logger.warning(f"⚠️ Parallel optimization failed: {e}")

            # Convert back to original format
            if is_dataframe:
                result = pd.DataFrame(data, columns=columns, index=index)
            else:
                result = data

            # Update performance stats
            execution_time = time.time() - start_time
            self.performance_stats['total_operations'] += 1
            self.performance_stats['average_execution_time'] = (
                (self.performance_stats['average_execution_time'] *
                 (self.performance_stats['total_operations'] - 1)) + execution_time
            ) / self.performance_stats['total_operations']

            self.logger.info(f"✅ Dataframe optimization completed in {execution_time:.3f}s")
            return result

        except Exception as e:
            self.logger.error(f"❌ Dataframe optimization failed: {e}")
            # Return original data on failure
            return df

    def calculate_pairwise_similarities(self, feature_vectors: 'np.ndarray', method: str = 'cosine_with_cv_filtering') -> 'np.ndarray':
        """
        Calculate pairwise similarities between feature vectors with M1 optimization.

        Args:
            feature_vectors: Matrix of feature vectors (n_samples, n_features)
            method: Similarity calculation method

        Returns:
            Similarity matrix (n_samples, n_samples)
        """
        if not NUMPY_AVAILABLE:
            raise ImportError("NumPy is required for similarity calculations")

        try:
            self.logger.info(f"🔄 Calculating pairwise similarities using method: {method}")
            start_time = time.time()

            n_samples = feature_vectors.shape[0]

            if method == 'cosine_with_cv_filtering' or method == 'cosine':
                # Normalize feature vectors for cosine similarity
                if self.enable_gpu and self.gpu_manager and n_samples > 100:
                    try:
                        # GPU-accelerated normalization
                        norms = self.gpu_manager.vector_norm(feature_vectors, axis=1, keepdims=True)
                        norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
                        normalized_vectors = self.gpu_manager.divide(feature_vectors, norms)

                        # GPU-accelerated similarity calculation
                        similarity_matrix = self.gpu_manager.matrix_multiply(normalized_vectors, normalized_vectors.T)

                        self.logger.info("🚀 Used GPU acceleration for similarity calculation")
                    except Exception as gpu_error:
                        self.logger.warning(f"⚠️ GPU similarity calculation failed: {gpu_error}, using CPU")
                        # Fallback to CPU
                        norms = np.linalg.norm(feature_vectors, axis=1, keepdims=True)
                        norms[norms == 0] = 1
                        normalized_vectors = feature_vectors / norms
                        similarity_matrix = np.dot(normalized_vectors, normalized_vectors.T)
                else:
                    # CPU calculation
                    norms = np.linalg.norm(feature_vectors, axis=1, keepdims=True)
                    norms[norms == 0] = 1
                    normalized_vectors = feature_vectors / norms
                    similarity_matrix = np.dot(normalized_vectors, normalized_vectors.T)

                # Ensure diagonal is 1.0 and values are in [0, 1]
                np.fill_diagonal(similarity_matrix, 1.0)
                similarity_matrix = np.clip(similarity_matrix, 0.0, 1.0)

            elif method == 'euclidean':
                # Calculate Euclidean distance and convert to similarity
                try:
                    from scipy.spatial.distance import pdist, squareform
                    distances = squareform(pdist(feature_vectors, metric='euclidean'))
                    # Convert distance to similarity (closer = more similar)
                    max_dist = np.max(distances)
                    if max_dist > 0:
                        similarity_matrix = 1.0 - (distances / max_dist)
                    else:
                        similarity_matrix = np.ones_like(distances)
                except ImportError:
                    self.logger.warning("SciPy not available, using manual Euclidean calculation")
                    # Manual calculation
                    similarity_matrix = np.zeros((n_samples, n_samples))
                    for i in range(n_samples):
                        for j in range(n_samples):
                            if i == j:
                                similarity_matrix[i, j] = 1.0
                            else:
                                dist = np.linalg.norm(feature_vectors[i] - feature_vectors[j])
                                similarity_matrix[i, j] = 1.0 / (1.0 + dist)  # Convert to similarity

            else:
                self.logger.warning(f"Unknown similarity method: {method}, using cosine")
                return self.calculate_pairwise_similarities(feature_vectors, 'cosine')

            execution_time = time.time() - start_time
            self.logger.info(f"✅ Similarity matrix calculated in {execution_time:.3f}s: {similarity_matrix.shape}")

            # Update performance stats
            self.performance_stats['total_operations'] += 1
            self.performance_stats['average_execution_time'] = (
                (self.performance_stats['average_execution_time'] *
                 (self.performance_stats['total_operations'] - 1)) + execution_time
            ) / self.performance_stats['total_operations']

            return similarity_matrix

        except Exception as e:
            self.logger.error(f"❌ Similarity calculation failed: {e}")
            # Return identity matrix as fallback
            return np.eye(feature_vectors.shape[0])

    def apply_cv_filtering(self, similarity_matrix: 'np.ndarray', cv_values: 'np.ndarray', max_cv_difference: float = 0.5) -> 'np.ndarray':
        """
        Apply CV (coefficient of variation) filtering to similarity matrix with M1 optimization.

        Args:
            similarity_matrix: Input similarity matrix
            cv_values: CV values for each sample
            max_cv_difference: Maximum allowed CV difference for similarity

        Returns:
            Filtered similarity matrix
        """
        if not NUMPY_AVAILABLE:
            raise ImportError("NumPy is required for CV filtering")

        try:
            self.logger.info(f"🔄 Applying CV filtering with max_cv_difference: {max_cv_difference}")
            start_time = time.time()

            filtered_matrix = similarity_matrix.copy()
            n_samples = len(cv_values)

            if self.enable_gpu and self.gpu_manager and n_samples > 100:
                try:
                    # GPU-accelerated CV filtering
                    cv_diff_matrix = self.gpu_manager.abs(
                        self.gpu_manager.subtract(cv_values.reshape(-1, 1), cv_values.reshape(1, -1))
                    )

                    # Create reduction factor matrix
                    reduction_factors = np.where(
                        cv_diff_matrix > max_cv_difference,
                        np.minimum(cv_diff_matrix / max_cv_difference, 5.0),
                        1.0
                    )

                    # Apply filtering (keep diagonal unchanged)
                    mask = ~np.eye(n_samples, dtype=bool)
                    filtered_matrix[mask] = (similarity_matrix[mask] / reduction_factors[mask])
                    filtered_matrix[mask] = np.maximum(filtered_matrix[mask], 0.01)  # Minimum similarity

                    self.logger.info("🚀 Used GPU acceleration for CV filtering")

                except Exception as gpu_error:
                    self.logger.warning(f"⚠️ GPU CV filtering failed: {gpu_error}, using CPU")
                    # Fallback to CPU
                    for i in range(n_samples):
                        for j in range(n_samples):
                            if i != j:  # Don't modify diagonal
                                cv_diff = abs(cv_values[i] - cv_values[j])
                                if cv_diff > max_cv_difference:
                                    reduction_factor = min(cv_diff / max_cv_difference, 5.0)
                                    filtered_matrix[i, j] *= (1.0 / reduction_factor)
                                    filtered_matrix[i, j] = max(filtered_matrix[i, j], 0.01)
            else:
                # CPU calculation
                for i in range(n_samples):
                    for j in range(n_samples):
                        if i != j:  # Don't modify diagonal
                            cv_diff = abs(cv_values[i] - cv_values[j])
                            if cv_diff > max_cv_difference:
                                reduction_factor = min(cv_diff / max_cv_difference, 5.0)
                                filtered_matrix[i, j] *= (1.0 / reduction_factor)
                                filtered_matrix[i, j] = max(filtered_matrix[i, j], 0.01)

            execution_time = time.time() - start_time
            self.logger.info(f"✅ CV filtering completed in {execution_time:.3f}s")

            # Update performance stats
            self.performance_stats['total_operations'] += 1
            self.performance_stats['average_execution_time'] = (
                (self.performance_stats['average_execution_time'] *
                 (self.performance_stats['total_operations'] - 1)) + execution_time
            ) / self.performance_stats['total_operations']

            return filtered_matrix

        except Exception as e:
            self.logger.error(f"❌ CV filtering failed: {e}")
            # Return original matrix on failure
            return similarity_matrix

    def calculate_regime_stability(self, regime_predictions: 'np.ndarray',
                                  timestamps: 'np.ndarray') -> 'np.ndarray':
        """
        Calculate regime stability scores for each time point.

        Args:
            regime_predictions: Array of regime labels for each time point
            timestamps: Array of timestamps corresponding to regime predictions

        Returns:
            Array of stability scores (0-1, higher is more stable)
        """
        try:
            stability_scores = np.zeros(len(regime_predictions))

            for i in range(len(regime_predictions)):
                current_regime = regime_predictions[i]

                # Look ahead and behind for regime consistency
                lookback = min(10, i)
                lookahead = min(10, len(regime_predictions) - i - 1)

                if lookback > 0:
                    past_regimes = regime_predictions[i-lookback:i]
                    past_consistency = np.mean(past_regimes == current_regime)
                else:
                    past_consistency = 1.0

                if lookahead > 0:
                    future_regimes = regime_predictions[i+1:i+1+lookahead]
                    future_consistency = np.mean(future_regimes == current_regime)
                else:
                    future_consistency = 1.0

                stability_scores[i] = (past_consistency + future_consistency) / 2.0

            return stability_scores

        except Exception as e:
            self.logger.warning(f"Regime stability calculation failed: {e}")
            return np.ones(len(regime_predictions)) * 0.5

    def calculate_transition_probabilities(self, regime_predictions: 'np.ndarray',
                                         n_regimes: int) -> 'np.ndarray':
        """
        Calculate regime transition probability matrix.

        Args:
            regime_predictions: Array of regime labels for each time point
            n_regimes: Number of unique regimes

        Returns:
            Transition probability matrix (n_regimes x n_regimes)
        """
        try:
            transition_matrix = np.zeros((n_regimes, n_regimes))

            for i in range(len(regime_predictions) - 1):
                current_regime = regime_predictions[i]
                next_regime = regime_predictions[i + 1]
                transition_matrix[current_regime, next_regime] += 1

            # Normalize transition matrix
            row_sums = transition_matrix.sum(axis=1)
            transition_matrix = transition_matrix / (row_sums[:, np.newaxis] + 1e-8)

            return transition_matrix

        except Exception as e:
            self.logger.warning(f"Transition probability calculation failed: {e}")
            # Return uniform transition matrix as fallback
            return np.ones((n_regimes, n_regimes)) / n_regimes

    def rolling_operations(self, data: Union['np.ndarray', 'pd.DataFrame'],
                          windows: List[int] = [5, 10, 20, 50],
                          operations: List[str] = ['mean', 'std', 'min', 'max'],
                          **kwargs) -> Union['np.ndarray', 'pd.DataFrame']:
        """
        Perform rolling operations using VectorBTRollingOptimizer.

        Args:
            data: Input data
            windows: List of window sizes
            operations: List of operations to perform
            **kwargs: Additional arguments

        Returns:
            Result with rolling features
        """
        # Try VectorBTRollingOptimizer first if available
        if self.rolling_optimizer and self._should_use_rolling_optimizer(data, windows):
            try:
                result = self.rolling_optimizer.batch_rolling_operations(
                    data, windows=windows, operations=operations, **kwargs
                )
                self.performance_stats['rolling_operations'] = self.performance_stats.get('rolling_operations', 0) + 1
                self.logger.debug("✅ VectorBTRollingOptimizer rolling operations completed")
                return result
            except Exception as e:
                self.logger.warning(f"⚠️ VectorBTRollingOptimizer rolling operations failed: {e}, falling back to standard method")

        # Fallback to standard implementation
        return self._standard_rolling_operations(data, windows, operations, **kwargs)

    def _should_use_rolling_optimizer(self, data: Union['np.ndarray', 'pd.DataFrame'], windows: List[int]) -> bool:
        """Determine if VectorBTRollingOptimizer should be used for rolling operations."""
        if not self.rolling_optimizer:
            return False

        # Convert to numpy array for size check
        if isinstance(data, pd.DataFrame):
            data_array = data.values
        else:
            data_array = data

        # Use VectorBTRollingOptimizer for medium to large datasets
        return data_array.size > 1000  # 1K elements threshold

    def _standard_rolling_operations(self, data: Union['np.ndarray', 'pd.DataFrame'],
                                   windows: List[int], operations: List[str], **kwargs) -> Union['np.ndarray', 'pd.DataFrame']:
        """Standard rolling operations implementation."""
        if isinstance(data, pd.DataFrame):
            result = data.copy()
            is_dataframe = True
        else:
            result = data.copy()
            is_dataframe = False

        # Implement basic rolling operations
        for window in windows:
            for operation in operations:
                if operation == 'mean':
                    if is_dataframe:
                        result[f'rolling_mean_{window}'] = data.rolling(window=window, min_periods=1).mean()
                    else:
                        # For numpy arrays, we'd need to implement rolling manually
                        pass
                elif operation == 'std':
                    if is_dataframe:
                        result[f'rolling_std_{window}'] = data.rolling(window=window, min_periods=1).std()
                elif operation == 'min':
                    if is_dataframe:
                        result[f'rolling_min_{window}'] = data.rolling(window=window, min_periods=1).min()
                elif operation == 'max':
                    if is_dataframe:
                        result[f'rolling_max_{window}'] = data.rolling(window=window, min_periods=1).max()

        return result

# Alias for backward compatibility
M1EnhancedMatrixOperations = UnifiedMatrixOperations

# Factory functions for backward compatibility and easy access
# Global instance cache for singleton pattern
_unified_matrix_operations_instance: Optional[UnifiedMatrixOperations] = None
_unified_matrix_operations_config: Optional[Dict[str, bool]] = None

def get_unified_matrix_operations(enable_gpu: bool = True,
                                enable_memory_optimization: bool = True,
                                enable_parallel: bool = True) -> UnifiedMatrixOperations:
    """
    Factory function to get unified matrix operations instance (singleton pattern).

    Args:
        enable_gpu: Whether to enable GPU acceleration
        enable_memory_optimization: Whether to enable memory optimization
        enable_parallel: Whether to enable parallel processing

    Returns:
        Configured UnifiedMatrixOperations instance (reused if already created)
    """
    global _unified_matrix_operations_instance, _unified_matrix_operations_config

    current_config = {
        'enable_gpu': enable_gpu,
        'enable_memory_optimization': enable_memory_optimization,
        'enable_parallel': enable_parallel
    }

    # Return existing instance if available and configuration matches
    if (_unified_matrix_operations_instance is not None and
        _unified_matrix_operations_config == current_config):
        return _unified_matrix_operations_instance

    # Create new instance if none exists or configuration changed
    _unified_matrix_operations_instance = UnifiedMatrixOperations(
        enable_gpu=enable_gpu,
        enable_memory_optimization=enable_memory_optimization,
        enable_parallel=enable_parallel
    )
    _unified_matrix_operations_config = current_config

    return _unified_matrix_operations_instance

# Legacy compatibility functions (deprecated but maintained)
def get_enhanced_matrix_operations():
    """Legacy function for backward compatibility."""
    logger.warning("⚠️ get_enhanced_matrix_operations() is deprecated. Use get_unified_matrix_operations() instead.")
    return get_unified_matrix_operations()

def m1_matrix_multiply(A: 'np.ndarray', B: 'np.ndarray') -> 'np.ndarray':
    """Legacy M1 matrix multiplication function."""
    logger.warning("⚠️ m1_matrix_multiply() is deprecated. Use UnifiedMatrixOperations.matrix_multiply() instead.")
    ops = get_unified_matrix_operations()
    return ops.matrix_multiply(A, B)

# Convenience functions for common operations
def safe_matrix_multiply(A: 'np.ndarray', B: 'np.ndarray') -> 'np.ndarray':
    """Safe matrix multiplication with validation."""
    ops = get_unified_matrix_operations()
    return ops.matrix_multiply(A, B)

def safe_correlation_matrix(data: Union['np.ndarray', 'pd.DataFrame']) -> 'np.ndarray':
    """Safe correlation matrix computation."""
    ops = get_unified_matrix_operations()
    return ops.safe_correlation_matrix(data)

def safe_matrix_inverse(matrix: 'np.ndarray') -> 'np.ndarray':
    """Safe matrix inversion."""
    ops = get_unified_matrix_operations()
    return ops.matrix_inverse(matrix)

if __name__ == "__main__":
    # Example usage
    print("🚀 Unified Matrix Operations Demo")
    print("="*50)

    # Create sample matrices
    np.random.seed(42)
    A = np.random.randn(500, 500)
    B = np.random.randn(500, 500)
    data = np.random.randn(1000, 10)

    # Initialize operations
    ops = get_unified_matrix_operations()

    # Test matrix multiplication
    print("\n🧮 Testing Matrix Multiplication...")
    result = ops.matrix_multiply(A, B)
    print(f"✅ Matrix multiplication: {result.shape}")

    # Test correlation matrix
    print("\n📊 Testing Correlation Matrix...")
    corr = ops.safe_correlation_matrix(data)
    print(f"✅ Correlation matrix: {corr.shape}")

    # Test memory optimization
    print("\n🧠 Testing Memory Optimization...")
    memory_stats = ops.optimize_memory_usage()
    print(f"✅ Memory optimization: {memory_stats}")

    # Print performance stats
    print("\n📈 Performance Statistics:")
    stats = ops.get_performance_stats()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")

    print("\n🎉 Unified Matrix Operations Demo Complete!")
    print("All optimizations are focused on Apple Silicon M1/M2/M3 Macs")
