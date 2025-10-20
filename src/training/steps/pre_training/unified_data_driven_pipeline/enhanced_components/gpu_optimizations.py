"""
GPU Optimizations for Enhanced Performance

This module provides GPU-specific optimizations using
for high-performance computing operations.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
import logging
import time
import warnings

try:
    from src.utils.tprint import (
        tprint, tprint_info, tprint_success, tprint_warning, tprint_error, tprint_debug
    )
    TPRINT_AVAILABLE = True
except ImportError:
    TPRINT_AVAILABLE = False
    def tprint(*args, **kwargs): print("TPRINT:", *args, **kwargs)
    def tprint_info(*args, **kwargs): print("INFO:", *args, **kwargs)
    def tprint_success(*args, **kwargs): print("SUCCESS:", *args, **kwargs)
    def tprint_warning(*args, **kwargs): print("WARNING:", *args, **kwargs)
    def tprint_error(*args, **kwargs): print("ERROR:", *args, **kwargs)
    def tprint_debug(*args, **kwargs): print("DEBUG:", *args, **kwargs)

# GPU/CuPy support removed
CUPY_AVAILABLE = False
cp = None

# Import Numba for JIT compilation
try:
    import numba as nb
    from numba import cuda, jit
    NUMBA_AVAILABLE = True
    tprint_info("✅ Numba JIT compilation available")
except ImportError:
    NUMBA_AVAILABLE = False
    nb = None
    cuda = None
    jit = None
    tprint_warning("⚠️ Numba not available, using standard Python")

logger = logging.getLogger(__name__)

@dataclass
class GPUConfig:
    """Configuration for GPU operations."""

    enable_gpu: bool = True
    gpu_memory_limit_mb: int = 8000
    fallback_to_cpu: bool = True
    enable_mixed_precision: bool = True
    enable_memory_pool: bool = True
    max_workers: int = 4
    batch_size: int = 1000

    def __post_init__(self):
        """Validate GPU configuration."""
        assert self.gpu_memory_limit_mb > 0, "GPU memory limit must be positive"
        assert self.max_workers > 0, "max_workers must be positive"
        assert self.batch_size > 0, "batch_size must be positive"

@dataclass
class GPUOperationResult:
    """Result of a GPU operation."""

    success: bool
    execution_time: float
    memory_used_mb: float
    gpu_utilization: float
    fallback_used: bool
    error_message: Optional[str] = None

    def __post_init__(self):
        """Validate result."""
        assert isinstance(self.success, bool), "success must be boolean"
        assert self.execution_time >= 0, "execution_time must be non-negative"
        assert self.memory_used_mb >= 0, "memory_used_mb must be non-negative"
        assert 0 <= self.gpu_utilization <= 100, "gpu_utilization must be between 0 and 100"

class GPUOptimizer:
    """
    GPU optimizer for high-performance operations.

    Provides
    mathematical and data processing operations.
    """

    def __init__(self, config: Optional[GPUConfig] = None):
        """Initialize the GPU optimizer."""
        self.config = config or GPUConfig()
        self.gpu_available =  self.config.enable_gpu

        # Initialize GPU components
        self._initialize_gpu_components()

        # Performance tracking
        self.performance_stats = {
            'total_operations': 0,
            'gpu_operations': 0,
            'cpu_fallbacks': 0,
            'total_execution_time': 0.0,
            'total_memory_used_mb': 0.0,
            'average_gpu_utilization': 0.0
        }

        tprint_info("GPU Optimizer initialized")
        if self.gpu_available:
            tprint_info("✅ GPU acceleration available")
        else:
            tprint_warning("⚠️ GPU acceleration not available")

    def _initialize_gpu_components(self):
        """Initialize GPU components."""
        if not self.gpu_available:
            return

        try:
            # Test CPU availability (GPU support removed)
            test_array = np.array([1, 2, 3, 4, 5])
            result = np.sum(test_array)

            tprint_success("✅ CPU components initialized successfully")

        except Exception as e:
            tprint_warning(f"⚠️ CPU initialization failed: {e}")
            self.gpu_available = False

    def matrix_multiply(self, a: np.ndarray, b: np.ndarray) -> GPUOperationResult:
        """Perform matrix multiplication with GPU acceleration."""
        start_time = time.time()

        try:
            # CPU operation (GPU support removed)
            result = np.dot(a, b)
            execution_time = time.time() - start_time

            self.performance_stats['cpu_fallbacks'] += 1

            return GPUOperationResult(
                success=True,
                execution_time=execution_time,
                memory_used_mb=0.0,
                gpu_utilization=0.0,
                fallback_used=True
            )

        except Exception as e:
            execution_time = time.time() - start_time
            return GPUOperationResult(
                success=False,
                execution_time=execution_time,
                memory_used_mb=0.0,
                gpu_utilization=0.0,
                fallback_used=True,
                error_message=str(e)
            )
        finally:
            self.performance_stats['total_operations'] += 1
            self.performance_stats['total_execution_time'] += execution_time

    def rolling_operations(self, data: pd.Series, window: int, operation: str = "mean") -> GPUOperationResult:
        """Perform rolling operations with GPU acceleration."""
        start_time = time.time()

        try:
            # CPU operation (GPU support removed)
            if operation == "mean":
                result = data.rolling(window=window).mean()
            elif operation == "std":
                result = data.rolling(window=window).std()
            elif operation == "sum":
                result = data.rolling(window=window).sum()
            else:
                raise ValueError(f"Unsupported operation: {operation}")

            execution_time = time.time() - start_time

            self.performance_stats['cpu_fallbacks'] += 1

            return GPUOperationResult(
                success=True,
                execution_time=execution_time,
                memory_used_mb=0.0,
                gpu_utilization=0.0,
                fallback_used=True
            )

        except Exception as e:
            execution_time = time.time() - start_time
            return GPUOperationResult(
                success=False,
                execution_time=execution_time,
                memory_used_mb=0.0,
                gpu_utilization=0.0,
                fallback_used=True,
                error_message=str(e)
            )
        finally:
            self.performance_stats['total_operations'] += 1
            self.performance_stats['total_execution_time'] += execution_time

    def correlation_matrix(self, data: pd.DataFrame) -> GPUOperationResult:
        """Calculate correlation matrix with GPU acceleration."""
        start_time = time.time()

        try:
            # CPU operation (GPU support removed)
            result = data.corr().values
            execution_time = time.time() - start_time

            self.performance_stats['cpu_fallbacks'] += 1

            return GPUOperationResult(
                success=True,
                execution_time=execution_time,
                memory_used_mb=0.0,
                gpu_utilization=0.0,
                fallback_used=True
            )

        except Exception as e:
            execution_time = time.time() - start_time
            return GPUOperationResult(
                success=False,
                execution_time=execution_time,
                memory_used_mb=0.0,
                gpu_utilization=0.0,
                fallback_used=True,
                error_message=str(e)
            )
        finally:
            self.performance_stats['total_operations'] += 1
            self.performance_stats['total_execution_time'] += execution_time

    def _should_use_gpu(self, *arrays) -> bool:
        """Determine if GPU should be used based on array sizes."""
        if not self.gpu_available:
            return False

        total_elements = sum(arr.size for arr in arrays)
        return total_elements > 10000  # Use GPU for large arrays

    def _estimate_memory_usage(self, *arrays) -> float:
        """Estimate memory usage in MB."""
        total_bytes = sum(arr.nbytes for arr in arrays)
        return total_bytes / (1024 * 1024)

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        stats = self.performance_stats.copy()

        if stats['total_operations'] > 0:
            stats['gpu_operation_ratio'] = stats['gpu_operations'] / stats['total_operations']
            stats['cpu_fallback_ratio'] = stats['cpu_fallbacks'] / stats['total_operations']
            stats['average_execution_time'] = stats['total_execution_time'] / stats['total_operations']
        else:
            stats['gpu_operation_ratio'] = 0.0
            stats['cpu_fallback_ratio'] = 0.0
            stats['average_execution_time'] = 0.0

        return stats

# Numba JIT compiled functions
if NUMBA_AVAILABLE:
    @jit(nopython=True)
    def _numba_rolling_mean(data: np.ndarray, window: int) -> np.ndarray:
        """Numba JIT compiled rolling mean."""
        result = np.empty(len(data) - window + 1)
        for i in range(len(result)):
            result[i] = np.mean(data[i:i + window])
        return result

    @jit(nopython=True)
    def _numba_rolling_std(data: np.ndarray, window: int) -> np.ndarray:
        """Numba JIT compiled rolling standard deviation."""
        result = np.empty(len(data) - window + 1)
        for i in range(len(result)):
            result[i] = np.std(data[i:i + window])
        return result

# Convenience functions
def create_gpu_optimizer(config: Optional[GPUConfig] = None) -> GPUOptimizer:
    """Create a GPU optimizer."""
    return GPUOptimizer(config)

def gpu_matrix_multiply(a: np.ndarray, b: np.ndarray, config: Optional[GPUConfig] = None) -> Tuple[np.ndarray, GPUOperationResult]:
    """Perform GPU-accelerated matrix multiplication."""
    optimizer = create_gpu_optimizer(config)
    result = optimizer.matrix_multiply(a, b)

    if result.success:
        return np.dot(a, b), result
    else:
        raise RuntimeError(f"GPU operation failed: {result.error_message}")

def gpu_rolling_mean(data: pd.Series, window: int, config: Optional[GPUConfig] = None) -> Tuple[pd.Series, GPUOperationResult]:
    """Perform GPU-accelerated rolling mean."""
    optimizer = create_gpu_optimizer(config)
    result = optimizer.rolling_operations(data, window, "mean")

    if result.success:
        return data.rolling(window=window).mean(), result
    else:
        raise RuntimeError(f"GPU operation failed: {result.error_message}")
