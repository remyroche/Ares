"""
Enhanced Matrix Operations - GPU Accelerated Implementation

This module consolidates enhanced matrix operations with GPU acceleration
from scattered sources into a single, unified interface.
"""

import logging
from contextlib import contextmanager
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
import traceback
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
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

try:
    from scipy import linalg, sparse
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    linalg = None
    sparse = None

logger = logging.getLogger(__name__)

class BatchOptimizationStrategy(Enum):
    """Batch size optimization strategies."""
    ADAPTIVE = "adaptive"
    MEMORY_BASED = "memory_based"
    PERFORMANCE_BASED = "performance_based"
    HYBRID = "hybrid"

class OperationComplexity(Enum):
    """Matrix operation complexity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"

@dataclass
class BatchOptimizationMetrics:
    """Metrics for batch optimization."""
    operation_name: str
    batch_size: int
    execution_time: float
    memory_usage: float
    throughput: float
    efficiency_score: float
    timestamp: float = field(default_factory=time.time)

@dataclass
class DynamicBatchOptimizer:
    """Dynamic batch size optimizer for matrix operations."""

    def __init__(self, max_batch_size: int = 10000, min_batch_size: int = 100,
                 optimization_strategy: BatchOptimizationStrategy = BatchOptimizationStrategy.HYBRID,
                 enable_learning: bool = True):
        """Initialize dynamic batch optimizer."""
        self.max_batch_size = max_batch_size
        self.min_batch_size = min_batch_size
        self.optimization_strategy = optimization_strategy
        self.enable_learning = enable_learning

        # Performance history
        self.performance_history: deque = deque(maxlen=100)
        self.operation_profiles: Dict[str, Dict[str, Any]] = {}

        # Current optimization state
        self.current_batch_sizes: Dict[str, int] = {}
        self.adaptation_factors: Dict[str, float] = {}

        self.logger = logging.getLogger(f"{__name__}.DynamicBatchOptimizer")
        self.logger.info(f"🔧 Dynamic Batch Optimizer initialized (strategy: {optimization_strategy.value})")

    def optimize_batch_size(self, operation_name: str, data_shape: Tuple[int, ...],
                          operation_complexity: OperationComplexity = OperationComplexity.MEDIUM,
                          available_memory_mb: Optional[float] = None) -> int:
        """Determine optimal batch size for an operation."""
        total_elements = np.prod(data_shape)

        # Start with baseline batch size
        baseline_size = self._calculate_baseline_batch_size(data_shape, operation_complexity)

        # Apply optimization strategy
        if self.optimization_strategy == BatchOptimizationStrategy.ADAPTIVE:
            optimal_size = self._adaptive_batch_optimization(operation_name, baseline_size, total_elements)
        elif self.optimization_strategy == BatchOptimizationStrategy.MEMORY_BASED:
            optimal_size = self._memory_based_optimization(operation_name, baseline_size, available_memory_mb)
        elif self.optimization_strategy == BatchOptimizationStrategy.PERFORMANCE_BASED:
            optimal_size = self._performance_based_optimization(operation_name, baseline_size, total_elements)
        else:  # HYBRID
            optimal_size = self._hybrid_optimization(operation_name, baseline_size, total_elements, available_memory_mb)

        # Apply bounds and learning
        optimal_size = self._apply_bounds_and_learning(operation_name, optimal_size, total_elements)

        self.logger.debug(f"📏 Optimal batch size for {operation_name}: {optimal_size} (baseline: {baseline_size})")
        return optimal_size

    def _calculate_baseline_batch_size(self, data_shape: Tuple[int, ...],
                                     complexity: OperationComplexity) -> int:
        """Calculate baseline batch size based on data shape and complexity."""
        total_elements = np.prod(data_shape)

        # Complexity factors
        complexity_factors = {
            OperationComplexity.LOW: 1.0,
            OperationComplexity.MEDIUM: 0.7,
            OperationComplexity.HIGH: 0.4,
            OperationComplexity.VERY_HIGH: 0.2
        }

        factor = complexity_factors.get(complexity, 0.5)

        # Base batch size calculation
        if len(data_shape) == 1:
            # Vector operations
            baseline = min(int(total_elements * factor), self.max_batch_size)
        elif len(data_shape) == 2:
            # Matrix operations
            rows, cols = data_shape
            # Prefer processing complete rows when possible
            baseline = min(rows, int(self.max_batch_size / cols) if cols > 0 else self.max_batch_size)
        else:
            # Higher dimensional operations
            baseline = min(int(np.power(total_elements, 1/3) * factor), self.max_batch_size)

        return max(self.min_batch_size, baseline)

    def _adaptive_batch_optimization(self, operation_name: str, baseline_size: int, total_elements: int) -> int:
        """Adaptive batch optimization based on historical performance."""
        if not self.enable_learning or operation_name not in self.operation_profiles:
            return baseline_size

        profile = self.operation_profiles[operation_name]

        # Use historical performance to adjust batch size
        historical_efficiency = profile.get('average_efficiency', 0.5)
        historical_throughput = profile.get('average_throughput', 1.0)

        # Adjust based on efficiency
        if historical_efficiency > 0.8:
            # Very efficient - can increase batch size
            adjustment_factor = 1.2
        elif historical_efficiency < 0.4:
            # Inefficient - reduce batch size
            adjustment_factor = 0.8
        else:
            adjustment_factor = 1.0

        return int(baseline_size * adjustment_factor)

    def _memory_based_optimization(self, operation_name: str, baseline_size: int,
                                 available_memory_mb: Optional[float]) -> int:
        """Memory-based batch size optimization."""
        if available_memory_mb is None:
            return baseline_size

        # Estimate memory per element (rough approximation)
        memory_per_element_mb = 8 / (1024 * 1024)  # Assume float64

        # Calculate safe batch size based on available memory
        safe_batch_size = int((available_memory_mb * 0.7) / memory_per_element_mb)  # Use 70% of available memory

        # Adjust for operation overhead
        if operation_name in ['matrix_multiply', 'svd', 'eigendecomposition']:
            safe_batch_size = int(safe_batch_size * 0.5)  # More conservative for complex operations

        return min(baseline_size, safe_batch_size, self.max_batch_size)

    def _performance_based_optimization(self, operation_name: str, baseline_size: int, total_elements: int) -> int:
        """Performance-based batch size optimization."""
        if not self.enable_learning:
            return baseline_size

        # Analyze recent performance history
        recent_metrics = [m for m in self.performance_history if m.operation_name == operation_name][-10:]

        if len(recent_metrics) < 3:
            return baseline_size

        # Calculate average throughput and efficiency
        avg_throughput = np.mean([m.throughput for m in recent_metrics])
        avg_efficiency = np.mean([m.efficiency_score for m in recent_metrics])

        # Adjust batch size based on performance
        if avg_efficiency > 0.8 and avg_throughput > np.median([m.throughput for m in recent_metrics]):
            # High performance - slight increase
            return int(baseline_size * 1.1)
        elif avg_efficiency < 0.5:
            # Low performance - reduce batch size
            return int(baseline_size * 0.9)
        else:
            return baseline_size

    def _hybrid_optimization(self, operation_name: str, baseline_size: int, total_elements: int,
                           available_memory_mb: Optional[float]) -> int:
        """Hybrid optimization combining multiple strategies."""
        # Combine adaptive and memory-based optimization
        adaptive_size = self._adaptive_batch_optimization(operation_name, baseline_size, total_elements)
        memory_size = self._memory_based_optimization(operation_name, baseline_size, available_memory_mb)

        # Take the more conservative of the two
        conservative_size = min(adaptive_size, memory_size)

        # Apply performance-based fine-tuning
        performance_size = self._performance_based_optimization(operation_name, conservative_size, total_elements)

        return performance_size

    def _apply_bounds_and_learning(self, operation_name: str, optimal_size: int, total_elements: int) -> int:
        """Apply bounds checking and learning updates."""
        # Apply bounds
        optimal_size = max(self.min_batch_size, min(optimal_size, self.max_batch_size))

        # Ensure batch size doesn't exceed total elements
        optimal_size = min(optimal_size, total_elements)

        # Update current batch size
        self.current_batch_sizes[operation_name] = optimal_size

        return optimal_size

    def record_performance(self, operation_name: str, batch_size: int, execution_time: float,
                          memory_usage: float, data_processed: int):
        """Record performance metrics for learning."""
        throughput = data_processed / execution_time if execution_time > 0 else 0

        # Calculate efficiency score (0-1, higher is better)
        # This is a simplified efficiency calculation
        efficiency = min(1.0, throughput / (batch_size * 1000))  # Normalize by batch size

        metrics = BatchOptimizationMetrics(
            operation_name=operation_name,
            batch_size=batch_size,
            execution_time=execution_time,
            memory_usage=memory_usage,
            throughput=throughput,
            efficiency_score=efficiency
        )

        self.performance_history.append(metrics)

        # Update operation profile
        if operation_name not in self.operation_profiles:
            self.operation_profiles[operation_name] = {
                'total_executions': 0,
                'average_efficiency': 0.0,
                'average_throughput': 0.0,
                'best_batch_size': batch_size,
                'best_efficiency': efficiency
            }

        profile = self.operation_profiles[operation_name]
        profile['total_executions'] += 1

        # Update running averages
        alpha = 0.1  # Learning rate
        profile['average_efficiency'] = (
            profile['average_efficiency'] * (1 - alpha) + efficiency * alpha
        )
        profile['average_throughput'] = (
            profile['average_throughput'] * (1 - alpha) + throughput * alpha
        )

        # Update best batch size
        if efficiency > profile['best_efficiency']:
            profile['best_batch_size'] = batch_size
            profile['best_efficiency'] = efficiency

    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization statistics and recommendations."""
        return {
            'total_operations_optimized': len(self.operation_profiles),
            'performance_history_size': len(self.performance_history),
            'current_batch_sizes': self.current_batch_sizes.copy(),
            'operation_profiles': self.operation_profiles.copy(),
            'optimization_strategy': self.optimization_strategy.value,
            'learning_enabled': self.enable_learning
        }

    def reset_learning(self):
        """Reset learning state."""
        self.performance_history.clear()
        self.operation_profiles.clear()
        self.current_batch_sizes.clear()
        self.adaptation_factors.clear()
        self.logger.info("🔄 Batch optimizer learning state reset")

class EnhancedMatrixOperations:
    """Enhanced matrix operations with GPU acceleration and memory optimization."""

    def __init__(self, use_gpu: bool = True, memory_efficient: bool = True,
                 chunk_size: int = 10000, dtype: 'torch.dtype' = None,
                 enable_dynamic_batch: bool = True):
        """Initialize enhanced matrix operations."""
        self.use_gpu = use_gpu
        self.memory_efficient = memory_efficient
        self.chunk_size = chunk_size
        # Set default dtype
        if dtype is None:
            self.dtype = torch.float32 if torch is not None else float
        else:
            self.dtype = dtype
        self.enable_dynamic_batch = enable_dynamic_batch

        # Initialize optimization components
        self._init_accelerators()

        # Initialize dynamic batch optimizer
        if self.enable_dynamic_batch:
            self.batch_optimizer = DynamicBatchOptimizer(
                max_batch_size=self.chunk_size,
                optimization_strategy=BatchOptimizationStrategy.HYBRID
            )
        else:
            self.batch_optimizer = None

        self.logger = logger.getChild('EnhancedMatrixOperations')
        self.logger.info(f"🔧 Enhanced Matrix Operations initialized (GPU: {self.use_gpu}, Dynamic Batch: {self.enable_dynamic_batch})")

    def _init_accelerators(self):
        """Initialize GPU and optimization accelerators."""
        try:
            from ..hardware.m1_gpu_utils import get_m1_gpu_manager
            from ..hardware.m1_memory_optimizer import get_m1_memory_optimizer
            from .vectorized_core import get_vectorized_processing_core

            self.gpu_manager = get_integrated_hardware_manager() if self.use_gpu else None
            self.memory_optimizer = get_integrated_hardware_manager()
            self.vectorized_core = get_vectorized_processing_core()

        except ImportError:
            self.gpu_manager = None
            self.memory_optimizer = None
            self.vectorized_core = None
            self.logger.warning("⚠️ Optimization components not available, using CPU fallback")

    @contextmanager
    def matrix_operation_context(self, operation_name: str = "matrix_op"):
        """Context manager for matrix operations with memory management."""
        start_time = time.time()

        if self.memory_optimizer:
            with self.memory_optimizer.memory_checkpoint(operation_name):
                try:
                    yield
                finally:
                    self._log_operation_performance(operation_name, start_time)
        else:
            try:
                yield
            finally:
                self._log_operation_performance(operation_name, start_time)

    def _log_operation_performance(self, operation_name: str, start_time: float):
        """Log matrix operation performance."""
        execution_time = time.time() - start_time
        # Keep concise INFO for visibility; detailed timing at DEBUG
        if execution_time >= 0.2:
            self.logger.info(f"✅ {operation_name} completed in {execution_time*1000:.0f} ms")
        else:
            self.logger.debug(f"⚡ {operation_name} completed in {execution_time:.4f}s")

    def matrix_multiply(self, a: Union['np.ndarray', 'torch.Tensor'],
                       b: Union['np.ndarray', 'torch.Tensor'],
                       use_gpu: Optional[bool] = None) -> Union['np.ndarray', 'torch.Tensor']:
        """Optimized matrix multiplication."""
        use_gpu = use_gpu if use_gpu is not None else self.use_gpu

        with self.matrix_operation_context("matrix_multiply"):
            # Convert to tensors
            a_tensor = self.to_tensor(a) if not isinstance(a, torch.Tensor) else a
            b_tensor = self.to_tensor(b) if not isinstance(b, torch.Tensor) else b

            # Perform multiplication
            if use_gpu and self.gpu_manager:
                result = self.gpu_manager.matrix_multiply_mps(a_tensor, b_tensor)
            else:
                result = torch.matmul(a_tensor, b_tensor)

            # Convert back to numpy if input was numpy
            if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
                return result.cpu().numpy()
            return result

    def correlation_matrix(self, data: Union['pd.DataFrame', 'np.ndarray']) -> 'np.ndarray':
        """Compute correlation matrix with GPU acceleration."""
        with self.matrix_operation_context("correlation_matrix"):
            # Convert to numpy array if needed
            if isinstance(data, pd.DataFrame):
                data_array = data.values
            else:
                data_array = data

            # Compute correlation matrix
            try:
                # Use GPU acceleration if available
                if self.use_gpu and self.gpu_manager:
                    # Use numpy correlation for now (can be enhanced with GPU later)
                    correlation = np.corrcoef(data_array.T)
                else:
                    correlation = np.corrcoef(data_array.T)

                return correlation

            except Exception as e:
                self.logger.warning(f"GPU correlation failed: {e}, using CPU fallback")
                # Fallback to numpy correlation
                return np.corrcoef(data_array.T)

    def to_tensor(self, data: Union['np.ndarray', 'pd.DataFrame', List],
                 dtype: Optional['torch.dtype'] = None) -> 'torch.Tensor':
        """Convert data to tensor with optimization."""
        if dtype is None:
            dtype = self.dtype

        # Convert to numpy first if needed
        if isinstance(data, pd.DataFrame):
            data = data.values.astype(np.float32 if np is not None else float)
        elif isinstance(data, list):
            data = np.array(data, dtype=np.float32 if np is not None else float)
        elif isinstance(data, np.ndarray) and hasattr(data, 'dtype') and data.dtype != (np.float32 if np is not None else float):
            data = data.astype(np.float32 if np is not None else float)

        # Create tensor
        tensor = torch.from_numpy(data).to(dtype)

        # Move to GPU if available and requested
        if self.use_gpu and self.gpu_manager:
            tensor = self.gpu_manager.to_device(tensor, "matrix_mult")

        return tensor

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for matrix operations."""
        stats = {
            'gpu_enabled': self.use_gpu and self.gpu_manager is not None,
            'memory_efficient': self.memory_efficient,
            'chunk_size': self.chunk_size,
            'dtype': str(self.dtype),
            'dynamic_batch_enabled': self.enable_dynamic_batch
        }

        if self.gpu_manager:
            stats['gpu_device'] = str(self.gpu_manager.device)
            stats['gpu_memory_available'] = self.gpu_manager.memory_info.get('available_gb', 0)

        if self.memory_optimizer:
            stats['memory_stats'] = self.memory_optimizer.get_memory_report()

        if self.batch_optimizer:
            stats['batch_optimization_stats'] = self.batch_optimizer.get_optimization_stats()

        return stats

# Global instance
_enhanced_matrix_ops = None

def get_enhanced_matrix_operations() -> EnhancedMatrixOperations:
    """Get global enhanced matrix operations instance."""
    global _enhanced_matrix_ops
    if _enhanced_matrix_ops is None:
        _enhanced_matrix_ops = EnhancedMatrixOperations()
    return _enhanced_matrix_ops

# Convenience functions
def gpu_matrix_multiply(a: 'np.ndarray', b: 'np.ndarray') -> 'np.ndarray':
    """GPU-accelerated matrix multiplication."""
    ops = get_enhanced_matrix_operations()
    return ops.matrix_multiply(a, b)

def correlation_matrix_gpu(data: Union['pd.DataFrame', 'np.ndarray']) -> 'np.ndarray':
    """GPU-accelerated correlation matrix."""
    ops = get_enhanced_matrix_operations()
    return ops.correlation_matrix(data)

def eigendecomposition_gpu(matrix: 'np.ndarray') -> Tuple['np.ndarray', 'np.ndarray']:
    """GPU-accelerated eigendecomposition."""
    ops = get_enhanced_matrix_operations()
    return ops.eigendecomposition(matrix)

def svd_gpu(matrix: 'np.ndarray', k: Optional[int] = None) -> Tuple['np.ndarray', 'np.ndarray', 'np.ndarray']:
    """GPU-accelerated SVD."""
    ops = get_enhanced_matrix_operations()
    return ops.svd_decomposition(matrix, k)

def optimize_batch_size(operation_name: str, data_shape: Tuple[int, ...],
                       complexity: OperationComplexity = OperationComplexity.MEDIUM,
                       available_memory_mb: Optional[float] = None) -> int:
    """Optimize batch size for matrix operations."""
    ops = get_enhanced_matrix_operations()
    if ops.batch_optimizer:
        return ops.batch_optimizer.optimize_batch_size(
            operation_name, data_shape, complexity, available_memory_mb
        )
    else:
        # Fallback to simple calculation
        return min(10000, np.prod(data_shape))

def record_batch_performance(operation_name: str, batch_size: int, execution_time: float,
                           memory_usage: float, data_processed: int):
    """Record performance metrics for batch optimization learning."""
    ops = get_enhanced_matrix_operations()
    if ops.batch_optimizer:
        ops.batch_optimizer.record_performance(
            operation_name, batch_size, execution_time, memory_usage, data_processed
        )

def get_batch_optimization_stats() -> Dict[str, Any]:
    """Get batch optimization statistics."""
    ops = get_enhanced_matrix_operations()
    if ops.batch_optimizer:
        return ops.batch_optimizer.get_optimization_stats()
    return {}

# Custom matrix operations support
class CustomMatrixOperation:
    """Base class for custom matrix operations."""

    def __init__(self, name: str, operation_func: Callable, **kwargs):
        self.name = name
        self.operation_func = operation_func
        self.kwargs = kwargs
        self.performance_stats = []

    def execute(self, *args, **kwargs) -> Any:
        """Execute the custom operation."""
        start_time = time.time()
        try:
            result = self.operation_func(*args, **kwargs)
            execution_time = time.time() - start_time
            self.performance_stats.append({
                'execution_time': execution_time,
                'success': True,
                'timestamp': time.time()
            })
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            self.performance_stats.append({
                'execution_time': execution_time,
                'success': False,
                'error': str(e),
                'timestamp': time.time()
            })
            raise e

    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        if not self.performance_stats:
            return {'total_executions': 0}

        successful = [s for s in self.performance_stats if s['success']]
        return {
            'total_executions': len(self.performance_stats),
            'successful_executions': len(successful),
            'failed_executions': len(self.performance_stats) - len(successful),
            'average_execution_time': np.mean([s['execution_time'] for s in successful]) if successful else 0,
            'success_rate': len(successful) / len(self.performance_stats) if self.performance_stats else 0
        }

class CustomMatrixOperationsRegistry:
    """Registry for custom matrix operations."""

    def __init__(self):
        self.operations: Dict[str, CustomMatrixOperation] = {}
        self.logger = logging.getLogger(f"{__name__}.CustomMatrixOperationsRegistry")
        self._registration_guard = set()  # Track registered operations to prevent duplicates

    def register_operation(self, name: str, operation_func: Callable, **kwargs) -> None:
        """Register a custom matrix operation."""
        if name in self.operations:
            self.logger.debug(f"Operation '{name}' already registered, skipping duplicate registration")
            return

        self.operations[name] = CustomMatrixOperation(name, operation_func, **kwargs)
        self._registration_guard.add(name)
        self.logger.info(f"📝 Registered custom matrix operation: {name}")

    def get_operation(self, name: str) -> Optional[CustomMatrixOperation]:
        """Get a registered operation by name."""
        return self.operations.get(name)

    def list_operations(self) -> List[str]:
        """List all registered operations."""
        return list(self.operations.keys())

    def execute_operation(self, name: str, *args, **kwargs) -> Any:
        """Execute a registered operation."""
        operation = self.get_operation(name)
        if operation is None:
            raise ValueError(f"Operation '{name}' not found")

        return operation.execute(*args, **kwargs)

    def get_operation_stats(self, name: str) -> Dict[str, Any]:
        """Get statistics for a specific operation."""
        operation = self.get_operation(name)
        if operation is None:
            return {'error': f"Operation '{name}' not found"}

        return operation.get_stats()

    def get_registry_stats(self) -> Dict[str, Any]:
        """Get statistics for the entire registry."""
        return {
            'total_operations': len(self.operations),
            'operation_names': self.list_operations(),
            'operations_stats': {
                name: op.get_stats() for name, op in self.operations.items()
            }
        }

# Global registry instance
_custom_ops_registry = None

def get_custom_operations_registry() -> CustomMatrixOperationsRegistry:
    """Get global custom operations registry."""
    global _custom_ops_registry
    if _custom_ops_registry is None:
        _custom_ops_registry = CustomMatrixOperationsRegistry()
    return _custom_ops_registry

def register_custom_matrix_operation(name: str, operation_func: Callable, **kwargs) -> None:
    """Register a custom matrix operation."""
    registry = get_custom_operations_registry()
    registry.register_operation(name, operation_func, **kwargs)

def execute_custom_matrix_operation(name: str, *args, **kwargs) -> Any:
    """Execute a custom matrix operation."""
    registry = get_custom_operations_registry()
    return registry.execute_operation(name, *args, **kwargs)

def list_custom_matrix_operations() -> List[str]:
    """List all registered custom matrix operations."""
    registry = get_custom_operations_registry()
    return registry.list_operations()

# Example custom operations
_default_operations_registered = False

def register_default_custom_operations():
    """Register some useful default custom matrix operations."""
    global _default_operations_registered

    if _default_operations_registered:
        return  # Already registered, skip

    _default_operations_registered = True

    # Matrix condition number with GPU support
    def gpu_condition_number(matrix: 'np.ndarray') -> float:
        """Compute matrix condition number with GPU acceleration."""
        ops = get_enhanced_matrix_operations()
        return ops.condition_number(matrix)

    # Matrix rank with GPU support
    def gpu_matrix_rank(matrix: 'np.ndarray', tol: Optional[float] = None) -> int:
        """Compute matrix rank with GPU acceleration."""
        ops = get_enhanced_matrix_operations()
        # Use SVD to compute rank
        if ops.use_gpu and ops.gpu_manager:
            U, S, V = ops.svd_decomposition(matrix, k=min(matrix.shape))
            if tol is None:
                try:
                    tol = S.max() * max(matrix.shape) * np.finfo(S.dtype).eps
                except (AttributeError, NameError):
                    # Fallback if numpy is not available
                    tol = 1e-15
            rank = np.sum(S > tol)
        else:
            rank = np.linalg.matrix_rank(matrix, tol=tol)
        return int(rank)

    # Custom matrix normalization
    def matrix_normalize(matrix: 'np.ndarray', method: str = 'l2', axis: int = 0) -> 'np.ndarray':
        """Normalize matrix using various methods."""
        if method == 'l2':
            norms = np.linalg.norm(matrix, ord=2, axis=axis, keepdims=True)
            norms[norms == 0] = 1  # Avoid division by zero
            return matrix / norms
        elif method == 'l1':
            norms = np.linalg.norm(matrix, ord=1, axis=axis, keepdims=True)
            norms[norms == 0] = 1
            return matrix / norms
        elif method == 'max':
            norms = np.max(np.abs(matrix), axis=axis, keepdims=True)
            norms[norms == 0] = 1
            return matrix / norms
        else:
            raise ValueError(f"Unsupported normalization method: {method}")

    # Custom matrix feature extraction
    def extract_matrix_features(matrix: 'np.ndarray') -> Dict[str, float]:
        """Extract various features from a matrix."""
        features = {}

        # Basic statistics
        features['mean'] = np.mean(matrix)
        features['std'] = np.std(matrix)
        features['min'] = np.min(matrix)
        features['max'] = np.max(matrix)

        # Matrix properties
        features['condition_number'] = gpu_condition_number(matrix)
        features['rank'] = gpu_matrix_rank(matrix)
        features['determinant'] = np.linalg.det(matrix) if matrix.shape[0] == matrix.shape[1] else 0
        features['trace'] = np.trace(matrix) if matrix.shape[0] == matrix.shape[1] else np.sum(np.diag(matrix))

        # Spectral properties
        if matrix.shape[0] == matrix.shape[1]:
            eigenvals = np.linalg.eigvals(matrix)
            features['eigenvalues_real_mean'] = np.mean(eigenvals.real)
            features['eigenvalues_imag_mean'] = np.mean(eigenvals.imag)

        return features

    # Register the operations
    registry = get_custom_operations_registry()

    registry.register_operation('gpu_condition_number', gpu_condition_number)
    registry.register_operation('gpu_matrix_rank', gpu_matrix_rank)
    registry.register_operation('matrix_normalize', matrix_normalize)
    registry.register_operation('extract_matrix_features', extract_matrix_features)

# Register default custom operations
register_default_custom_operations()
