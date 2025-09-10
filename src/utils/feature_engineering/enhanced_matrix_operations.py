"""
Enhanced Matrix Operations with GPU Acceleration and Memory Optimization.

This module provides comprehensive matrix operations optimized for machine learning
workflows, including GPU acceleration, memory management, and vectorized computations.
"""

import numpy as np
import pandas as pd
import torch
from typing import Any, Dict, List, Optional, Tuple, Union, Callable, Iterator
from scipy import linalg, sparse
import logging
from contextlib import contextmanager
import time
from collections import deque

from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
import traceback

import os

logger = logging.getLogger(__name__)

# Comprehensive Error Handling Framework
class OptimizationError(Exception):
    """Base exception for optimization-related errors."""
    def __init__(self, message: str, operation: str = None, details: Dict[str, Any] = None):
        super().__init__(message)
        self.operation = operation
        self.details = details or {}
        self.timestamp = time.time()

    def to_dict(self) -> Dict[str, Any]:
        return {
            'error_type': self.__class__.__name__,
            'message': str(self),
            'operation': self.operation,
            'details': self.details,
            'timestamp': self.timestamp,
            'traceback': traceback.format_exc()
        }

class GPUError(OptimizationError):
    """GPU-related errors."""
    pass

class MemoryError(OptimizationError):
    """Memory-related errors."""
    pass

class MatrixOperationError(OptimizationError):
    """Matrix operation errors."""
    pass

class DataProcessingError(OptimizationError):
    """Data processing errors."""
    pass

class ConfigurationError(OptimizationError):
    """Configuration-related errors."""
    pass

@dataclass
class ErrorRecoveryResult:
    """Result of error recovery attempt."""
    success: bool
    fallback_used: bool
    recovery_method: str
    execution_time: float
    error_details: Dict[str, Any] = field(default_factory=dict)

class ErrorHandler:
    """Comprehensive error handler with recovery mechanisms."""

    def __init__(self, enable_logging: bool = True, enable_recovery: bool = True):
        self.enable_logging = enable_logging
        self.enable_recovery = enable_recovery
        self.error_history: deque = deque(maxlen=1000)
        self.recovery_strategies = self._init_recovery_strategies()
        self.logger = logging.getLogger(f"{__name__}.ErrorHandler")

    def _init_recovery_strategies(self) -> Dict[str, Callable]:
        """Initialize recovery strategies for different error types."""
        return {
            'gpu_memory_error': self._recover_gpu_memory_error,
            'cpu_memory_error': self._recover_cpu_memory_error,
            'matrix_singular_error': self._recover_matrix_singular_error,
            'file_io_error': self._recover_file_io_error,
            'network_error': self._recover_network_error,
            'timeout_error': self._recover_timeout_error
        }

    def handle_error(self, error: Exception, operation: str, context: Dict[str, Any] = None) -> ErrorRecoveryResult:
        """Handle an error with appropriate recovery mechanism."""
        error_info = {
            'error_type': error.__class__.__name__,
            'error_message': str(error),
            'operation': operation,
            'context': context or {},
            'timestamp': time.time(),
            'traceback': traceback.format_exc()
        }

        # Log error if enabled
        if self.enable_logging:
            self.logger.error(f"🚨 {operation} failed: {error}")
            self.error_history.append(error_info)

        # Attempt recovery if enabled
        if self.enable_recovery:
            recovery_result = self._attempt_recovery(error, operation, context)
            if recovery_result.success:
                self.logger.info(f"✅ Recovered from {operation} error using {recovery_result.recovery_method}")
                return recovery_result

        # Return failure result
        return ErrorRecoveryResult(
            success=False,
            fallback_used=False,
            recovery_method='none',
            execution_time=0.0,
            error_details=error_info
        )

    def _attempt_recovery(self, error: Exception, operation: str, context: Dict[str, Any]) -> ErrorRecoveryResult:
        """Attempt to recover from an error."""
        error_type = self._classify_error(error)

        if error_type in self.recovery_strategies:
            try:
                start_time = time.time()
                result = self.recovery_strategies[error_type](error, operation, context)
                execution_time = time.time() - start_time

                if result:
                    return ErrorRecoveryResult(
                        success=True,
                        fallback_used=True,
                        recovery_method=error_type,
                        execution_time=execution_time
                    )
            except Exception as recovery_error:
                self.logger.warning(f"Recovery attempt failed: {recovery_error}")

        return ErrorRecoveryResult(
            success=False,
            fallback_used=False,
            recovery_method='failed',
            execution_time=0.0
        )

    def _classify_error(self, error: Exception) -> str:
        """Classify error type for appropriate recovery strategy."""
        error_str = str(error).lower()
        error_type = error.__class__.__name__.lower()

        # GPU-related errors
        if any(keyword in error_str for keyword in ['cuda', 'gpu', 'mps', 'memory']):
            return 'gpu_memory_error'

        # CPU memory errors
        if any(keyword in error_str for keyword in ['memory', 'allocation', 'out of memory']):
            return 'cpu_memory_error'

        # Matrix operation errors
        if any(keyword in error_str for keyword in ['singular', 'not positive definite', 'linear dependence']):
            return 'matrix_singular_error'

        # File I/O errors
        if any(keyword in error_type for keyword in ['file', 'io', 'permission']):
            return 'file_io_error'

        # Network errors
        if any(keyword in error_str for keyword in ['connection', 'timeout', 'network']):
            return 'network_error'

        # Timeout errors
        if 'timeout' in error_str:
            return 'timeout_error'

        return 'unknown_error'

    def _recover_gpu_memory_error(self, error: Exception, operation: str, context: Dict[str, Any]) -> bool:
        """Recover from GPU memory errors."""
        try:
            # Clear GPU caches
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            elif torch.backends.mps.is_available():
                torch.mps.empty_cache()

            # Try with smaller batch size if available
            if 'batch_size' in context:
                context['batch_size'] = max(1, context['batch_size'] // 2)
                self.logger.info(f"Reduced batch size to {context['batch_size']} for recovery")

            return True
        except Exception:
            return False

    def _recover_cpu_memory_error(self, error: Exception, operation: str, context: Dict[str, Any]) -> bool:
        """Recover from CPU memory errors."""
        try:
            # Force garbage collection
            import gc
            gc.collect()

            # Clear any large caches
            if hasattr(context, 'clear_cache'):
                context['clear_cache']()

            return True
        except Exception:
            return False

    def _recover_matrix_singular_error(self, error: Exception, operation: str, context: Dict[str, Any]) -> bool:
        """Recover from matrix singularity errors."""
        try:
            # Add regularization if matrix operation
            if 'matrix' in context:
                matrix = context['matrix']
                # Add small diagonal regularization
                if hasattr(matrix, 'shape') and len(matrix.shape) == 2:
                    regularization = np.eye(matrix.shape[0]) * 1e-8
                    context['matrix'] = matrix + regularization
                    return True
            return False
        except Exception:
            return False

    def _recover_file_io_error(self, error: Exception, operation: str, context: Dict[str, Any]) -> bool:
        """Recover from file I/O errors."""
        try:
            # Try alternative file paths or formats
            if 'filepath' in context:
                filepath = context['filepath']

                # Try with .tmp extension first
                if not filepath.endswith('.tmp'):
                    context['filepath'] = filepath + '.tmp'
                    return True

                # Try in temp directory
                import tempfile
                temp_dir = tempfile.gettempdir()
                filename = os.path.basename(filepath)
                context['filepath'] = os.path.join(temp_dir, filename)
                return True

            return False
        except Exception:
            return False

    def _recover_network_error(self, error: Exception, operation: str, context: Dict[str, Any]) -> bool:
        """Recover from network errors."""
        try:
            # Implement exponential backoff
            if 'retry_count' not in context:
                context['retry_count'] = 0

            context['retry_count'] += 1
            if context['retry_count'] <= 3:
                # Wait with exponential backoff
                wait_time = 2 ** context['retry_count']
                time.sleep(wait_time)
                return True

            return False
        except Exception:
            return False

    def _recover_timeout_error(self, error: Exception, operation: str, context: Dict[str, Any]) -> bool:
        """Recover from timeout errors."""
        try:
            # Increase timeout or reduce operation complexity
            if 'timeout' in context:
                context['timeout'] *= 1.5  # Increase timeout by 50%
                return True

            if 'batch_size' in context:
                context['batch_size'] = max(1, context['batch_size'] // 2)
                return True

            return False
        except Exception:
            return False

    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error statistics."""
        if not self.error_history:
            return {'total_errors': 0}

        total_errors = len(self.error_history)
        error_types = {}

        for error in self.error_history:
            error_type = error.get('error_type', 'unknown')
            error_types[error_type] = error_types.get(error_type, 0) + 1

        # Calculate error rate over time
        recent_errors = [e for e in self.error_history if time.time() - e['timestamp'] < 3600]  # Last hour
        error_rate_per_hour = len(recent_errors)

        return {
            'total_errors': total_errors,
            'error_types': error_types,
            'error_rate_per_hour': error_rate_per_hour,
            'most_common_error': max(error_types, key=error_types.get) if error_types else 'none',
            'recovery_success_rate': self._calculate_recovery_rate()
        }

    def _calculate_recovery_rate(self) -> float:
        """Calculate recovery success rate."""
        if not self.error_history:
            return 0.0

        recovery_attempts = [e for e in self.error_history if 'recovery_attempted' in e]
        successful_recoveries = [e for e in recovery_attempts if e.get('recovery_success', False)]

        return len(successful_recoveries) / len(recovery_attempts) if recovery_attempts else 0.0

# Error handling decorators
def with_error_handling(operation_name: str = None, enable_recovery: bool = True,
                       log_errors: bool = True, reraise: bool = True):
    """Decorator for comprehensive error handling."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get operation name
            op_name = operation_name or func.__name__

            # Create error handler
            error_handler = ErrorHandler(enable_logging=log_errors, enable_recovery=enable_recovery)

            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Handle the error
                context = {
                    'function': func.__name__,
                    'args_count': len(args),
                    'kwargs_keys': list(kwargs.keys()),
                    'module': func.__module__
                }

                recovery_result = error_handler.handle_error(e, op_name, context)

                if recovery_result.success:
                    # Try to re-execute with recovery context
                    try:
                        # Update kwargs with recovery context
                        recovery_kwargs = kwargs.copy()
                        recovery_kwargs.update(context)
                        return func(*args, **recovery_kwargs)
                    except Exception as retry_error:
                        logger.error(f"Retry after recovery failed: {retry_error}")

                if reraise:
                    if isinstance(e, OptimizationError):
                        raise
                    else:
                        # Wrap in appropriate error type
                        raise OptimizationError(
                            f"{op_name} failed: {str(e)}",
                            operation=op_name,
                            details={'original_error': str(e), 'context': context}
                        ) from e
                else:
                    logger.warning(f"Error in {op_name} suppressed: {e}")
                    return None

        return wrapper
    return decorator

def with_gpu_fallback(operation_name: str = None):
    """Decorator that provides GPU fallback to CPU."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_str = str(e).lower()
                if any(keyword in error_str for keyword in ['cuda', 'gpu', 'mps', 'memory']):
                    logger.warning(f"GPU operation failed, falling back to CPU: {e}")

                    # Try with CPU fallback
                    cpu_kwargs = kwargs.copy()
                    cpu_kwargs['use_gpu'] = False
                    cpu_kwargs['device'] = 'cpu'

                    try:
                        return func(*args, **cpu_kwargs)
                    except Exception as cpu_error:
                        logger.error(f"CPU fallback also failed: {cpu_error}")
                        raise OptimizationError(
                            f"Both GPU and CPU operations failed for {operation_name or func.__name__}",
                            operation=operation_name or func.__name__,
                            details={'gpu_error': str(e), 'cpu_error': str(cpu_error)}
                        ) from cpu_error
                else:
                    # Not a GPU error, re-raise
                    raise

        return wrapper
    return decorator

def with_memory_optimization(operation_name: str = None, max_retries: int = 3):
    """Decorator that optimizes memory usage and handles memory errors."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            op_name = operation_name or func.__name__
            last_exception = None

            for attempt in range(max_retries):
                try:
                    # Memory cleanup before execution
                    if attempt > 0:
                        import gc
                        gc.collect()

                        # Clear caches if available
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        elif torch.backends.mps.is_available():
                            torch.mps.empty_cache()

                        # Reduce batch size if specified
                        if 'batch_size' in kwargs:
                            kwargs['batch_size'] = max(1, kwargs['batch_size'] // 2)
                            logger.info(f"Reduced batch size to {kwargs['batch_size']} for retry")

                    return func(*args, **kwargs)

                except Exception as e:
                    last_exception = e
                    error_str = str(e).lower()

                    if any(keyword in error_str for keyword in ['memory', 'allocation', 'out of memory']):
                        if attempt < max_retries - 1:
                            logger.warning(f"Memory error in {op_name} (attempt {attempt + 1}/{max_retries}): {e}")
                            continue
                        else:
                            logger.error(f"Memory error in {op_name} persisted after {max_retries} attempts: {e}")
                    else:
                        # Not a memory error, don't retry
                        break

            # If we get here, all retries failed or it wasn't a memory error
            raise OptimizationError(
                f"{op_name} failed after {max_retries} attempts",
                operation=op_name,
                details={'last_error': str(last_exception), 'attempts': max_retries}
            ) from last_exception

        return wrapper
    return decorator

# Global error handler instance
_global_error_handler = None

def get_global_error_handler() -> ErrorHandler:
    """Get global error handler instance."""
    global _global_error_handler
    if _global_error_handler is None:
        _global_error_handler = ErrorHandler()
    return _global_error_handler

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
        """Initialize dynamic batch optimizer.

        Args:
            max_batch_size: Maximum allowed batch size
            min_batch_size: Minimum allowed batch size
            optimization_strategy: Strategy for batch optimization
            enable_learning: Whether to learn from past executions
        """
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
        """Determine optimal batch size for an operation.

        Args:
            operation_name: Name of the operation
            data_shape: Shape of the data to process
            operation_complexity: Complexity level of the operation
            available_memory_mb: Available memory in MB

        Returns:
            Optimal batch size
        """
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
                 chunk_size: int = 10000, dtype: torch.dtype = torch.float32,
                 enable_dynamic_batch: bool = True):
        """Initialize enhanced matrix operations.

        Args:
            use_gpu: Whether to use GPU acceleration
            memory_efficient: Whether to use memory-efficient operations
            chunk_size: Chunk size for large matrices
            dtype: Default data type for tensors
            enable_dynamic_batch: Whether to use dynamic batch optimization
        """
        self.use_gpu = use_gpu
        self.memory_efficient = memory_efficient
        self.chunk_size = chunk_size
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
            from .m1_gpu_utils import get_m1_gpu_manager
            from .m1_memory_optimizer import get_m1_memory_optimizer
            from .vectorized_processing_core import get_vectorized_processing_core

            self.gpu_manager = get_m1_gpu_manager() if self.use_gpu else None
            self.memory_optimizer = get_m1_memory_optimizer()
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

    def to_tensor(self, data: Union[np.ndarray, pd.DataFrame, List],
                 dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """Convert data to tensor with optimization."""
        if dtype is None:
            dtype = self.dtype

        # Convert to numpy first if needed
        if isinstance(data, pd.DataFrame):
            data = data.values.astype(np.float32)
        elif isinstance(data, list):
            data = np.array(data, dtype=np.float32)
        elif isinstance(data, np.ndarray) and data.dtype != np.float32:
            data = data.astype(np.float32)

        # Create tensor
        tensor = torch.from_numpy(data).to(dtype)

        # Move to GPU if available and requested
        if self.use_gpu and self.gpu_manager:
            tensor = self.gpu_manager.to_device(tensor, "matrix_mult")

        return tensor

    def matrix_multiply(self, a: Union[np.ndarray, torch.Tensor],
                       b: Union[np.ndarray, torch.Tensor],
                       use_gpu: Optional[bool] = None) -> Union[np.ndarray, torch.Tensor]:
        """Optimized matrix multiplication."""
        use_gpu = use_gpu if use_gpu is not None else self.use_gpu

        with self.matrix_operation_context("matrix_multiply"):
            # Convert to tensors
            a_tensor = self.to_tensor(a) if not isinstance(a, torch.Tensor) else a
            b_tensor = self.to_tensor(b) if not isinstance(b, torch.Tensor) else b

            # Decide on tiling based on memory and shape
            try:
                elem_bytes = a_tensor.element_size()
            except Exception:
                elem_bytes = 4
            try:
                total_elems = a_tensor.numel() + b_tensor.numel()
            except Exception:
                total_elems = 0
            data_size_mb = (total_elems * elem_bytes) / (1024**2) if total_elems else 0

            use_tiling = False
            if hasattr(self, 'memory_optimizer') and self.memory_optimizer:
                try:
                    use_tiling = self.memory_optimizer.should_chunk_data(data_size_mb, "matrix_mult")
                except Exception:
                    use_tiling = False

            # Additional shape-based heuristic
            try:
                m, k = int(a_tensor.shape[-2]), int(a_tensor.shape[-1])
                k2, n = int(b_tensor.shape[-2]), int(b_tensor.shape[-1])
                if k2 != k:
                    raise ValueError("Inner dimensions must match for matrix multiply")
                if max(m, n) >= 4096 or (m * n) >= 8_000_000:
                    use_tiling = True
            except Exception:
                pass

            if use_tiling:
                result = self._matrix_multiply_tiled(a, b, use_gpu=use_gpu)
                return result

            # Perform multiplication
            if use_gpu and self.gpu_manager:
                result = self.gpu_manager.matrix_multiply_mps(a_tensor, b_tensor)
            else:
                result = torch.matmul(a_tensor, b_tensor)

            # Convert back to numpy if input was numpy
            if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
                return result.cpu().numpy()
            return result

    def _matrix_multiply_tiled(self,
                               a: Union[np.ndarray, torch.Tensor],
                               b: Union[np.ndarray, torch.Tensor],
                               use_gpu: bool) -> Union[np.ndarray, torch.Tensor]:
        """Tiled matrix multiplication to prevent OOM and improve locality.

        Returns numpy array if any input was numpy; otherwise returns torch.Tensor on CPU.
        """
        # Normalize to numpy for tiling and assemble result
        def _to_numpy(x: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
            if isinstance(x, np.ndarray):
                return x
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().numpy()
            return np.asarray(x, dtype=np.float32)

        a_np = _to_numpy(a).astype(np.float32, copy=False)
        b_np = _to_numpy(b).astype(np.float32, copy=False)

        m, k = a_np.shape
        k2, n = b_np.shape
        if k2 != k:
            raise ValueError("Inner dimensions must match for matrix multiply")

        # Determine tile size using memory optimizer when available
        tile_rows = min(self.chunk_size, m)
        tile_cols = min(self.chunk_size, n)
        if hasattr(self, 'memory_optimizer') and self.memory_optimizer:
            try:
                # Use available memory to choose a safer row tile size
                tile_rows = max(128, self.memory_optimizer.calculate_optimal_chunk_size((m, k), 'matrix_mult'))
                tile_cols = tile_rows
            except Exception:
                tile_rows = min(self.chunk_size, m)
                tile_cols = min(self.chunk_size, n)

        result_np = np.zeros((m, n), dtype=np.float32)

        # Process tiles
        for i in range(0, m, tile_rows):
            end_i = min(i + tile_rows, m)
            a_block = a_np[i:end_i, :]

            for j in range(0, n, tile_cols):
                end_j = min(j + tile_cols, n)
                b_block = b_np[:, j:end_j]

                # Compute block product on device chosen by GPU manager
                a_t = torch.from_numpy(a_block).to(self.dtype)
                b_t = torch.from_numpy(b_block).to(self.dtype)
                if use_gpu and self.gpu_manager:
                    a_t = self.gpu_manager.to_device(a_t, "matrix_mult")
                    b_t = self.gpu_manager.to_device(b_t, "matrix_mult")
                c_block_t = torch.matmul(a_t, b_t)
                c_block = c_block_t.detach().cpu().numpy().astype(np.float32, copy=False)

                result_np[i:end_i, j:end_j] = c_block

                # Periodic memory cleanup
                if (i // tile_rows + j // tile_cols) % 6 == 0 and self.gpu_manager is not None:
                    try:
                        self.gpu_manager.optimize_memory()
                    except Exception:
                        pass

        # Preserve return type: numpy if any input was numpy; else torch tensor
        if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
            return result_np
        return torch.from_numpy(result_np)

    def batch_matrix_multiply(self, matrices_a: List[np.ndarray],
                            matrices_b: List[np.ndarray],
                            batch_size: Optional[int] = None) -> List[np.ndarray]:
        """Batch matrix multiplication with dynamic batch optimization."""
        start_time = time.time()

        # Determine optimal batch size if not provided
        if batch_size is None and self.batch_optimizer is not None:
            # Use first matrix as sample for optimization
            if matrices_a:
                sample_shape = matrices_a[0].shape
                available_memory = None
                if hasattr(self, 'memory_optimizer') and self.memory_optimizer:
                    memory_report = self.memory_optimizer.get_memory_report()
                    available_memory = memory_report.get('available_gb', 8.0) * 1024  # Convert to MB

                batch_size = self.batch_optimizer.optimize_batch_size(
                    operation_name="batch_matrix_multiply",
                    data_shape=sample_shape,
                    operation_complexity=OperationComplexity.HIGH,
                    available_memory_mb=available_memory
                )
        elif batch_size is None:
            batch_size = self.chunk_size

        results = []
        total_processed = 0

        with self.matrix_operation_context("batch_matrix_multiply"):
            for i in range(0, len(matrices_a), batch_size):
                end_idx = min(i + batch_size, len(matrices_a))
                current_batch_size = end_idx - i

                batch_start_time = time.time()

                batch_a = matrices_a[i:end_idx]
                batch_b = matrices_b[i:end_idx]

                # Convert batches to tensors
                batch_a_tensor = torch.stack([self.to_tensor(a) for a in batch_a])
                batch_b_tensor = torch.stack([self.to_tensor(b) for b in batch_b])

                # Batch matrix multiplication
                if self.use_gpu and self.gpu_manager:
                    batch_result = torch.bmm(batch_a_tensor, batch_b_tensor)
                else:
                    batch_result = torch.bmm(batch_a_tensor, batch_b_tensor)

                # Convert back to numpy
                batch_results = []
                for j in range(batch_result.shape[0]):
                    results.append(batch_result[j].cpu().numpy())
                    batch_results.append(batch_result[j].cpu().numpy())

                # Record performance for learning
                if self.batch_optimizer is not None:
                    batch_execution_time = time.time() - batch_start_time
                    batch_memory_usage = sum(arr.nbytes for arr in batch_results) / (1024 * 1024)  # MB
                    batch_data_processed = sum(np.prod(arr.shape) for arr in batch_results)

                    self.batch_optimizer.record_performance(
                        operation_name="batch_matrix_multiply",
                        batch_size=current_batch_size,
                        execution_time=batch_execution_time,
                        memory_usage=batch_memory_usage,
                        data_processed=batch_data_processed
                    )

                total_processed += len(batch_results)

                # Memory cleanup
                if self.memory_optimizer and (i // batch_size) % 5 == 0:
                    self.memory_optimizer.optimize_memory()

        # Log overall performance
        total_time = time.time() - start_time
        self.logger.info(
            f"📊 Batch matrix multiplication completed: {len(results)} operations in {total_time:.2f}s "
            f"(avg: {total_time/len(results):.3f}s per operation)"
        )

        return results

    def correlation_matrix(self, data: Union[pd.DataFrame, np.ndarray],
                         method: str = 'pearson') -> np.ndarray:
        """Compute correlation matrix with optimization."""
        with self.matrix_operation_context("correlation_matrix"):
            if isinstance(data, pd.DataFrame):
                if method == 'pearson':
                    return data.corr().values
                elif method == 'spearman':
                    return data.corr(method='spearman').values
                else:  # kendall
                    return data.corr(method='kendall').values
            else:
                # NumPy array correlation
                if method == 'pearson':
                    return np.corrcoef(data.T)
                else:
                    # For spearman/kendall with numpy, use pandas
                    df = pd.DataFrame(data.T)
                    return df.corr(method=method).values

    def covariance_matrix(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Compute covariance matrix with optimization."""
        with self.matrix_operation_context("covariance_matrix"):
            if isinstance(data, pd.DataFrame):
                return data.cov().values
            else:
                return np.cov(data.T)

    def eigendecomposition(self, matrix: np.ndarray,
                          use_gpu: Optional[bool] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Eigendecomposition with GPU acceleration."""
        use_gpu = use_gpu if use_gpu is not None else self.use_gpu

        with self.matrix_operation_context("eigendecomposition"):
            if use_gpu and self.gpu_manager:
                # GPU-accelerated eigendecomposition
                matrix_tensor = self.to_tensor(matrix)

                try:
                    eigenvalues, eigenvectors = torch.linalg.eigh(matrix_tensor)
                    return eigenvalues.cpu().numpy(), eigenvectors.cpu().numpy()
                except:
                    # Fallback to CPU
                    self.logger.warning("GPU eigendecomposition failed, using CPU")
                    return np.linalg.eigh(matrix)
            else:
                return np.linalg.eigh(matrix)

    def svd_decomposition(self, matrix: np.ndarray,
                         k: Optional[int] = None,
                         use_gpu: Optional[bool] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """SVD decomposition with GPU acceleration."""
        use_gpu = use_gpu if use_gpu is not None else self.use_gpu

        with self.matrix_operation_context("svd_decomposition"):
            if use_gpu and self.gpu_manager:
                matrix_tensor = self.to_tensor(matrix)

                try:
                    U, S, V = torch.linalg.svd(matrix_tensor)

                    # Truncate if k is specified
                    if k is not None:
                        U = U[:, :k]
                        S = S[:k]
                        V = V[:k, :]

                    return U.cpu().numpy(), S.cpu().numpy(), V.cpu().numpy()
                except:
                    self.logger.warning("GPU SVD failed, using CPU")
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

    def matrix_inverse(self, matrix: np.ndarray,
                      use_gpu: Optional[bool] = None) -> np.ndarray:
        """Matrix inversion with GPU acceleration."""
        use_gpu = use_gpu if use_gpu is not None else self.use_gpu

        with self.matrix_operation_context("matrix_inverse"):
            if use_gpu and self.gpu_manager:
                matrix_tensor = self.to_tensor(matrix)

                try:
                    inverse = torch.linalg.inv(matrix_tensor)
                    return inverse.cpu().numpy()
                except:
                    self.logger.warning("GPU matrix inversion failed, using CPU")
                    return np.linalg.inv(matrix)
            else:
                return np.linalg.inv(matrix)

    def cholesky_decomposition(self, matrix: np.ndarray,
                              use_gpu: Optional[bool] = None) -> np.ndarray:
        """Cholesky decomposition with GPU acceleration."""
        use_gpu = use_gpu if use_gpu is not None else self.use_gpu

        with self.matrix_operation_context("cholesky_decomposition"):
            if use_gpu and self.gpu_manager:
                matrix_tensor = self.to_tensor(matrix)

                try:
                    cholesky = torch.linalg.cholesky(matrix_tensor)
                    return cholesky.cpu().numpy()
                except:
                    self.logger.warning("GPU Cholesky failed, using CPU")
                    return np.linalg.cholesky(matrix)
            else:
                return np.linalg.cholesky(matrix)

    def matrix_power(self, matrix: np.ndarray, power: float,
                    use_gpu: Optional[bool] = None) -> np.ndarray:
        """Matrix power with support for fractional exponents.

        - Integer powers use GPU when available; negative integers are supported by torch/numpy.
        - Fractional powers are computed on CPU via scipy.linalg.fractional_matrix_power.
        """
        use_gpu = use_gpu if use_gpu is not None else self.use_gpu

        with self.matrix_operation_context("matrix_power"):
            # Determine if power is an integer
            is_int_power = isinstance(power, (int, np.integer)) or (isinstance(power, float) and float(power).is_integer())

            if is_int_power:
                int_power = int(power)
                if use_gpu and self.gpu_manager:
                    matrix_tensor = self.to_tensor(matrix)
                    try:
                        result = torch.linalg.matrix_power(matrix_tensor, int_power)
                        return result.cpu().numpy()
                    except Exception:
                        self.logger.warning("GPU integer matrix power failed, falling back to CPU")
                        return np.linalg.matrix_power(matrix, int_power)
                else:
                    return np.linalg.matrix_power(matrix, int_power)

            # Fractional power path (CPU via SciPy)
            try:
                return linalg.fractional_matrix_power(matrix, power)
            except Exception as e:
                self.logger.error(f"Fractional matrix power failed: {e}")
                raise

    def solve_linear_system(self, a: np.ndarray, b: np.ndarray,
                           use_gpu: Optional[bool] = None) -> np.ndarray:
        """Solve linear system Ax = b with GPU acceleration."""
        use_gpu = use_gpu if use_gpu is not None else self.use_gpu

        with self.matrix_operation_context("solve_linear_system"):
            if use_gpu and self.gpu_manager:
                a_tensor = self.to_tensor(a)
                b_tensor = self.to_tensor(b)

                try:
                    solution = torch.linalg.solve(a_tensor, b_tensor)
                    return solution.cpu().numpy()
                except:
                    self.logger.warning("GPU linear system solve failed, using CPU")
                    return np.linalg.solve(a, b)
            else:
                return np.linalg.solve(a, b)

    def qr_decomposition(self, matrix: np.ndarray,
                        use_gpu: Optional[bool] = None) -> Tuple[np.ndarray, np.ndarray]:
        """QR decomposition with GPU acceleration."""
        use_gpu = use_gpu if use_gpu is not None else self.use_gpu

        with self.matrix_operation_context("qr_decomposition"):
            if use_gpu and self.gpu_manager:
                matrix_tensor = self.to_tensor(matrix)

                try:
                    Q, R = torch.linalg.qr(matrix_tensor)
                    return Q.cpu().numpy(), R.cpu().numpy()
                except:
                    self.logger.warning("GPU QR decomposition failed, using CPU")
                    return np.linalg.qr(matrix)
            else:
                return np.linalg.qr(matrix)

    def matrix_norm(self, matrix: np.ndarray, ord: Union[str, int] = 'fro',
                   use_gpu: Optional[bool] = None) -> float:
        """Matrix norm computation with GPU acceleration."""
        use_gpu = use_gpu if use_gpu is not None else self.use_gpu

        with self.matrix_operation_context("matrix_norm"):
            if use_gpu and self.gpu_manager:
                matrix_tensor = self.to_tensor(matrix)

                try:
                    norm = torch.linalg.norm(matrix_tensor, ord=ord)
                    return float(norm.cpu().numpy())
                except:
                    self.logger.warning("GPU matrix norm failed, using CPU")
                    return np.linalg.norm(matrix, ord=ord)
            else:
                return np.linalg.norm(matrix, ord=ord)

    def condition_number(self, matrix: np.ndarray,
                        use_gpu: Optional[bool] = None) -> float:
        """Compute matrix condition number with GPU acceleration."""
        use_gpu = use_gpu if use_gpu is not None else self.use_gpu

        with self.matrix_operation_context("condition_number"):
            if use_gpu and self.gpu_manager:
                matrix_tensor = self.to_tensor(matrix)

                try:
                    cond = torch.linalg.cond(matrix_tensor)
                    return float(cond.cpu().numpy())
                except:
                    self.logger.warning("GPU condition number failed, using CPU")
                    return np.linalg.cond(matrix)
            else:
                return np.linalg.cond(matrix)

    def chunked_matrix_operation(self, matrix_a: np.ndarray, matrix_b: np.ndarray,
                               operation: Callable[[np.ndarray, np.ndarray], np.ndarray],
                               chunk_size: Optional[int] = None) -> np.ndarray:
        """Perform chunked matrix operations for large matrices."""
        if chunk_size is None:
            chunk_size = self.chunk_size

        with self.matrix_operation_context("chunked_operation"):
            rows_a, cols_a = matrix_a.shape
            rows_b, cols_b = matrix_b.shape

            # Determine output shape based on operation
            if operation == self.matrix_multiply:
                output_shape = (rows_a, cols_b)
            else:
                output_shape = matrix_a.shape  # Assume same shape for other operations

            result = np.zeros(output_shape, dtype=np.float32)

            # Chunked processing
            for i in range(0, rows_a, chunk_size):
                end_i = min(i + chunk_size, rows_a)

                for j in range(0, cols_b, chunk_size):
                    end_j = min(j + chunk_size, cols_b)

                    # Extract chunks
                    chunk_a = matrix_a[i:end_i, :]
                    chunk_b = matrix_b[:, j:end_j]

                    # Perform operation on chunk
                    chunk_result = operation(chunk_a, chunk_b)

                    # Store result
                    result[i:end_i, j:end_j] = chunk_result

                    # Memory cleanup
                    if self.memory_optimizer and ((i // chunk_size) * (j // chunk_size)) % 10 == 0:
                        self.memory_optimizer.optimize_memory()

            return result

    def sparse_matrix_multiply(self, a: Union[sparse.spmatrix, np.ndarray],
                             b: Union[sparse.spmatrix, np.ndarray],
                             format: str = 'csr') -> sparse.spmatrix:
        """Sparse matrix multiplication with GPU acceleration when beneficial."""
        with self.matrix_operation_context("sparse_matrix_multiply"):
            # Convert to sparse if needed
            if not sparse.issparse(a):
                a = sparse.csr_matrix(a)
            if not sparse.issparse(b):
                b = sparse.csr_matrix(b)

            # Check if GPU acceleration would be beneficial
            density_a = a.nnz / (a.shape[0] * a.shape[1])
            density_b = b.nnz / (b.shape[0] * b.shape[1])
            avg_density = (density_a + density_b) / 2

            # Use GPU for dense-like sparse matrices
            if avg_density > 0.3 and self.use_gpu and self.gpu_manager:
                self.logger.info(f"🔄 Using GPU for sparse matrix multiplication (density: {avg_density:.3f})")
                return self._gpu_sparse_multiply(a, b, format)
            else:
                # CPU sparse multiplication
                result = a * b
                if format != 'csr':
                    result = result.asformat(format)
                return result

    def _gpu_sparse_multiply(self, a: sparse.spmatrix, b: sparse.spmatrix,
                           format: str) -> sparse.spmatrix:
        """GPU-accelerated sparse matrix multiplication."""
        # Convert to dense for GPU processing (only for reasonably sized matrices)
        max_size = 50000  # Maximum size for dense conversion

        if a.shape[0] * a.shape[1] > max_size or b.shape[0] * b.shape[1] > max_size:
            self.logger.info("⚠️ Matrix too large for GPU sparse multiplication, using CPU")
            result = a * b
        else:
            # Convert to dense and use GPU
            a_dense = torch.from_numpy(a.toarray()).to(self.dtype)
            b_dense = torch.from_numpy(b.toarray()).to(self.dtype)

            if self.use_gpu and self.gpu_manager:
                a_dense = self.gpu_manager.to_device(a_dense, "matrix_mult")
                b_dense = self.gpu_manager.to_device(b_dense, "matrix_mult")

            result_dense = torch.matmul(a_dense, b_dense)
            result = sparse.csr_matrix(result_dense.cpu().numpy())

        if format != 'csr':
            result = result.asformat(format)

        return result

    def sparse_svd(self, matrix: sparse.spmatrix, k: Optional[int] = None,
                  solver: str = 'arpack') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sparse SVD decomposition."""
        with self.matrix_operation_context("sparse_svd"):
            # Use randomized SVD for better performance on large sparse matrices
            if k is None:
                k = min(100, min(matrix.shape) - 1)

            if solver == 'randomized':
                # Randomized SVD for sparse matrices
                from sklearn.utils.extmath import randomized_svd
                U, sigma, Vt = randomized_svd(matrix, n_components=k, random_state=42)
            else:
                # Standard sparse SVD
                U, sigma, Vt = sparse.linalg.svds(matrix, k=k)

            return U, sigma, Vt

    def sparse_eigen(self, matrix: sparse.spmatrix, k: int = 10,
                    which: str = 'LM') -> Tuple[np.ndarray, np.ndarray]:
        """Sparse eigenvalue decomposition."""
        with self.matrix_operation_context("sparse_eigen"):
            # Use sparse eigenvalue solver
            eigenvalues, eigenvectors = sparse.linalg.eigs(matrix, k=k, which=which)
            return eigenvalues.real, eigenvectors.real

    def sparse_to_dense_threshold(self, matrix: sparse.spmatrix,
                                density_threshold: float = 0.3) -> Union[sparse.spmatrix, np.ndarray]:
        """Convert sparse to dense if density exceeds threshold."""
        density = matrix.nnz / (matrix.shape[0] * matrix.shape[1])

        if density > density_threshold:
            self.logger.info(f"🔄 Converting sparse to dense (density: {density:.3f})")
            return matrix.toarray()
        else:
            return matrix

    def optimize_sparse_format(self, matrix: sparse.spmatrix,
                             target_format: Optional[str] = None) -> sparse.spmatrix:
        """Optimize sparse matrix format for operations."""
        if target_format is None:
            # Auto-select optimal format based on operation patterns
            # CSR is good for row operations, CSC for column operations
            # BSR for block operations
            target_format = 'csr'  # Default to CSR

        if matrix.format != target_format:
            matrix = matrix.asformat(target_format)

        return matrix

    def sparse_matrix_norm(self, matrix: sparse.spmatrix, ord: Union[str, int] = 'fro') -> float:
        """Compute sparse matrix norm."""
        with self.matrix_operation_context("sparse_matrix_norm"):
            return sparse.linalg.norm(matrix, ord=ord)

    def sparse_solve_linear(self, a: sparse.spmatrix, b: np.ndarray,
                          solver: str = 'spsolve') -> np.ndarray:
        """Solve sparse linear system."""
        with self.matrix_operation_context("sparse_solve_linear"):
            if solver == 'spsolve':
                return sparse.linalg.spsolve(a, b)
            elif solver == 'lsqr':
                x, istop, itn, r1norm = sparse.linalg.lsqr(a, b)[:4]
                return x
            else:
                raise ValueError(f"Unsupported solver: {solver}")

    def create_sparse_from_dense(self, matrix: np.ndarray,
                                sparsity_threshold: float = 0.1) -> Union[sparse.spmatrix, np.ndarray]:
        """Create sparse matrix from dense matrix if beneficial."""
        density = np.count_nonzero(matrix) / matrix.size

        if density < sparsity_threshold:
            self.logger.info(f"📦 Creating sparse matrix (density: {density:.3f})")
            return sparse.csr_matrix(matrix)
        else:
            return matrix

    def sparse_batch_operations(self, matrices: List[sparse.spmatrix],
                              operation: Callable[[sparse.spmatrix], sparse.spmatrix],
                              batch_size: Optional[int] = None) -> List[sparse.spmatrix]:
        """Batch operations on sparse matrices."""
        if batch_size is None:
            batch_size = self.chunk_size

        results = []

        with self.matrix_operation_context("sparse_batch_operations"):
            for i in range(0, len(matrices), batch_size):
                end_idx = min(i + batch_size, len(matrices))
                batch = matrices[i:end_idx]

                batch_results = []
                for matrix in batch:
                    result = operation(matrix)
                    batch_results.append(result)

                results.extend(batch_results)

                # Memory cleanup
                if self.memory_optimizer and (i // batch_size) % 5 == 0:
                    self.memory_optimizer.optimize_memory()

        return results

    def get_sparse_matrix_stats(self) -> Dict[str, Any]:
        """Get statistics about sparse matrix operations."""
        return {
            'sparse_operations_supported': True,
            'supported_formats': ['csr', 'csc', 'bsr', 'coo'],
            'gpu_sparse_support': self.use_gpu and self.gpu_manager is not None,
            'memory_efficient': self.memory_efficient
        }

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

        stats['sparse_matrix_stats'] = self.get_sparse_matrix_stats()

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
def gpu_matrix_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """GPU-accelerated matrix multiplication."""
    ops = get_enhanced_matrix_operations()
    return ops.matrix_multiply(a, b)

def correlation_matrix_gpu(data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
    """GPU-accelerated correlation matrix."""
    ops = get_enhanced_matrix_operations()
    return ops.correlation_matrix(data)

def eigendecomposition_gpu(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """GPU-accelerated eigendecomposition."""
    ops = get_enhanced_matrix_operations()
    return ops.eigendecomposition(matrix)

def svd_gpu(matrix: np.ndarray, k: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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

def sparse_matrix_multiply(a: Union[sparse.spmatrix, np.ndarray],
                          b: Union[sparse.spmatrix, np.ndarray],
                          format: str = 'csr') -> sparse.spmatrix:
    """Sparse matrix multiplication with GPU acceleration."""
    ops = get_enhanced_matrix_operations()
    return ops.sparse_matrix_multiply(a, b, format)

def sparse_svd(matrix: sparse.spmatrix, k: Optional[int] = None,
              solver: str = 'arpack') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sparse SVD decomposition."""
    ops = get_enhanced_matrix_operations()
    return ops.sparse_svd(matrix, k, solver)

def sparse_eigen(matrix: sparse.spmatrix, k: int = 10,
                which: str = 'LM') -> Tuple[np.ndarray, np.ndarray]:
    """Sparse eigenvalue decomposition."""
    ops = get_enhanced_matrix_operations()
    return ops.sparse_eigen(matrix, k, which)

def create_sparse_matrix(matrix: np.ndarray,
                        sparsity_threshold: float = 0.1) -> Union[sparse.spmatrix, np.ndarray]:
    """Create sparse matrix from dense matrix if beneficial."""
    ops = get_enhanced_matrix_operations()
    return ops.create_sparse_from_dense(matrix, sparsity_threshold)

def sparse_solve(a: sparse.spmatrix, b: np.ndarray,
                solver: str = 'spsolve') -> np.ndarray:
    """Solve sparse linear system."""
    ops = get_enhanced_matrix_operations()
    return ops.sparse_solve_linear(a, b, solver)

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

    def register_operation(self, name: str, operation_func: Callable, **kwargs) -> None:
        """Register a custom matrix operation."""
        if name in self.operations:
            self.logger.warning(f"Operation '{name}' already registered, overwriting")

        self.operations[name] = CustomMatrixOperation(name, operation_func, **kwargs)
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

# Extend EnhancedMatrixOperations with custom operations support
def add_custom_operation_to_enhanced_ops():
    """Add custom operations support to the enhanced matrix operations class."""
    # This function will be called to extend the class dynamically
    ops_class = EnhancedMatrixOperations

    # Add custom operations registry to the class
    if not hasattr(ops_class, 'custom_registry'):
        ops_class.custom_registry = get_custom_operations_registry()

    # Add methods for custom operations
    def execute_custom_operation(self, name: str, *args, **kwargs) -> Any:
        """Execute a custom matrix operation with optimization."""
        with self.matrix_operation_context(f"custom_{name}"):
            return self.custom_registry.execute_operation(name, *args, **kwargs)

    def register_custom_operation(self, name: str, operation_func: Callable, **kwargs) -> None:
        """Register a custom matrix operation."""
        self.custom_registry.register_operation(name, operation_func, **kwargs)

    def get_custom_operations_info(self) -> Dict[str, Any]:
        """Get information about custom operations."""
        return {
            'custom_operations': self.custom_registry.list_operations(),
            'registry_stats': self.custom_registry.get_registry_stats(),
            'supported': True
        }

    # Monkey patch the methods onto the class
    ops_class.execute_custom_operation = execute_custom_operation
    ops_class.register_custom_operation = register_custom_operation
    ops_class.get_custom_operations_info = get_custom_operations_info

# Initialize custom operations support
add_custom_operation_to_enhanced_ops()

# Example custom operations
def register_default_custom_operations():
    """Register some useful default custom matrix operations."""

    # Matrix condition number with GPU support
    def gpu_condition_number(matrix: np.ndarray) -> float:
        """Compute matrix condition number with GPU acceleration."""
        ops = get_enhanced_matrix_operations()
        return ops.condition_number(matrix)

    # Matrix rank with GPU support
    def gpu_matrix_rank(matrix: np.ndarray, tol: Optional[float] = None) -> int:
        """Compute matrix rank with GPU acceleration."""
        ops = get_enhanced_matrix_operations()
        # Use SVD to compute rank
        if ops.use_gpu and ops.gpu_manager:
            U, S, V = ops.svd_decomposition(matrix, k=min(matrix.shape))
            if tol is None:
                tol = S.max() * max(matrix.shape) * np.finfo(S.dtype).eps
            rank = np.sum(S > tol)
        else:
            rank = np.linalg.matrix_rank(matrix, tol=tol)
        return int(rank)

    # Custom matrix normalization
    def matrix_normalize(matrix: np.ndarray, method: str = 'l2', axis: int = 0) -> np.ndarray:
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
    def extract_matrix_features(matrix: np.ndarray) -> Dict[str, float]:
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
