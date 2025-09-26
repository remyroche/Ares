#!/usr/bin/env python3
"""
Performance Optimization Utilities for NAS Components

This module provides performance optimization utilities including caching,
vectorization, parallel processing, and memory-efficient operations.
"""

import time
import functools
import threading
import concurrent.futures
from typing import Any, Callable, Dict, List, Optional, Union, TypeVar, Generic
from dataclasses import dataclass, field
from contextlib import contextmanager
import logging
import weakref
import gc
import psutil
import numpy as np
from pathlib import Path

from .nas_error_handling import (
    NASComputationError, ErrorContext, error_context, 
    safe_execute, get_error_handler
)
from .nas_threading import ThreadSafeCache, ThreadSafeCounter, get_thread_pool
from .nas_resource_manager import ResourceType, managed_resource

T = TypeVar('T')


@dataclass
class PerformanceMetrics:
    """Performance metrics for operations."""
    operation_name: str
    start_time: float
    end_time: float
    duration: float
    memory_before: float
    memory_after: float
    memory_delta: float
    cpu_percent: float
    success: bool
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class PerformanceProfiler:
    """Profiles performance of operations and provides optimization suggestions."""
    
    def __init__(self, max_history: int = 1000):
        self._metrics_history: List[PerformanceMetrics] = []
        self._max_history = max_history
        self._lock = threading.RLock()
        self._error_handler = get_error_handler()
        self._logger = logging.getLogger(__name__)
    
    def profile_operation(
        self,
        operation_name: str,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Profile a single operation."""
        start_time = time.time()
        memory_before = self._get_memory_usage()
        cpu_before = psutil.cpu_percent()
        
        try:
            result = func(*args, **kwargs)
            success = True
            error_message = None
        except Exception as e:
            success = False
            error_message = str(e)
            result = None
        
        end_time = time.time()
        memory_after = self._get_memory_usage()
        cpu_after = psutil.cpu_percent()
        
        metrics = PerformanceMetrics(
            operation_name=operation_name,
            start_time=start_time,
            end_time=end_time,
            duration=end_time - start_time,
            memory_before=memory_before,
            memory_after=memory_after,
            memory_delta=memory_after - memory_before,
            cpu_percent=(cpu_before + cpu_after) / 2,
            success=success,
            error_message=error_message
        )
        
        self._record_metrics(metrics)
        
        if not success:
            raise Exception(error_message)
        
        return result
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except Exception:
            return 0.0
    
    def _record_metrics(self, metrics: PerformanceMetrics) -> None:
        """Record performance metrics."""
        with self._lock:
            self._metrics_history.append(metrics)
            
            # Keep only recent history
            if len(self._metrics_history) > self._max_history:
                self._metrics_history = self._metrics_history[-self._max_history:]
    
    def get_operation_stats(self, operation_name: str) -> Dict[str, Any]:
        """Get statistics for a specific operation."""
        with self._lock:
            operation_metrics = [
                m for m in self._metrics_history 
                if m.operation_name == operation_name
            ]
            
            if not operation_metrics:
                return {}
            
            durations = [m.duration for m in operation_metrics]
            memory_deltas = [m.memory_delta for m in operation_metrics]
            success_rate = sum(1 for m in operation_metrics if m.success) / len(operation_metrics)
            
            return {
                'operation_name': operation_name,
                'total_calls': len(operation_metrics),
                'success_rate': success_rate,
                'avg_duration': np.mean(durations),
                'min_duration': np.min(durations),
                'max_duration': np.max(durations),
                'std_duration': np.std(durations),
                'avg_memory_delta': np.mean(memory_deltas),
                'min_memory_delta': np.min(memory_deltas),
                'max_memory_delta': np.max(memory_deltas),
                'std_memory_delta': np.std(memory_deltas)
            }
    
    def get_all_stats(self) -> Dict[str, Any]:
        """Get statistics for all operations."""
        with self._lock:
            operation_names = list(set(m.operation_name for m in self._metrics_history))
            return {
                name: self.get_operation_stats(name)
                for name in operation_names
            }
    
    def get_slowest_operations(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get the slowest operations."""
        with self._lock:
            operation_stats = self.get_all_stats()
            sorted_operations = sorted(
                operation_stats.values(),
                key=lambda x: x.get('avg_duration', 0),
                reverse=True
            )
            return sorted_operations[:limit]
    
    def get_memory_intensive_operations(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get the most memory-intensive operations."""
        with self._lock:
            operation_stats = self.get_all_stats()
            sorted_operations = sorted(
                operation_stats.values(),
                key=lambda x: x.get('avg_memory_delta', 0),
                reverse=True
            )
            return sorted_operations[:limit]


class VectorizedOperations:
    """Provides vectorized operations for better performance."""
    
    def __init__(self):
        self._error_handler = get_error_handler()
        self._logger = logging.getLogger(__name__)
    
    def vectorized_apply(
        self,
        func: Callable,
        data: Union[List, np.ndarray],
        batch_size: int = 1000,
        use_multiprocessing: bool = False
    ) -> List[Any]:
        """Apply function to data in vectorized batches."""
        try:
            if isinstance(data, list):
                data = np.array(data)
            
            results = []
            for i in range(0, len(data), batch_size):
                batch = data[i:i + batch_size]
                
                if use_multiprocessing:
                    with concurrent.futures.ProcessPoolExecutor() as executor:
                        batch_results = list(executor.map(func, batch))
                else:
                    batch_results = [func(item) for item in batch]
                
                results.extend(batch_results)
            
            return results
            
        except Exception as e:
            context = ErrorContext("vectorized_apply", "vectorized_operations")
            self._error_handler.handle_error(e, context, reraise=False)
            return []
    
    def vectorized_reduce(
        self,
        func: Callable,
        data: Union[List, np.ndarray],
        initial_value: Any = None
    ) -> Any:
        """Apply reduction function to data in vectorized manner."""
        try:
            if isinstance(data, list):
                data = np.array(data)
            
            if initial_value is None:
                result = data[0]
                for item in data[1:]:
                    result = func(result, item)
            else:
                result = initial_value
                for item in data:
                    result = func(result, item)
            
            return result
            
        except Exception as e:
            context = ErrorContext("vectorized_reduce", "vectorized_operations")
            self._error_handler.handle_error(e, context, reraise=False)
            return initial_value
    
    def vectorized_filter(
        self,
        predicate: Callable,
        data: Union[List, np.ndarray]
    ) -> List[Any]:
        """Filter data using vectorized operations."""
        try:
            if isinstance(data, list):
                data = np.array(data)
            
            mask = np.array([predicate(item) for item in data])
            return data[mask].tolist()
            
        except Exception as e:
            context = ErrorContext("vectorized_filter", "vectorized_operations")
            self._error_handler.handle_error(e, context, reraise=False)
            return []
    
    def vectorized_map(
        self,
        func: Callable,
        data: Union[List, np.ndarray]
    ) -> List[Any]:
        """Map function over data using vectorized operations."""
        try:
            if isinstance(data, list):
                data = np.array(data)
            
            return [func(item) for item in data]
            
        except Exception as e:
            context = ErrorContext("vectorized_map", "vectorized_operations")
            self._error_handler.handle_error(e, context, reraise=False)
            return []


class MemoryEfficientOperations:
    """Provides memory-efficient operations for large datasets."""
    
    def __init__(self):
        self._error_handler = get_error_handler()
        self._logger = logging.getLogger(__name__)
    
    def chunked_processing(
        self,
        data: List[Any],
        chunk_size: int = 1000,
        func: Callable = None
    ) -> List[Any]:
        """Process data in chunks to reduce memory usage."""
        try:
            results = []
            for i in range(0, len(data), chunk_size):
                chunk = data[i:i + chunk_size]
                
                if func:
                    chunk_result = func(chunk)
                    results.append(chunk_result)
                else:
                    results.append(chunk)
                
                # Force garbage collection after each chunk
                gc.collect()
            
            return results
            
        except Exception as e:
            context = ErrorContext("chunked_processing", "memory_efficient_operations")
            self._error_handler.handle_error(e, context, reraise=False)
            return []
    
    def streaming_processing(
        self,
        data_generator: Callable,
        func: Callable,
        buffer_size: int = 1000
    ) -> List[Any]:
        """Process data in streaming fashion to minimize memory usage."""
        try:
            results = []
            buffer = []
            
            for item in data_generator():
                buffer.append(item)
                
                if len(buffer) >= buffer_size:
                    # Process buffer
                    buffer_results = func(buffer)
                    results.extend(buffer_results)
                    
                    # Clear buffer and force garbage collection
                    buffer.clear()
                    gc.collect()
            
            # Process remaining items
            if buffer:
                buffer_results = func(buffer)
                results.extend(buffer_results)
            
            return results
            
        except Exception as e:
            context = ErrorContext("streaming_processing", "memory_efficient_operations")
            self._error_handler.handle_error(e, context, reraise=False)
            return []
    
    def lazy_evaluation(
        self,
        data: List[Any],
        func: Callable
    ) -> Callable:
        """Create a lazy evaluation function for data processing."""
        def lazy_func():
            for item in data:
                yield func(item)
        
        return lazy_func
    
    def memory_mapped_processing(
        self,
        file_path: str,
        func: Callable,
        chunk_size: int = 1024 * 1024  # 1MB chunks
    ) -> List[Any]:
        """Process large files using memory mapping."""
        try:
            results = []
            
            with open(file_path, 'rb') as f:
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    
                    chunk_result = func(chunk)
                    results.append(chunk_result)
                    
                    # Force garbage collection
                    gc.collect()
            
            return results
            
        except Exception as e:
            context = ErrorContext("memory_mapped_processing", "memory_efficient_operations")
            self._error_handler.handle_error(e, context, reraise=False)
            return []


class ParallelProcessing:
    """Provides parallel processing utilities for CPU-intensive tasks."""
    
    def __init__(self, max_workers: int = None):
        self.max_workers = max_workers or psutil.cpu_count()
        self._error_handler = get_error_handler()
        self._logger = logging.getLogger(__name__)
    
    def parallel_map(
        self,
        func: Callable,
        data: List[Any],
        use_threads: bool = True
    ) -> List[Any]:
        """Apply function to data in parallel."""
        try:
            if use_threads:
                with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    results = list(executor.map(func, data))
            else:
                with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                    results = list(executor.map(func, data))
            
            return results
            
        except Exception as e:
            context = ErrorContext("parallel_map", "parallel_processing")
            self._error_handler.handle_error(e, context, reraise=False)
            return []
    
    def parallel_reduce(
        self,
        func: Callable,
        data: List[Any],
        initial_value: Any = None
    ) -> Any:
        """Apply reduction function to data in parallel."""
        try:
            if len(data) <= self.max_workers:
                # Use sequential processing for small datasets
                if initial_value is None:
                    result = data[0]
                    for item in data[1:]:
                        result = func(result, item)
                else:
                    result = initial_value
                    for item in data:
                        result = func(result, item)
                return result
            
            # Split data into chunks
            chunk_size = len(data) // self.max_workers
            chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
            
            # Process chunks in parallel
            with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                chunk_results = list(executor.map(
                    lambda chunk: self._reduce_chunk(func, chunk, initial_value),
                    chunks
                ))
            
            # Combine results
            if initial_value is None:
                result = chunk_results[0]
                for chunk_result in chunk_results[1:]:
                    result = func(result, chunk_result)
            else:
                result = initial_value
                for chunk_result in chunk_results:
                    result = func(result, chunk_result)
            
            return result
            
        except Exception as e:
            context = ErrorContext("parallel_reduce", "parallel_processing")
            self._error_handler.handle_error(e, context, reraise=False)
            return initial_value
    
    def _reduce_chunk(self, func: Callable, chunk: List[Any], initial_value: Any) -> Any:
        """Reduce a single chunk."""
        if initial_value is None:
            result = chunk[0]
            for item in chunk[1:]:
                result = func(result, item)
        else:
            result = initial_value
            for item in chunk:
                result = func(result, item)
        return result
    
    def parallel_filter(
        self,
        predicate: Callable,
        data: List[Any]
    ) -> List[Any]:
        """Filter data in parallel."""
        try:
            with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                results = list(executor.map(predicate, data))
            
            return [item for item, keep in zip(data, results) if keep]
            
        except Exception as e:
            context = ErrorContext("parallel_filter", "parallel_processing")
            self._error_handler.handle_error(e, context, reraise=False)
            return []


class CachingOptimizer:
    """Optimizes caching strategies for better performance."""
    
    def __init__(self, max_cache_size: int = 10000, ttl_seconds: float = 3600.0):
        self._cache = ThreadSafeCache(max_cache_size, ttl_seconds)
        self._cache_stats = ThreadSafeCounter()
        self._error_handler = get_error_handler()
        self._logger = logging.getLogger(__name__)
    
    def cached_function(
        self,
        func: Callable,
        cache_key: str = None,
        ttl_seconds: float = None
    ) -> Callable:
        """Create a cached version of a function."""
        def wrapper(*args, **kwargs):
            try:
                # Generate cache key if not provided
                if cache_key is None:
                    key = f"{func.__name__}_{hash(str(args) + str(kwargs))}"
                else:
                    key = f"{cache_key}_{hash(str(args) + str(kwargs))}"
                
                # Check cache
                cached_result = self._cache.get(key)
                if cached_result is not None:
                    self._cache_stats.increment()
                    return cached_result
                
                # Execute function and cache result
                result = func(*args, **kwargs)
                self._cache.set(key, result)
                
                return result
                
            except Exception as e:
                context = ErrorContext("cached_function", "caching_optimizer")
                self._error_handler.handle_error(e, context, reraise=False)
                return func(*args, **kwargs)
        
        return wrapper
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            'cache_hits': self._cache_stats.get_value(),
            'cache_size': self._cache.size(),
            'cache_stats': self._cache.get_stats() if hasattr(self._cache, 'get_stats') else {}
        }
    
    def clear_cache(self) -> None:
        """Clear the cache."""
        self._cache.clear()
        self._cache_stats.reset()


class PerformanceOptimizer:
    """Main performance optimizer that coordinates all optimization strategies."""
    
    def __init__(self):
        self._profiler = PerformanceProfiler()
        self._vectorized_ops = VectorizedOperations()
        self._memory_efficient_ops = MemoryEfficientOperations()
        self._parallel_ops = ParallelProcessing()
        self._caching_optimizer = CachingOptimizer()
        self._error_handler = get_error_handler()
        self._logger = logging.getLogger(__name__)
    
    def optimize_function(
        self,
        func: Callable,
        optimization_strategy: str = "auto"
    ) -> Callable:
        """Optimize a function based on the specified strategy."""
        try:
            if optimization_strategy == "auto":
                # Analyze function and choose best strategy
                optimization_strategy = self._analyze_function(func)
            
            if optimization_strategy == "caching":
                return self._caching_optimizer.cached_function(func)
            elif optimization_strategy == "vectorization":
                return self._vectorized_ops.vectorized_apply
            elif optimization_strategy == "parallel":
                return self._parallel_ops.parallel_map
            elif optimization_strategy == "memory_efficient":
                return self._memory_efficient_ops.chunked_processing
            else:
                return func
                
        except Exception as e:
            context = ErrorContext("optimize_function", "performance_optimizer")
            self._error_handler.handle_error(e, context, reraise=False)
            return func
    
    def _analyze_function(self, func: Callable) -> str:
        """Analyze a function to determine the best optimization strategy."""
        # Simple heuristic-based analysis
        func_name = func.__name__.lower()
        
        if any(keyword in func_name for keyword in ['cache', 'memoize', 'lookup']):
            return "caching"
        elif any(keyword in func_name for keyword in ['vector', 'batch', 'array']):
            return "vectorization"
        elif any(keyword in func_name for keyword in ['parallel', 'concurrent', 'thread']):
            return "parallel"
        elif any(keyword in func_name for keyword in ['chunk', 'stream', 'memory']):
            return "memory_efficient"
        else:
            return "caching"  # Default to caching
    
    def profile_operation(self, operation_name: str, func: Callable, *args, **kwargs) -> Any:
        """Profile an operation."""
        return self._profiler.profile_operation(operation_name, func, *args, **kwargs)
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        return {
            'profiler_stats': self._profiler.get_all_stats(),
            'slowest_operations': self._profiler.get_slowest_operations(),
            'memory_intensive_operations': self._profiler.get_memory_intensive_operations(),
            'cache_stats': self._caching_optimizer.get_cache_stats()
        }
    
    def optimize_memory_usage(self) -> Dict[str, Any]:
        """Optimize memory usage."""
        try:
            # Force garbage collection
            collected = gc.collect()
            
            # Get memory stats
            process = psutil.Process()
            memory_info = process.memory_info()
            
            return {
                'garbage_collected': collected,
                'memory_usage_mb': memory_info.rss / 1024 / 1024,
                'optimization_time': time.time()
            }
            
        except Exception as e:
            context = ErrorContext("optimize_memory_usage", "performance_optimizer")
            self._error_handler.handle_error(e, context, reraise=False)
            return {}


# Global performance optimizer
_global_performance_optimizer = PerformanceOptimizer()


def profile_operation(operation_name: str):
    """Decorator to profile an operation."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return _global_performance_optimizer.profile_operation(
                operation_name, func, *args, **kwargs
            )
        return wrapper
    return decorator


def optimize_function(optimization_strategy: str = "auto"):
    """Decorator to optimize a function."""
    def decorator(func: Callable) -> Callable:
        return _global_performance_optimizer.optimize_function(func, optimization_strategy)
    return decorator


def cached_function(cache_key: str = None, ttl_seconds: float = None):
    """Decorator to cache function results."""
    def decorator(func: Callable) -> Callable:
        return _global_performance_optimizer._caching_optimizer.cached_function(
            func, cache_key, ttl_seconds
        )
    return decorator


def vectorized_operation(batch_size: int = 1000):
    """Decorator to vectorize operations."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(data, *args, **kwargs):
            return _global_performance_optimizer._vectorized_ops.vectorized_apply(
                func, data, batch_size
            )
        return wrapper
    return decorator


def parallel_operation(max_workers: int = None):
    """Decorator to parallelize operations."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(data, *args, **kwargs):
            return _global_performance_optimizer._parallel_ops.parallel_map(
                func, data
            )
        return wrapper
    return decorator


def memory_efficient_operation(chunk_size: int = 1000):
    """Decorator to make operations memory-efficient."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(data, *args, **kwargs):
            return _global_performance_optimizer._memory_efficient_ops.chunked_processing(
                data, chunk_size, func
            )
        return wrapper
    return decorator


@contextmanager
def performance_monitoring(operation_name: str):
    """Context manager for performance monitoring."""
    start_time = time.time()
    memory_before = psutil.Process().memory_info().rss / 1024 / 1024
    
    try:
        yield
    finally:
        end_time = time.time()
        memory_after = psutil.Process().memory_info().rss / 1024 / 1024
        
        duration = end_time - start_time
        memory_delta = memory_after - memory_before
        
        _global_performance_optimizer._logger.info(
            f"Operation {operation_name}: {duration:.3f}s, "
            f"Memory delta: {memory_delta:.1f}MB"
        )


def get_performance_optimizer() -> PerformanceOptimizer:
    """Get the global performance optimizer."""
    return _global_performance_optimizer


def get_performance_report() -> Dict[str, Any]:
    """Get performance report."""
    return _global_performance_optimizer.get_performance_report()


def optimize_memory_usage() -> Dict[str, Any]:
    """Optimize memory usage."""
    return _global_performance_optimizer.optimize_memory_usage()


# Export main classes and functions
__all__ = [
    'PerformanceMetrics',
    'PerformanceProfiler',
    'VectorizedOperations',
    'MemoryEfficientOperations',
    'ParallelProcessing',
    'CachingOptimizer',
    'PerformanceOptimizer',
    'profile_operation',
    'optimize_function',
    'cached_function',
    'vectorized_operation',
    'parallel_operation',
    'memory_efficient_operation',
    'performance_monitoring',
    'get_performance_optimizer',
    'get_performance_report',
    'optimize_memory_usage'
]