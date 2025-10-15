"""
M1 Hardware Integration for Performance Optimization

This module provides M1-specific performance optimizations,
integrating with M1 hardware utilities and other performance modules.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import weakref

# Optional dependencies
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

# M1 hardware imports
try:
    from src.utils.hardware.m1_memory_optimizer import get_m1_memory_optimizer
    from src.utils.hardware.m1_cpu_optimizer import get_m1_cpu_optimizer
    from src.utils.hardware.m1_gpu_utils import get_m1_gpu_manager
    M1_HARDWARE_AVAILABLE = True
except ImportError:
    M1_HARDWARE_AVAILABLE = False
    get_m1_memory_optimizer = None
    get_m1_cpu_optimizer = None
    get_m1_gpu_manager = None

# Import performance modules
try:
    from .caching_strategies import get_ml_common_cache, CacheConfig, CacheStrategy
    from .memory_profiler import get_memory_profiler, MemoryOptimizationConfig
    from .async_patterns import get_async_operation_manager, AsyncOperationType
    PERFORMANCE_MODULES_AVAILABLE = True
except ImportError:
    PERFORMANCE_MODULES_AVAILABLE = False

logger = logging.getLogger(__name__)

class M1OperationType(Enum):
    """Types of M1-optimized operations."""
    MEMORY_INTENSIVE = "memory_intensive"
    CPU_INTENSIVE = "cpu_intensive"
    GPU_ACCELERATED = "gpu_accelerated"
    VECTORIZED = "vectorized"
    PARALLEL = "parallel"
    CACHED = "cached"

@dataclass
class M1PerformanceConfig:
    """Configuration for M1 performance optimization."""
    
    # Basic settings
    enable_m1_optimizations: bool = True
    enable_memory_optimization: bool = True
    enable_cpu_optimization: bool = True
    enable_gpu_optimization: bool = True
    enable_caching: bool = True
    enable_async_processing: bool = True
    
    # Memory settings
    memory_limit_gb: Optional[float] = None
    enable_memory_profiling: bool = True
    memory_threshold_mb: float = 1000.0
    gc_threshold: int = 10
    
    # CPU settings
    max_workers: int = 4
    use_performance_cores: bool = True
    enable_conservative_mode: bool = False
    
    # GPU settings
    enable_mps: bool = True
    use_gpu_for_large_arrays: bool = True
    gpu_threshold_elements: int = 1000000
    
    # Caching settings
    cache_ttl_seconds: int = 3600
    enable_memory_cache: bool = True
    enable_disk_cache: bool = False
    
    # Async settings
    enable_async_operations: bool = True
    max_concurrent_operations: int = 4
    operation_timeout: float = 30.0

class M1PerformanceOptimizer:
    """M1 performance optimizer with hardware-specific optimizations."""
    
    def __init__(self, config: Optional[M1PerformanceConfig] = None):
        self.config = config or M1PerformanceConfig()
        self.logger = logger.getChild('M1PerformanceOptimizer')
        
        # Initialize M1 hardware optimizers
        self._memory_optimizer = None
        self._cpu_optimizer = None
        self._gpu_manager = None
        
        if M1_HARDWARE_AVAILABLE and self.config.enable_m1_optimizations:
            if self.config.enable_memory_optimization:
                self._memory_optimizer = get_m1_memory_optimizer(
                    memory_limit_gb=self.config.memory_limit_gb
                )
            
            if self.config.enable_cpu_optimization:
                self._cpu_optimizer = get_m1_cpu_optimizer()
                if self.config.enable_conservative_mode:
                    self._cpu_optimizer.set_conservative_mode()
            
            if self.config.enable_gpu_optimization:
                self._gpu_manager = get_m1_gpu_manager()
        
        # Initialize performance modules
        self._cache = None
        self._memory_profiler = None
        self._async_manager = None
        
        if PERFORMANCE_MODULES_AVAILABLE:
            if self.config.enable_caching:
                cache_config = CacheConfig(
                    strategy=CacheStrategy.M1_OPTIMIZED,
                    enable_m1_optimizations=True,
                    ttl_seconds=self.config.cache_ttl_seconds
                )
                self._cache = get_ml_common_cache(cache_config)
            
            if self.config.enable_memory_profiling:
                memory_config = MemoryOptimizationConfig(
                    enable_m1_optimizations=True,
                    enable_memory_profiling=True
                )
                self._memory_profiler = get_memory_profiler(memory_config)
            
            if self.config.enable_async_processing:
                from .async_patterns import AsyncConfig
                async_config = AsyncConfig(
                    enable_m1_optimizations=True,
                    max_workers=self.config.max_workers,
                    max_concurrent_operations=self.config.max_concurrent_operations
                )
                self._async_manager = get_async_operation_manager(async_config)
    
    async def initialize(self):
        """Initialize the M1 performance optimizer."""
        if self._cache:
            await self._cache.initialize()
        
        if self._async_manager:
            await self._async_manager.initialize()
        
        if self._memory_optimizer:
            self._memory_optimizer.start_monitoring()
        
        self.logger.info("M1 performance optimizer initialized")
    
    def optimize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame for M1 hardware."""
        if not PANDAS_AVAILABLE:
            return df
        
        try:
            # Apply M1 memory optimization
            if self._memory_optimizer:
                df = self._memory_optimizer.optimize_dataframe_memory(df)
            
            # Apply M1 GPU optimization if data is large enough
            if (self._gpu_manager and 
                self.config.use_gpu_for_large_arrays and 
                df.size > self.config.gpu_threshold_elements):
                df = self._gpu_manager.optimize_dataframe_for_m1(df)
            
            return df
            
        except Exception as e:
            self.logger.warning(f"DataFrame optimization failed: {e}")
            return df
    
    def optimize_array(self, array: np.ndarray) -> np.ndarray:
        """Optimize NumPy array for M1 hardware."""
        if not NUMPY_AVAILABLE:
            return array
        
        try:
            # Apply M1 GPU optimization if available
            if self._gpu_manager and array.size > self.config.gpu_threshold_elements:
                return self._gpu_manager.optimize_tensor_operations(array)
            
            # Apply M1 memory optimization
            if self._memory_optimizer:
                # Convert to DataFrame for optimization, then back to array
                df = pd.DataFrame(array)
                optimized_df = self._memory_optimizer.optimize_dataframe_memory(df)
                return optimized_df.values
            
            return array
            
        except Exception as e:
            self.logger.warning(f"Array optimization failed: {e}")
            return array
    
    async def memory_intensive_operation(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Execute memory-intensive operation with M1 optimizations."""
        # Generate cache key
        cache_key = f"m1_memory_{func.__name__}_{hash(str(args) + str(sorted(kwargs.items())))}"
        
        # Check cache first
        if self._cache:
            cached_result = await self._cache.get(cache_key)
            if cached_result is not None:
                self.logger.debug(f"Cache hit for memory-intensive {func.__name__}")
                return cached_result
        
        # Execute with memory profiling
        if self._memory_profiler:
            with self._memory_profiler.memory_checkpoint(f"m1_memory_{func.__name__}"):
                if self._async_manager:
                    result = await self._async_manager.execute_async(
                        func,
                        *args,
                        operation_type=AsyncOperationType.MEMORY_INTENSIVE,
                        **kwargs
                    )
                else:
                    if asyncio.iscoroutinefunction(func):
                        result = await func(*args, **kwargs)
                    else:
                        result = func(*args, **kwargs)
        else:
            if self._async_manager:
                result = await self._async_manager.execute_async(
                    func,
                    *args,
                    operation_type=AsyncOperationType.MEMORY_INTENSIVE,
                    **kwargs
                )
            else:
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
        
        # Cache result
        if self._cache:
            await self._cache.set(cache_key, result, ttl=self.config.cache_ttl_seconds)
        
        return result
    
    async def cpu_intensive_operation(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Execute CPU-intensive operation with M1 optimizations."""
        # Generate cache key
        cache_key = f"m1_cpu_{func.__name__}_{hash(str(args) + str(sorted(kwargs.items())))}"
        
        # Check cache first
        if self._cache:
            cached_result = await self._cache.get(cache_key)
            if cached_result is not None:
                self.logger.debug(f"Cache hit for CPU-intensive {func.__name__}")
                return cached_result
        
        # Execute with CPU optimization
        if self._cpu_optimizer:
            optimized_func = self._cpu_optimizer.optimize_function_for_m1(func)
        else:
            optimized_func = func
        
        if self._async_manager:
            result = await self._async_manager.execute_async(
                optimized_func,
                *args,
                operation_type=AsyncOperationType.CPU_INTENSIVE,
                **kwargs
            )
        else:
            if asyncio.iscoroutinefunction(optimized_func):
                result = await optimized_func(*args, **kwargs)
            else:
                result = optimized_func(*args, **kwargs)
        
        # Cache result
        if self._cache:
            await self._cache.set(cache_key, result, ttl=self.config.cache_ttl_seconds)
        
        return result
    
    async def gpu_accelerated_operation(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Execute GPU-accelerated operation with M1 optimizations."""
        if not self._gpu_manager or not self._gpu_manager.mps_available:
            # Fallback to CPU operation
            return await self.cpu_intensive_operation(func, *args, **kwargs)
        
        # Generate cache key
        cache_key = f"m1_gpu_{func.__name__}_{hash(str(args) + str(sorted(kwargs.items())))}"
        
        # Check cache first
        if self._cache:
            cached_result = await self._cache.get(cache_key)
            if cached_result is not None:
                self.logger.debug(f"Cache hit for GPU-accelerated {func.__name__}")
                return cached_result
        
        # Execute with GPU optimization
        if self._async_manager:
            result = await self._async_manager.execute_async(
                func,
                *args,
                operation_type=AsyncOperationType.M1_OPTIMIZED,
                **kwargs
            )
        else:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
        
        # Cache result
        if self._cache:
            await self._cache.set(cache_key, result, ttl=self.config.cache_ttl_seconds)
        
        return result
    
    async def vectorized_operation(
        self,
        data: Any,
        operation_func: Callable,
        **kwargs
    ) -> Any:
        """Execute vectorized operation with M1 optimizations."""
        # Optimize data
        if isinstance(data, pd.DataFrame):
            optimized_data = self.optimize_dataframe(data)
        elif isinstance(data, np.ndarray):
            optimized_data = self.optimize_array(data)
        else:
            optimized_data = data
        
        # Generate cache key
        cache_key = f"m1_vectorized_{operation_func.__name__}_{hash(str(optimized_data.shape))}"
        
        # Check cache first
        if self._cache:
            cached_result = await self._cache.get(cache_key)
            if cached_result is not None:
                self.logger.debug(f"Cache hit for vectorized {operation_func.__name__}")
                return cached_result
        
        # Execute operation
        if self._async_manager:
            result = await self._async_manager.execute_async(
                operation_func,
                optimized_data,
                operation_type=AsyncOperationType.M1_OPTIMIZED,
                **kwargs
            )
        else:
            if asyncio.iscoroutinefunction(operation_func):
                result = await operation_func(optimized_data, **kwargs)
            else:
                result = operation_func(optimized_data, **kwargs)
        
        # Cache result
        if self._cache:
            await self._cache.set(cache_key, result, ttl=self.config.cache_ttl_seconds)
        
        return result
    
    async def parallel_operation(
        self,
        operations: List[Tuple[Callable, tuple, dict]],
        max_workers: Optional[int] = None
    ) -> List[Any]:
        """Execute multiple operations in parallel with M1 optimizations."""
        if not self._async_manager:
            # Fallback to sequential execution
            results = []
            for func, args, kwargs in operations:
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                results.append(result)
            return results
        
        # Execute operations in parallel
        async_operations = []
        for func, args, kwargs in operations:
            operation = (
                func,
                args,
                kwargs,
                AsyncOperationType.M1_OPTIMIZED
            )
            async_operations.append(operation)
        
        results = await self._async_manager.execute_batch_async(async_operations)
        return results
    
    def get_hardware_info(self) -> Dict[str, Any]:
        """Get M1 hardware information."""
        info = {
            'm1_available': M1_HARDWARE_AVAILABLE,
            'memory_optimizer_available': self._memory_optimizer is not None,
            'cpu_optimizer_available': self._cpu_optimizer is not None,
            'gpu_manager_available': self._gpu_manager is not None
        }
        
        if self._memory_optimizer:
            info['memory_stats'] = self._memory_optimizer.get_memory_stats()
        
        if self._cpu_optimizer:
            info['cpu_info'] = self._cpu_optimizer.get_cpu_info()
        
        if self._gpu_manager:
            info['gpu_info'] = self._gpu_manager.get_gpu_info()
        
        return info
    
    def cleanup(self):
        """Cleanup M1 optimizations."""
        if self._memory_optimizer:
            self._memory_optimizer.stop_monitoring()
        
        if self._cpu_optimizer:
            # CPU optimizer cleanup if needed
            pass
        
        if self._gpu_manager:
            # GPU manager cleanup if needed
            pass

# Global M1 performance optimizer
_global_m1_optimizer: Optional[M1PerformanceOptimizer] = None

def get_m1_performance_optimizer(config: Optional[M1PerformanceConfig] = None) -> M1PerformanceOptimizer:
    """Get the global M1 performance optimizer."""
    global _global_m1_optimizer
    
    if _global_m1_optimizer is None:
        _global_m1_optimizer = M1PerformanceOptimizer(config)
    
    return _global_m1_optimizer

def m1_optimize(data: Any) -> Any:
    """Optimize data for M1 hardware."""
    optimizer = get_m1_performance_optimizer()
    
    if isinstance(data, pd.DataFrame):
        return optimizer.optimize_dataframe(data)
    elif isinstance(data, np.ndarray):
        return optimizer.optimize_array(data)
    else:
        return data

def m1_cached(operation_type: M1OperationType = M1OperationType.MEMORY_INTENSIVE):
    """Decorator for M1 operations with caching."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            optimizer = get_m1_performance_optimizer()
            await optimizer.initialize()
            
            # Choose appropriate operation method
            if operation_type == M1OperationType.MEMORY_INTENSIVE:
                return await optimizer.memory_intensive_operation(func, *args, **kwargs)
            elif operation_type == M1OperationType.CPU_INTENSIVE:
                return await optimizer.cpu_intensive_operation(func, *args, **kwargs)
            elif operation_type == M1OperationType.GPU_ACCELERATED:
                return await optimizer.gpu_accelerated_operation(func, *args, **kwargs)
            elif operation_type == M1OperationType.VECTORIZED:
                return await optimizer.vectorized_operation(args[0], func, **kwargs)
            else:
                # Default to memory intensive
                return await optimizer.memory_intensive_operation(func, *args, **kwargs)
        
        return async_wrapper
    return decorator

def m1_async_execute(
    operation_type: M1OperationType = M1OperationType.MEMORY_INTENSIVE,
    timeout: float = 30.0
):
    """Decorator for async M1 operations."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            optimizer = get_m1_performance_optimizer()
            await optimizer.initialize()
            
            if optimizer._async_manager:
                return await optimizer._async_manager.execute_async(
                    func,
                    *args,
                    operation_type=AsyncOperationType.M1_OPTIMIZED,
                    timeout=timeout,
                    **kwargs
                )
            else:
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)
        
        return async_wrapper
    return decorator

# Convenience functions
async def m1_memory_operation(func: Callable, *args, **kwargs) -> Any:
    """Execute memory-intensive operation with M1 optimizations."""
    optimizer = get_m1_performance_optimizer()
    await optimizer.initialize()
    return await optimizer.memory_intensive_operation(func, *args, **kwargs)

async def m1_cpu_operation(func: Callable, *args, **kwargs) -> Any:
    """Execute CPU-intensive operation with M1 optimizations."""
    optimizer = get_m1_performance_optimizer()
    await optimizer.initialize()
    return await optimizer.cpu_intensive_operation(func, *args, **kwargs)

async def m1_gpu_operation(func: Callable, *args, **kwargs) -> Any:
    """Execute GPU-accelerated operation with M1 optimizations."""
    optimizer = get_m1_performance_optimizer()
    await optimizer.initialize()
    return await optimizer.gpu_accelerated_operation(func, *args, **kwargs)

async def m1_vectorized_operation(data: Any, operation_func: Callable, **kwargs) -> Any:
    """Execute vectorized operation with M1 optimizations."""
    optimizer = get_m1_performance_optimizer()
    await optimizer.initialize()
    return await optimizer.vectorized_operation(data, operation_func, **kwargs)

async def m1_parallel_operations(
    operations: List[Tuple[Callable, tuple, dict]],
    max_workers: Optional[int] = None
) -> List[Any]:
    """Execute multiple operations in parallel with M1 optimizations."""
    optimizer = get_m1_performance_optimizer()
    await optimizer.initialize()
    return await optimizer.parallel_operation(operations, max_workers)

def get_m1_hardware_info() -> Dict[str, Any]:
    """Get M1 hardware information."""
    optimizer = get_m1_performance_optimizer()
    return optimizer.get_hardware_info()