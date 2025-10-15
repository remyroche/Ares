"""
Advanced Async/Await Patterns for ML Common Operations

This module provides comprehensive async patterns for I/O operations,
integrating with M1 hardware optimizations and VectorBT for efficient processing.
"""

import asyncio
import logging
import time
import threading
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Awaitable
import weakref
from pathlib import Path
import json
import pickle

# Optional dependencies
try:
    import aiofiles
    AIOFILES_AVAILABLE = True
except ImportError:
    AIOFILES_AVAILABLE = False

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

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

# Import M1 optimizations
try:
    from src.utils.hardware.m1_memory_optimizer import get_m1_memory_optimizer
    from src.utils.hardware.m1_cpu_optimizer import get_m1_cpu_optimizer
    from src.utils.hardware.m1_gpu_utils import get_m1_gpu_manager
    M1_OPTIMIZATIONS_AVAILABLE = True
except ImportError:
    M1_OPTIMIZATIONS_AVAILABLE = False

# Import VectorBT optimizations
try:
    from src.feature_generation.utils.vectorbt_rolling_optimizer import VectorBTRollingOptimizer, get_vectorbt_rolling_optimizer
    VECTORBT_OPTIMIZATIONS_AVAILABLE = True
except ImportError:
    VECTORBT_OPTIMIZATIONS_AVAILABLE = False

# Import caching and memory profiling
try:
    from .caching_strategies import get_ml_common_cache, CacheConfig, CacheStrategy
    from .memory_profiler import get_memory_profiler, MemoryOptimizationConfig
    PERFORMANCE_MODULES_AVAILABLE = True
except ImportError:
    PERFORMANCE_MODULES_AVAILABLE = False

logger = logging.getLogger(__name__)

class AsyncOperationType(Enum):
    """Types of async operations."""
    FILE_IO = "file_io"
    NETWORK_IO = "network_io"
    CPU_INTENSIVE = "cpu_intensive"
    MEMORY_INTENSIVE = "memory_intensive"
    VECTORBT_OPERATION = "vectorbt_operation"
    M1_OPTIMIZED = "m1_optimized"

class AsyncExecutionStrategy(Enum):
    """Async execution strategies."""
    THREAD_POOL = "thread_pool"
    PROCESS_POOL = "process_pool"
    ASYNCIO = "asyncio"
    HYBRID = "hybrid"
    ADAPTIVE = "adaptive"

@dataclass
class AsyncConfig:
    """Configuration for async operations."""
    
    # Basic settings
    enable_async: bool = True
    max_workers: int = 4
    max_concurrent_operations: int = 10
    timeout_seconds: float = 30.0
    
    # Execution strategy
    execution_strategy: AsyncExecutionStrategy = AsyncExecutionStrategy.ADAPTIVE
    prefer_threads: bool = True
    prefer_processes: bool = False
    
    # M1 optimizations
    enable_m1_optimizations: bool = True
    use_m1_cpu_optimizer: bool = True
    use_m1_memory_optimizer: bool = True
    use_m1_gpu_optimizer: bool = True
    
    # VectorBT optimizations
    enable_vectorbt_optimizations: bool = True
    use_vectorbt_rolling: bool = True
    
    # Performance optimizations
    enable_caching: bool = True
    enable_memory_profiling: bool = True
    enable_progressive_loading: bool = True
    
    # I/O settings
    enable_file_async: bool = True
    enable_network_async: bool = True
    chunk_size: int = 8192
    buffer_size: int = 65536
    
    # Error handling
    enable_retry: bool = True
    max_retries: int = 3
    retry_delay: float = 1.0
    exponential_backoff: bool = True

@dataclass
class AsyncOperation:
    """Represents an async operation."""
    
    operation_id: str
    operation_type: AsyncOperationType
    function: Callable
    args: tuple
    kwargs: dict
    priority: int = 0
    timeout: Optional[float] = None
    retry_count: int = 0
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Any = None
    error: Optional[Exception] = None
    
    def is_completed(self) -> bool:
        """Check if operation is completed."""
        return self.completed_at is not None
    
    def is_failed(self) -> bool:
        """Check if operation failed."""
        return self.error is not None
    
    def execution_time(self) -> Optional[float]:
        """Get execution time in seconds."""
        if self.started_at and self.completed_at:
            return self.completed_at - self.started_at
        return None

class AsyncExecutor(ABC):
    """Base class for async executors."""
    
    def __init__(self, config: AsyncConfig):
        self.config = config
        self.logger = logger.getChild(self.__class__.__name__)
    
    @abstractmethod
    async def execute(self, operation: AsyncOperation) -> Any:
        """Execute an async operation."""
        pass
    
    @abstractmethod
    async def execute_batch(self, operations: List[AsyncOperation]) -> List[Any]:
        """Execute a batch of async operations."""
        pass

class ThreadPoolAsyncExecutor(AsyncExecutor):
    """Thread pool-based async executor."""
    
    def __init__(self, config: AsyncConfig):
        super().__init__(config)
        self._executor = ThreadPoolExecutor(max_workers=config.max_workers)
        self._m1_cpu_optimizer = None
        
        if M1_OPTIMIZATIONS_AVAILABLE and config.enable_m1_optimizations and config.use_m1_cpu_optimizer:
            self._m1_cpu_optimizer = get_m1_cpu_optimizer()
    
    async def execute(self, operation: AsyncOperation) -> Any:
        """Execute operation in thread pool."""
        try:
            operation.started_at = time.time()
            
            # Apply M1 optimizations if available
            if self._m1_cpu_optimizer:
                optimized_func = self._m1_cpu_optimizer.optimize_function_for_m1(operation.function)
            else:
                optimized_func = operation.function
            
            # Execute in thread pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self._executor,
                optimized_func,
                *operation.args,
                **operation.kwargs
            )
            
            operation.completed_at = time.time()
            operation.result = result
            return result
            
        except Exception as e:
            operation.error = e
            operation.completed_at = time.time()
            raise
    
    async def execute_batch(self, operations: List[AsyncOperation]) -> List[Any]:
        """Execute batch of operations in thread pool."""
        tasks = [self.execute(op) for op in operations]
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    def __del__(self):
        """Cleanup thread pool."""
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=False)

class ProcessPoolAsyncExecutor(AsyncExecutor):
    """Process pool-based async executor."""
    
    def __init__(self, config: AsyncConfig):
        super().__init__(config)
        self._executor = ProcessPoolExecutor(max_workers=config.max_workers)
    
    async def execute(self, operation: AsyncOperation) -> Any:
        """Execute operation in process pool."""
        try:
            operation.started_at = time.time()
            
            # Execute in process pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self._executor,
                operation.function,
                *operation.args,
                **operation.kwargs
            )
            
            operation.completed_at = time.time()
            operation.result = result
            return result
            
        except Exception as e:
            operation.error = e
            operation.completed_at = time.time()
            raise
    
    async def execute_batch(self, operations: List[AsyncOperation]) -> List[Any]:
        """Execute batch of operations in process pool."""
        tasks = [self.execute(op) for op in operations]
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    def __del__(self):
        """Cleanup process pool."""
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=False)

class AsyncioExecutor(AsyncExecutor):
    """Pure asyncio-based executor."""
    
    def __init__(self, config: AsyncConfig):
        super().__init__(config)
        self._semaphore = asyncio.Semaphore(config.max_concurrent_operations)
    
    async def execute(self, operation: AsyncOperation) -> Any:
        """Execute operation using asyncio."""
        async with self._semaphore:
            try:
                operation.started_at = time.time()
                
                # Check if function is already async
                if asyncio.iscoroutinefunction(operation.function):
                    result = await operation.function(*operation.args, **operation.kwargs)
                else:
                    # Run sync function in thread pool
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(
                        None,
                        operation.function,
                        *operation.args,
                        **operation.kwargs
                    )
                
                operation.completed_at = time.time()
                operation.result = result
                return result
                
            except Exception as e:
                operation.error = e
                operation.completed_at = time.time()
                raise
    
    async def execute_batch(self, operations: List[AsyncOperation]) -> List[Any]:
        """Execute batch of operations using asyncio."""
        tasks = [self.execute(op) for op in operations]
        return await asyncio.gather(*tasks, return_exceptions=True)

class HybridAsyncExecutor(AsyncExecutor):
    """Hybrid executor that chooses the best strategy for each operation."""
    
    def __init__(self, config: AsyncConfig):
        super().__init__(config)
        self._thread_executor = ThreadPoolAsyncExecutor(config)
        self._process_executor = ProcessPoolAsyncExecutor(config)
        self._asyncio_executor = AsyncioExecutor(config)
    
    async def execute(self, operation: AsyncOperation) -> Any:
        """Execute operation using the best strategy."""
        executor = self._choose_executor(operation)
        return await executor.execute(operation)
    
    async def execute_batch(self, operations: List[AsyncOperation]) -> List[Any]:
        """Execute batch using appropriate executors."""
        # Group operations by type
        thread_ops = []
        process_ops = []
        asyncio_ops = []
        
        for op in operations:
            executor = self._choose_executor(op)
            if executor == self._thread_executor:
                thread_ops.append(op)
            elif executor == self._process_executor:
                process_ops.append(op)
            else:
                asyncio_ops.append(op)
        
        # Execute each group
        results = []
        if thread_ops:
            results.extend(await self._thread_executor.execute_batch(thread_ops))
        if process_ops:
            results.extend(await self._process_executor.execute_batch(process_ops))
        if asyncio_ops:
            results.extend(await self._asyncio_executor.execute_batch(asyncio_ops))
        
        return results
    
    def _choose_executor(self, operation: AsyncOperation) -> AsyncExecutor:
        """Choose the best executor for an operation."""
        if operation.operation_type == AsyncOperationType.CPU_INTENSIVE:
            return self._process_executor
        elif operation.operation_type == AsyncOperationType.MEMORY_INTENSIVE:
            return self._thread_executor
        elif operation.operation_type == AsyncOperationType.VECTORBT_OPERATION:
            return self._thread_executor
        elif operation.operation_type == AsyncOperationType.M1_OPTIMIZED:
            return self._thread_executor
        else:
            return self._asyncio_executor

class AdaptiveAsyncExecutor(AsyncExecutor):
    """Adaptive executor that learns the best strategy."""
    
    def __init__(self, config: AsyncConfig):
        super().__init__(config)
        self._hybrid_executor = HybridAsyncExecutor(config)
        self._performance_history: Dict[str, List[float]] = {}
        self._strategy_preferences: Dict[str, AsyncExecutionStrategy] = {}
    
    async def execute(self, operation: AsyncOperation) -> Any:
        """Execute operation using adaptive strategy."""
        # Choose strategy based on history
        strategy = self._choose_adaptive_strategy(operation)
        
        # Execute with chosen strategy
        start_time = time.time()
        try:
            if strategy == AsyncExecutionStrategy.THREAD_POOL:
                result = await self._hybrid_executor._thread_executor.execute(operation)
            elif strategy == AsyncExecutionStrategy.PROCESS_POOL:
                result = await self._hybrid_executor._process_executor.execute(operation)
            else:
                result = await self._hybrid_executor._asyncio_executor.execute(operation)
            
            # Record performance
            execution_time = time.time() - start_time
            self._record_performance(operation, strategy, execution_time)
            
            return result
            
        except Exception as e:
            # Record failure
            self._record_performance(operation, strategy, float('inf'))
            raise
    
    async def execute_batch(self, operations: List[AsyncOperation]) -> List[Any]:
        """Execute batch using adaptive strategies."""
        results = []
        for operation in operations:
            try:
                result = await self.execute(operation)
                results.append(result)
            except Exception as e:
                results.append(e)
        return results
    
    def _choose_adaptive_strategy(self, operation: AsyncOperation) -> AsyncExecutionStrategy:
        """Choose strategy based on performance history."""
        operation_key = f"{operation.operation_type.value}_{operation.function.__name__}"
        
        if operation_key in self._strategy_preferences:
            return self._strategy_preferences[operation_key]
        
        # Default strategy based on operation type
        if operation.operation_type == AsyncOperationType.CPU_INTENSIVE:
            return AsyncExecutionStrategy.PROCESS_POOL
        elif operation.operation_type == AsyncOperationType.MEMORY_INTENSIVE:
            return AsyncExecutionStrategy.THREAD_POOL
        else:
            return AsyncExecutionStrategy.ASYNCIO
    
    def _record_performance(self, operation: AsyncOperation, strategy: AsyncExecutionStrategy, execution_time: float):
        """Record performance for adaptive learning."""
        operation_key = f"{operation.operation_type.value}_{operation.function.__name__}"
        strategy_key = f"{operation_key}_{strategy.value}"
        
        if strategy_key not in self._performance_history:
            self._performance_history[strategy_key] = []
        
        self._performance_history[strategy_key].append(execution_time)
        
        # Keep only recent history
        if len(self._performance_history[strategy_key]) > 100:
            self._performance_history[strategy_key] = self._performance_history[strategy_key][-50:]
        
        # Update strategy preference
        self._update_strategy_preference(operation_key)
    
    def _update_strategy_preference(self, operation_key: str):
        """Update strategy preference based on performance."""
        strategies = [AsyncExecutionStrategy.THREAD_POOL, AsyncExecutionStrategy.PROCESS_POOL, AsyncExecutionStrategy.ASYNCIO]
        best_strategy = None
        best_avg_time = float('inf')
        
        for strategy in strategies:
            strategy_key = f"{operation_key}_{strategy.value}"
            if strategy_key in self._performance_history:
                avg_time = sum(self._performance_history[strategy_key]) / len(self._performance_history[strategy_key])
                if avg_time < best_avg_time:
                    best_avg_time = avg_time
                    best_strategy = strategy
        
        if best_strategy:
            self._strategy_preferences[operation_key] = best_strategy

class AsyncOperationManager:
    """Manager for async operations."""
    
    def __init__(self, config: Optional[AsyncConfig] = None):
        self.config = config or AsyncConfig()
        self.logger = logger.getChild('AsyncOperationManager')
        self._executor: Optional[AsyncExecutor] = None
        self._cache = None
        self._memory_profiler = None
        
        # Initialize performance modules if available
        if PERFORMANCE_MODULES_AVAILABLE:
            if self.config.enable_caching:
                cache_config = CacheConfig(
                    strategy=CacheStrategy.MEMORY,
                    enable_m1_optimizations=self.config.enable_m1_optimizations,
                    enable_vectorbt_optimizations=self.config.enable_vectorbt_optimizations
                )
                self._cache = get_ml_common_cache(cache_config)
            
            if self.config.enable_memory_profiling:
                memory_config = MemoryOptimizationConfig(
                    enable_m1_optimizations=self.config.enable_m1_optimizations,
                    enable_vectorbt_optimizations=self.config.enable_vectorbt_optimizations
                )
                self._memory_profiler = get_memory_profiler(memory_config)
    
    async def initialize(self):
        """Initialize the async operation manager."""
        if self._executor is not None:
            return
        
        # Choose executor based on strategy
        if self.config.execution_strategy == AsyncExecutionStrategy.THREAD_POOL:
            self._executor = ThreadPoolAsyncExecutor(self.config)
        elif self.config.execution_strategy == AsyncExecutionStrategy.PROCESS_POOL:
            self._executor = ProcessPoolAsyncExecutor(self.config)
        elif self.config.execution_strategy == AsyncExecutionStrategy.ASYNCIO:
            self._executor = AsyncioExecutor(self.config)
        elif self.config.execution_strategy == AsyncExecutionStrategy.HYBRID:
            self._executor = HybridAsyncExecutor(self.config)
        else:  # ADAPTIVE
            self._executor = AdaptiveAsyncExecutor(self.config)
        
        # Initialize cache and memory profiler
        if self._cache:
            await self._cache.initialize()
        
        self.logger.info(f"Async operation manager initialized with {self.config.execution_strategy.value} strategy")
    
    async def execute_async(
        self,
        func: Callable,
        *args,
        operation_type: AsyncOperationType = AsyncOperationType.CPU_INTENSIVE,
        priority: int = 0,
        timeout: Optional[float] = None,
        use_cache: bool = True,
        **kwargs
    ) -> Any:
        """Execute function asynchronously."""
        if not self._executor:
            await self.initialize()
        
        # Generate operation ID
        operation_id = f"{func.__name__}_{int(time.time() * 1000)}"
        
        # Check cache first
        if use_cache and self._cache:
            cache_key = f"{func.__name__}_{hash(str(args) + str(sorted(kwargs.items())))}"
            cached_result = await self._cache.get(cache_key)
            if cached_result is not None:
                self.logger.debug(f"Cache hit for {func.__name__}")
                return cached_result
        
        # Create operation
        operation = AsyncOperation(
            operation_id=operation_id,
            operation_type=operation_type,
            function=func,
            args=args,
            kwargs=kwargs,
            priority=priority,
            timeout=timeout or self.config.timeout_seconds
        )
        
        # Execute with memory profiling if enabled
        if self._memory_profiler:
            with self._memory_profiler.memory_checkpoint(f"async_{func.__name__}"):
                result = await self._executor.execute(operation)
        else:
            result = await self._executor.execute(operation)
        
        # Cache result if enabled
        if use_cache and self._cache and not operation.is_failed():
            cache_key = f"{func.__name__}_{hash(str(args) + str(sorted(kwargs.items())))}"
            await self._cache.set(cache_key, result, ttl=3600)  # 1 hour TTL
        
        return result
    
    async def execute_batch_async(
        self,
        operations: List[Tuple[Callable, tuple, dict, AsyncOperationType]],
        use_cache: bool = True
    ) -> List[Any]:
        """Execute batch of operations asynchronously."""
        if not self._executor:
            await self.initialize()
        
        # Create operation objects
        operation_objects = []
        for i, (func, args, kwargs, op_type) in enumerate(operations):
            operation_id = f"{func.__name__}_{i}_{int(time.time() * 1000)}"
            operation = AsyncOperation(
                operation_id=operation_id,
                operation_type=op_type,
                function=func,
                args=args,
                kwargs=kwargs
            )
            operation_objects.append(operation)
        
        # Execute batch
        if self._memory_profiler:
            with self._memory_profiler.memory_checkpoint("async_batch"):
                results = await self._executor.execute_batch(operation_objects)
        else:
            results = await self._executor.execute_batch(operation_objects)
        
        return results

# Global async operation manager
_global_async_manager: Optional[AsyncOperationManager] = None

def get_async_operation_manager(config: Optional[AsyncConfig] = None) -> AsyncOperationManager:
    """Get the global async operation manager."""
    global _global_async_manager
    
    if _global_async_manager is None:
        _global_async_manager = AsyncOperationManager(config)
    
    return _global_async_manager

def async_execute(
    operation_type: AsyncOperationType = AsyncOperationType.CPU_INTENSIVE,
    priority: int = 0,
    timeout: Optional[float] = None,
    use_cache: bool = True
):
    """Decorator for async execution."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            manager = get_async_operation_manager()
            return await manager.execute_async(
                func, *args,
                operation_type=operation_type,
                priority=priority,
                timeout=timeout,
                use_cache=use_cache,
                **kwargs
            )
        
        return async_wrapper
    return decorator

# Convenience functions
async def execute_async(
    func: Callable,
    *args,
    operation_type: AsyncOperationType = AsyncOperationType.CPU_INTENSIVE,
    **kwargs
) -> Any:
    """Execute function asynchronously."""
    manager = get_async_operation_manager()
    return await manager.execute_async(func, *args, operation_type=operation_type, **kwargs)

async def execute_batch_async(
    operations: List[Tuple[Callable, tuple, dict, AsyncOperationType]]
) -> List[Any]:
    """Execute batch of operations asynchronously."""
    manager = get_async_operation_manager()
    return await manager.execute_batch_async(operations)

# File I/O async operations
async def async_read_file(filepath: str, encoding: str = 'utf-8') -> str:
    """Read file asynchronously."""
    if AIOFILES_AVAILABLE:
        async with aiofiles.open(filepath, 'r', encoding=encoding) as f:
            return await f.read()
    else:
        # Fallback to sync with thread pool
        def _read_file():
            with open(filepath, 'r', encoding=encoding) as f:
                return f.read()
        
        return await execute_async(_read_file, operation_type=AsyncOperationType.FILE_IO)

async def async_write_file(filepath: str, content: str, encoding: str = 'utf-8') -> None:
    """Write file asynchronously."""
    if AIOFILES_AVAILABLE:
        async with aiofiles.open(filepath, 'w', encoding=encoding) as f:
            await f.write(content)
    else:
        # Fallback to sync with thread pool
        def _write_file():
            with open(filepath, 'w', encoding=encoding) as f:
                f.write(content)
        
        await execute_async(_write_file, operation_type=AsyncOperationType.FILE_IO)

async def async_read_json(filepath: str) -> Dict[str, Any]:
    """Read JSON file asynchronously."""
    content = await async_read_file(filepath)
    return json.loads(content)

async def async_write_json(filepath: str, data: Dict[str, Any], indent: int = 2) -> None:
    """Write JSON file asynchronously."""
    content = json.dumps(data, indent=indent)
    await async_write_file(filepath, content)

# Network I/O async operations
async def async_http_get(url: str, timeout: float = 30.0) -> Dict[str, Any]:
    """Make HTTP GET request asynchronously."""
    if AIOHTTP_AVAILABLE:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
            async with session.get(url) as response:
                return await response.json()
    else:
        # Fallback to sync with thread pool
        import requests
        def _http_get():
            response = requests.get(url, timeout=timeout)
            return response.json()
        
        return await execute_async(_http_get, operation_type=AsyncOperationType.NETWORK_IO)

# Data processing async operations
async def async_process_dataframe(
    df: pd.DataFrame,
    processor_func: Callable,
    operation_type: AsyncOperationType = AsyncOperationType.MEMORY_INTENSIVE
) -> pd.DataFrame:
    """Process DataFrame asynchronously."""
    return await execute_async(
        processor_func,
        df,
        operation_type=operation_type
    )

async def async_vectorbt_operation(
    data: Any,
    operation_func: Callable,
    **kwargs
) -> Any:
    """Execute VectorBT operation asynchronously."""
    return await execute_async(
        operation_func,
        data,
        operation_type=AsyncOperationType.VECTORBT_OPERATION,
        **kwargs
    )

async def async_m1_optimized_operation(
    data: Any,
    operation_func: Callable,
    **kwargs
) -> Any:
    """Execute M1-optimized operation asynchronously."""
    return await execute_async(
        operation_func,
        data,
        operation_type=AsyncOperationType.M1_OPTIMIZED,
        **kwargs
    )