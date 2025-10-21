"""
Hardware-Optimized Parallel Processing for Apple Silicon

This module provides advanced parallel processing utilities with full hardware integration
for Apple Silicon and other platforms. It leverages the comprehensive hardware optimization
system for maximum performance and efficiency.
"""

import asyncio
import logging
import multiprocessing as mp
import os
import platform
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from functools import partial, wraps
from typing import Callable, Any, Optional, Dict, List, Union
from collections.abc import Iterable

import psutil
import numpy as np
import pandas as pd

# Hardware optimization imports
from .hardware.integrated_hardware_manager import (
    get_integrated_hardware_manager, IntegratedHardwareManager, WorkloadType
)
from .hardware.unified_hardware_manager import (
    get_unified_hardware_manager, UnifiedHardwareManager, OptimizationLevel
)
from .hardware.adaptive_optimization_engine import (
    get_adaptive_optimization_engine, AdaptiveOptimizationEngine, OptimizationTarget
)
from .hardware.advanced_memory_manager import (
    get_advanced_memory_manager, AdvancedMemoryManager
)
from .hardware.advanced_memory_optimizer import (
    MemoryStrategy
)
from .hardware.enhanced_gpu_manager import (
    get_enhanced_gpu_manager, EnhancedM1GPUManager, GPUOperationType
)

# VectorBT imports for native optimization
try:
    import vectorbt as vbt
    from vectorbt.generic import rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max, rolling_sum, rolling_apply, rolling_corr, rolling_cov
    from vectorbt.generic import scale, rank, zscore, winsorize, clip, quantile
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None
    rolling_mean = None
    rolling_std = None
    rolling_var = None
    rolling_min = None
    rolling_max = None
    rolling_sum = None
    rolling_apply = None
    rolling_corr = None
    rolling_cov = None
    scale = None
    rank = None
    zscore = None
    winsorize = None
    clip = None
    quantile = None
    import warnings
    warnings.warn("VectorBT not available. Install with: pip install vectorbt for optimized performance")

except ImportError:

    cp = None

logger = logging.getLogger(__name__)

# Hardware-aware decorators
def hardware_optimized(workload_type: WorkloadType, optimization_level: OptimizationLevel = OptimizationLevel.BALANCED):
    """Decorator for hardware-optimized functions."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with get_integrated_hardware_manager().optimization_context(workload_type, optimization_level):
                return func(*args, **kwargs)
        return wrapper
    return decorator

def memory_efficient_processing(memory_threshold_mb: float = 200.0, auto_cleanup: bool = True):
    """Decorator for memory-efficient processing."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Check memory pressure
            memory_pressure = _get_memory_pressure()
            if memory_pressure > 0.8:
                # Use aggressive memory optimization
                with get_advanced_memory_manager().memory_context(MemoryStrategy.AGGRESSIVE):
                    return func(*args, **kwargs)
            else:
                return func(*args, **kwargs)
        return wrapper
    return decorator

def gpu_accelerated(operation_type: GPUOperationType = GPUOperationType.MATRIX_MULTIPLICATION):
    """Decorator for GPU-accelerated operations."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            gpu_manager = get_enhanced_gpu_manager()
            if gpu_manager.is_gpu_available() and _is_gpu_suitable(func, args):
                return gpu_manager.execute_gpu_operation(func, operation_type, *args, **kwargs)
            else:
                return func(*args, **kwargs)
        return wrapper
    return decorator

def adaptive_workload_optimization():
    """Decorator for adaptive workload optimization."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            adaptive_engine = get_adaptive_optimization_engine()
            # Get optimal strategy based on function and data characteristics
            strategy = adaptive_engine.get_optimal_strategy(
                func.__name__,
                {
                    'memory_pressure': _get_memory_pressure(),
                    'data_size': _estimate_data_size(args),
                    'function_name': func.__name__
                }
            )
            
            # Apply strategy recommendations
            if strategy.get('use_gpu', False) and _is_gpu_suitable(func, args):
                return gpu_manager.execute_gpu_operation(func, GPUOperationType.MATRIX_MULTIPLICATION, *args, **kwargs)
            else:
                # Use CPU optimization with recommended settings
                with get_unified_hardware_manager().optimization_context(
                    WorkloadType.DATA_PROCESSING, 
                    OptimizationLevel.AGGRESSIVE
                ):
                    return func(*args, **kwargs)
        return wrapper
    return decorator

def _get_memory_pressure() -> float:
    """Get current memory pressure (0.0 to 1.0)."""
    try:
        memory = psutil.virtual_memory()
        return memory.percent / 100.0
    except:
        return 0.5

def _is_gpu_suitable(func: Callable, args: tuple) -> bool:
    """Determine if function is suitable for GPU acceleration."""
    # Check if function involves numerical operations on large arrays
    for arg in args:
        if isinstance(arg, (np.ndarray, pd.DataFrame)):
            if hasattr(arg, 'shape') and len(arg.shape) > 0:
                # GPU suitable for large numerical operations
                return arg.size > 1000 and np.issubdtype(getattr(arg, 'dtype', np.float64), np.number)
    return False

def _estimate_data_size(args: tuple) -> int:
    """Estimate total data size in elements."""
    total_size = 0
    for arg in args:
        if hasattr(arg, 'size'):
            total_size += arg.size
        elif hasattr(arg, '__len__'):
            total_size += len(arg)
    return total_size

class MacM1ParallelOptimizer:
    """
    Parallel processing optimizer with Apple Silicon awareness.
    """

    def __init__(self, max_workers: int | None = None, *, chunk_size: int = 1000, use_process_pool: bool = True, memory_limit_mb: int = 2048, enable_hardware_optimization: bool = True) -> None:
        """
        Initialize the hardware-optimized parallel optimizer.

        Args:
            max_workers: Maximum parallel workers. Defaults to cpu_count() or 4 on M1.
            chunk_size: Target chunk size for DataFrame splitting.
            use_process_pool: Use processes if True, threads if False.
            memory_limit_mb: Logical memory budget per worker (MB).
            enable_hardware_optimization: Enable hardware-aware optimizations.
        """
        cpu_count = mp.cpu_count() or 1
        self.is_m1_mac: bool = self._detect_m1_mac()
        default_workers = 4 if self.is_m1_mac else min(8, cpu_count)
        self.max_workers: int = max_workers if max_workers and max_workers > 0 else default_workers
        self.chunk_size: int = max(1, chunk_size)
        self.use_process_pool: bool = bool(use_process_pool)
        self.memory_limit_mb: int = max(128, memory_limit_mb)
        self.enable_hardware_optimization: bool = enable_hardware_optimization
        
        # Initialize hardware managers
        if self.enable_hardware_optimization:
            self.hardware_manager: IntegratedHardwareManager = get_integrated_hardware_manager()
            self.unified_manager: UnifiedHardwareManager = get_unified_hardware_manager()
            self.adaptive_engine: AdaptiveOptimizationEngine = get_adaptive_optimization_engine()
            self.memory_manager: AdvancedM1MemoryOptimizer = get_advanced_memory_manager()
            self.gpu_manager: EnhancedM1GPUManager = get_enhanced_gpu_manager()
            
            # Configure hardware for parallel processing
            self._configure_hardware_for_parallel_processing()
        else:
            self.hardware_manager = None
            self.unified_manager = None
            self.adaptive_engine = None
            self.memory_manager = None
            self.gpu_manager = None
        
        if self.is_m1_mac:
            logger.info('🍎 Detected Apple Silicon - applying M1-specific optimizations')
            self.memory_limit_mb = min(self.memory_limit_mb * 2, 8192)
        
        logger.info('🔧 Initialized Hardware-Optimized Parallel Optimizer:')
        logger.info(f'   Max workers: {self.max_workers}')
        logger.info(f'   Chunk size: {self.chunk_size}')
        logger.info(f"   Pool type: {('Process' if self.use_process_pool else 'Thread')}")
        logger.info(f'   Memory limit per worker: {self.memory_limit_mb} MB')
        logger.info(f'   M1 Mac detected: {self.is_m1_mac}')
        logger.info(f'   Hardware optimization enabled: {self.enable_hardware_optimization}')

    def _configure_hardware_for_parallel_processing(self):
        """Configure hardware managers for optimal parallel processing."""
        try:
            # Configure for data processing workload
            self.unified_manager.optimize_for_workload(WorkloadType.DATA_PROCESSING, OptimizationLevel.AGGRESSIVE)
            
            # Set up memory optimization for parallel processing
            self.memory_manager.set_memory_strategy(MemoryStrategy.ADAPTIVE)
            
            # Configure GPU for suitable operations
            if self.gpu_manager.is_gpu_available():
                self.gpu_manager.configure_for_operation(GPUOperationType.MATRIX_MULTIPLICATION)
            
            logger.info('✅ Hardware configured for parallel processing')
        except Exception as e:
            logger.warning(f'⚠️ Hardware configuration failed: {e}')

    def _get_optimal_strategy(self, data_size: int, operation_type: str = 'data_processing') -> Dict[str, Any]:
        """Get optimal processing strategy based on data characteristics and hardware."""
        if not self.enable_hardware_optimization:
            return {
                'use_gpu': False,
                'num_threads': self.max_workers,
                'chunk_size': self.chunk_size,
                'memory_strategy': 'balanced'
            }
        
        try:
            strategy = self.adaptive_engine.get_optimal_strategy(
                operation_type,
                {
                    'memory_pressure': _get_memory_pressure(),
                    'data_size': data_size,
                    'max_workers': self.max_workers
                }
            )
            return strategy
        except Exception as e:
            logger.warning(f'Failed to get optimal strategy: {e}')
            return {
                'use_gpu': False,
                'num_threads': self.max_workers,
                'chunk_size': self.chunk_size,
                'memory_strategy': 'balanced'
            }

    def _detect_m1_mac(self) -> bool:
        """
        Detect if running on Apple Silicon macOS.
        """
        try:
            if platform.system() != 'Darwin':
                return False
            machine = platform.machine().lower()
            if machine in {'arm64', 'aarch64'}:
                return True
            try:
                result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], capture_output = True, text = True, check = False)
                return 'apple' in result.stdout.lower()
            except Exception:
                return False
        except Exception:
            return False

    def _get_optimal_chunk_size(self, data_size: int) -> int:
        """
        Calculate optimal chunk size for parallel processing.
        """
        base_chunk_size = self.chunk_size * (2 if self.is_m1_mac else 1)
        denom = max(1, self.max_workers * 4)
        adaptive = max(1, data_size // denom)
        optimal = max(base_chunk_size, adaptive)
        return min(optimal, 10000)

    def _split_dataframe(self, df: pd.DataFrame, *, chunk_size: int | None = None) -> list[pd.DataFrame]:
        """
        Split DataFrame into chunks.
        """
        size = len(df)
        if size == 0:
            return [df.copy()]
        if chunk_size is None:
            chunk_size = self._get_optimal_chunk_size(size)
        chunks: list[pd.DataFrame] = []
        for i in range(0, size, chunk_size):
            chunks.append(df.iloc[i:i + chunk_size].copy())
        logger.debug(f'📦 Split DataFrame into {len(chunks)} chunks of ~{chunk_size} rows each')
        return chunks

    def _merge_chunks(self, chunks: Iterable[pd.DataFrame]) -> pd.DataFrame:
        """
        Merge DataFrame chunks back into a single DataFrame.
        """
        chunks_list = list(chunks)
        if not chunks_list:
            return pd.DataFrame()
        merged_df = pd.concat(chunks_list, ignore_index = True, copy = False)
        logger.debug(f'🔗 Merged {len(chunks_list)} chunks into DataFrame with {len(merged_df)} rows')
        return merged_df

    def parallel_apply(self, df: pd.DataFrame, func: Callable[[pd.DataFrame, Any], pd.DataFrame] | Callable[[pd.DataFrame], pd.DataFrame], *args: Any, **kwargs: Any) -> pd.DataFrame:
        """
        Apply a function to DataFrame chunks in parallel with hardware optimization.
        """
        if not isinstance(df, pd.DataFrame):
            msg = 'parallel_apply expects a pandas DataFrame as first argument'
            raise TypeError(msg)
        
        # Get optimal strategy
        strategy = self._get_optimal_strategy(len(df), 'data_processing')
        
        # Check if GPU acceleration is suitable
        if (strategy.get('use_gpu', False) and 
            self.enable_hardware_optimization and 
            self.gpu_manager and 
            self.gpu_manager.is_gpu_available() and
            _is_gpu_suitable(func, (df,))):
            return self._gpu_parallel_apply(df, func, *args, **kwargs)
        
        # Use memory optimization if enabled
        if self.enable_hardware_optimization and self.memory_manager:
            with self.memory_manager.memory_context(MemoryStrategy.ADAPTIVE):
                return self._cpu_parallel_apply(df, func, strategy, *args, **kwargs)
        else:
            return self._cpu_parallel_apply(df, func, strategy, *args, **kwargs)

    def _cpu_parallel_apply(self, df: pd.DataFrame, func: Callable, strategy: Dict[str, Any], *args: Any, **kwargs: Any) -> pd.DataFrame:
        """CPU-based parallel processing with hardware optimization."""
        if len(df) < self.chunk_size * 2:
            logger.debug('📊 Dataset small - processing sequentially')
            return func(df, *args, **kwargs)
        
        # Use strategy-recommended chunk size
        optimal_chunk_size = strategy.get('chunk_size', self.chunk_size)
        chunks = self._split_dataframe(df, chunk_size=optimal_chunk_size)
        partial_func = partial(func, *args, **kwargs)
        
        start_time = time.time()
        
        # Use strategy-recommended thread count
        num_threads = strategy.get('num_threads', self.max_workers)
        
        if self.use_process_pool:
            executor_cls = ProcessPoolExecutor
        else:
            executor_cls = ThreadPoolExecutor
        
        results: list[pd.DataFrame] = []
        with executor_cls(max_workers=num_threads) as executor:
            futures = [executor.submit(partial_func, chunk) for chunk in chunks]
            for future in as_completed(futures):
                results.append(future.result())
        
        processing_time = time.time() - start_time
        merged_result = self._merge_chunks(results)
        
        # Record performance for adaptive learning
        if self.enable_hardware_optimization and self.adaptive_engine:
            self.adaptive_engine.record_performance(
                execution_time=processing_time,
                throughput=len(df) / max(processing_time, 0.001),
                error_rate=0.0
            )
        
        logger.info('⚡ Hardware-optimized parallel processing completed:')
        logger.info(f'   Chunks processed: {len(chunks)}')
        logger.info(f'   Processing time: {processing_time:.2f}s')
        logger.info(f'   Threads used: {num_threads}')
        if processing_time > 0:
            logger.info(f'   Speed: {len(df) / processing_time:.0f} rows/second')
        
        return merged_result

    def _gpu_parallel_apply(self, df: pd.DataFrame, func: Callable, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """GPU-accelerated parallel processing."""
        try:
            logger.info('🚀 Using GPU acceleration for parallel processing')
            
            # Convert DataFrame to GPU-friendly format
            gpu_data = self.gpu_manager.prepare_data_for_gpu(df)
            
            # Execute on GPU
            result = self.gpu_manager.execute_gpu_operation(
                func, 
                GPUOperationType.MATRIX_MULTIPLICATION,
                gpu_data, 
                *args, 
                **kwargs
            )
            
            # Convert back to DataFrame
            return self.gpu_manager.convert_from_gpu(result, df.index, df.columns)
            
        except Exception as e:
            logger.warning(f'GPU processing failed: {e}, falling back to CPU')
            return self._cpu_parallel_apply(df, func, self._get_optimal_strategy(len(df)), *args, **kwargs)

    def parallel_feature_engineering(self, df: pd.DataFrame, feature_funcs: list[Callable[[pd.DataFrame], pd.DataFrame]], *args: Any, **kwargs: Any) -> pd.DataFrame:
        """
        Apply multiple feature engineering functions in parallel with hardware optimization.
        """
        if not feature_funcs:
            return df.copy()
        if len(feature_funcs) == 1:
            return self.parallel_apply(df, feature_funcs[0], *args, **kwargs)
        
        # Get optimal strategy for feature engineering
        strategy = self._get_optimal_strategy(len(df), 'feature_engineering')
        workers_per_func = max(1, strategy.get('num_threads', self.max_workers) // max(1, len(feature_funcs)))
        
        logger.info(f'🔧 Hardware-optimized parallel feature engineering with {len(feature_funcs)} functions | workers per func: {workers_per_func}')
        
        # Use memory optimization for feature engineering
        if self.enable_hardware_optimization and self.memory_manager:
            with self.memory_manager.memory_context(MemoryStrategy.ADAPTIVE):
                results = self._execute_feature_engineering(df, feature_funcs, workers_per_func, *args, **kwargs)
        else:
            results = self._execute_feature_engineering(df, feature_funcs, workers_per_func, *args, **kwargs)
        
        final_result = pd.concat(results, axis=1)
        logger.info('✅ Hardware-optimized parallel feature engineering completed')
        return final_result

    def _execute_feature_engineering(self, df: pd.DataFrame, feature_funcs: list[Callable], workers_per_func: int, *args: Any, **kwargs: Any) -> list[pd.DataFrame]:
        """Execute feature engineering functions with hardware optimization."""
        results: list[pd.DataFrame] = []
        
        for i, func in enumerate(feature_funcs):
            # Create optimized processor for each function
            temp_optimizer = MacM1ParallelOptimizer(
                max_workers=workers_per_func,
                chunk_size=self.chunk_size,
                use_process_pool=self.use_process_pool,
                memory_limit_mb=self.memory_limit_mb,
                enable_hardware_optimization=self.enable_hardware_optimization
            )
            
            # Apply hardware optimization context
            if self.enable_hardware_optimization and self.unified_manager:
                with self.unified_manager.optimization_context(WorkloadType.FEATURE_ENGINEERING, OptimizationLevel.AGGRESSIVE):
                    result = temp_optimizer.parallel_apply(df, func, *args, **kwargs)
            else:
                result = temp_optimizer.parallel_apply(df, func, *args, **kwargs)
            
            results.append(result)
            
            # Log progress
            logger.debug(f'Completed feature function {i+1}/{len(feature_funcs)}')
        
        return results

    def parallel_rolling_operations(self, df: pd.DataFrame, window_sizes: list[int], operation: str='mean') -> pd.DataFrame:
        """
        Perform rolling operations with different window sizes in parallel with hardware optimization.
        """
        def rolling_operation(chunk_df: pd.DataFrame, window_size: int, op: str) -> pd.DataFrame:
            numeric_cols = chunk_df.select_dtypes(include=[np.number]).columns
            result = chunk_df.copy()
            
            # Use VectorBT for large datasets if available
            if (self.enable_hardware_optimization and 
                len(chunk_df) > 1000 and 
                hasattr(self, 'gpu_manager') and 
                self.gpu_manager and 
                self.gpu_manager.is_gpu_available()):
                
                for col in numeric_cols:
                    try:
                        # Try GPU-accelerated rolling operations
                        if op == 'mean':
                            result[f'{col}_rolling_{window_size}'] = self.gpu_manager.rolling_mean(chunk_df[col], window_size)
                        elif op == 'std':
                            result[f'{col}_rolling_{window_size}_std'] = self.gpu_manager.rolling_std(chunk_df[col], window_size)
                        elif op == 'min':
                            result[f'{col}_rolling_{window_size}_min'] = self.gpu_manager.rolling_min(chunk_df[col], window_size)
                        elif op == 'max':
                            result[f'{col}_rolling_{window_size}_max'] = self.gpu_manager.rolling_max(chunk_df[col], window_size)
                    except Exception:
                        # Fallback to pandas
                        if op == 'mean':
                            result[f'{col}_rolling_{window_size}'] = chunk_df[col].rolling(window_size).mean()
                        elif op == 'std':
                            result[f'{col}_rolling_{window_size}_std'] = chunk_df[col].rolling(window_size).std()
                        elif op == 'min':
                            result[f'{col}_rolling_{window_size}_min'] = chunk_df[col].rolling(window_size).min()
                        elif op == 'max':
                            result[f'{col}_rolling_{window_size}_max'] = chunk_df[col].rolling(window_size).max()
            else:
                # Standard pandas operations
                for col in numeric_cols:
                    if op == 'mean':
                        result[f'{col}_rolling_{window_size}'] = chunk_df[col].rolling(window_size).mean()
                    elif op == 'std':
                        result[f'{col}_rolling_{window_size}_std'] = chunk_df[col].rolling(window_size).std()
                    elif op == 'min':
                        result[f'{col}_rolling_{window_size}_min'] = chunk_df[col].rolling(window_size).min()
                    elif op == 'max':
                        result[f'{col}_rolling_{window_size}_max'] = chunk_df[col].rolling(window_size).max()
            
            return result
        
        feature_funcs = [partial(rolling_operation, window_size=w, op=operation) for w in window_sizes]
        return self.parallel_feature_engineering(df, feature_funcs)

    def get_system_info(self) -> dict[str, Any]:
        """
        Get system information for optimization.
        """
        cpu_count = mp.cpu_count()
        memory_gb = psutil.virtual_memory().total / 1024 ** 3
        return {'cpu_count': cpu_count, 'memory_gb': memory_gb, 'is_m1_mac': self.is_m1_mac, 'max_workers': self.max_workers, 'chunk_size': self.chunk_size, 'memory_limit_mb': self.memory_limit_mb}

    def log_system_info(self) -> None:
        """Log system information for debugging."""
        info = self.get_system_info()
        logger.info('💻 System Information:')
        logger.info(f"   CPU cores: {info['cpu_count']}")
        logger.info(f"   Total memory: {info['memory_gb']:.1f} GB")
        logger.info(f"   M1 Mac: {info['is_m1_mac']}")
        logger.info(f"   Max workers: {info['max_workers']}")
        logger.info(f"   Chunk size: {info['chunk_size']}")
        logger.info(f"   Memory limit per worker: {info['memory_limit_mb']} MB")
_parallel_optimizer: MacM1ParallelOptimizer | None = None

def get_parallel_optimizer() -> MacM1ParallelOptimizer:
    """
    Get the global parallel optimizer instance.
    """
    global _parallel_optimizer
    if _parallel_optimizer is None:
        _parallel_optimizer = MacM1ParallelOptimizer()
    return _parallel_optimizer

def parallel_feature_engineering(max_workers: int = 4, enable_hardware_optimization: bool = True) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Decorator for hardware-optimized parallel feature engineering functions that return a DataFrame.
    Skips parallelization for async functions.
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        if asyncio.iscoroutinefunction(func):
            return func

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> None:
            optimizer = get_parallel_optimizer()
            optimizer.max_workers = max(1, max_workers)
            optimizer.enable_hardware_optimization = enable_hardware_optimization
            
            df_arg: pd.DataFrame | None = None
            for arg in args:
                if isinstance(arg, pd.DataFrame):
                    df_arg = arg
                    break
            if df_arg is None:
                for v in kwargs.values():
                    if isinstance(v, pd.DataFrame):
                        df_arg = v
                        break
            if df_arg is None:
                return func(*args, **kwargs)

            def apply_func(chunk: pd.DataFrame) -> pd.DataFrame:
                return func(chunk, *[a for a in args if not isinstance(a, pd.DataFrame)], **kwargs)
            
            return optimizer.parallel_apply(df_arg, apply_func)
        return wrapper
    return decorator

def optimize_for_m1_mac() -> None:
    """
    Apply Mac M1 specific optimizations via environment hints and hardware managers.
    """
    optimizer = get_parallel_optimizer()
    optimizer.log_system_info()
    
    if optimizer.is_m1_mac:
        # Set environment variables for M1 optimization
        os.environ['OMP_NUM_THREADS'] = str(optimizer.max_workers)
        os.environ['MKL_NUM_THREADS'] = str(optimizer.max_workers)
        os.environ['OPENBLAS_NUM_THREADS'] = str(optimizer.max_workers)
        
        # Initialize hardware optimization if available
        if optimizer.enable_hardware_optimization:
            try:
                # Configure hardware for M1
                optimizer._configure_hardware_for_parallel_processing()
                logger.info('🍎 Applied M1 hardware optimizations')
            except Exception as e:
                logger.warning(f'Failed to apply M1 hardware optimizations: {e}')
        
        logger.info('🍎 Applied Mac M1 specific optimizations')
        logger.info(f'   Set OMP_NUM_THREADS={optimizer.max_workers}')
        logger.info(f'   Set MKL_NUM_THREADS={optimizer.max_workers}')
        logger.info(f'   Set OPENBLAS_NUM_THREADS={optimizer.max_workers}')
        logger.info(f'   Hardware optimization enabled: {optimizer.enable_hardware_optimization}')

class ParallelProcessor:
    """
    Hardware-optimized parallel processor wrapper for ML Common utilities compatibility.
    """

    def __init__(self, max_workers=None, enable_hardware_optimization=True):
        """Initialize the hardware-optimized parallel processor.

        Args:
            max_workers: Maximum number of workers (optional, for compatibility)
            enable_hardware_optimization: Enable hardware-aware optimizations
        """
        self.max_workers = max_workers
        self.enable_hardware_optimization = enable_hardware_optimization
        self.optimizer = get_parallel_optimizer()
        self.logger = logger.getChild('ParallelProcessor')
        
        # Initialize hardware managers if enabled
        if self.enable_hardware_optimization:
            self.hardware_manager = get_integrated_hardware_manager()
            self.adaptive_engine = get_adaptive_optimization_engine()
            self.memory_manager = get_advanced_memory_manager()
            self.gpu_manager = get_enhanced_gpu_manager()
        else:
            self.hardware_manager = None
            self.adaptive_engine = None
            self.memory_manager = None
            self.gpu_manager = None

    def process_batch(self, items, func, max_workers=None):
        """
        Process items in parallel using hardware-optimized processing.

        Args:
            items: Items to process
            func: Function to apply to each item
            max_workers: Maximum number of workers (optional)

        Returns:
            List of results
        """
        if not items:
            return []

        try:
            # Get optimal strategy if hardware optimization is enabled
            strategy = None
            if self.enable_hardware_optimization and self.adaptive_engine:
                strategy = self.adaptive_engine.get_optimal_strategy(
                    'batch_processing',
                    {
                        'memory_pressure': _get_memory_pressure(),
                        'data_size': len(items),
                        'function_name': func.__name__
                    }
                )

            # Use memory optimization if enabled
            if self.enable_hardware_optimization and self.memory_manager:
                with self.memory_manager.memory_context(MemoryStrategy.ADAPTIVE):
                    return self._process_with_hardware_optimization(items, func, strategy)
            else:
                return self._process_with_hardware_optimization(items, func, strategy)

        except Exception as e:
            self.logger.error(f"Hardware-optimized processing failed: {e}, falling back to sequential")
            return [func(item) for item in items]

    def _process_with_hardware_optimization(self, items, func, strategy):
        """Process items with hardware optimization."""
        # Use the optimizer's parallel processing capabilities
        if hasattr(self.optimizer, 'parallel_apply'):
            # For DataFrame operations
            if hasattr(items, 'iterrows') or hasattr(items, 'itertuples'):
                return self.optimizer.parallel_apply(items, func)
            else:
                # For general iterables - convert to DataFrame for parallel processing
                df = pd.DataFrame({'item': list(items)})
                
                def process_item(df_chunk):
                    return df_chunk['item'].apply(func)
                
                result_df = self.optimizer.parallel_apply(df, process_item)
                return result_df['item'].tolist()
        else:
            # Fallback to sequential processing
            self.logger.warning("Parallel processing not available, falling back to sequential")
            return [func(item) for item in items]

    def _should_use_vectorbt(self, data) -> bool:
        """Determine if VectorBT should be used based on data size and configuration."""
        return (hasattr(self, 'use_vectorbt') and getattr(self, 'use_vectorbt', True) and
                len(data) >= getattr(self, 'vectorbt_threshold', 1000) and
                VECTORBT_AVAILABLE)

    def _vectorbt_rolling_operation(self, data: pd.Series, operation: str,
                                  window: int, **kwargs) -> pd.Series:
        """Perform VectorBT rolling operation with fallback to pandas."""
        if not self._should_use_vectorbt(data):
            return self._pandas_rolling_operation(data, operation, window, **kwargs)

        try:
            if operation == 'mean':
                return rolling_mean(data, window=window, **kwargs)
            elif operation == 'std':
                return rolling_std(data, window=window, **kwargs)
            elif operation == 'var':
                return rolling_var(data, window=window, **kwargs)
            elif operation == 'min':
                return rolling_min(data, window=window, **kwargs)
            elif operation == 'max':
                return rolling_max(data, window=window, **kwargs)
            elif operation == 'sum':
                return rolling_sum(data, window=window, **kwargs)
            else:
                raise ValueError(f"Unsupported operation: {operation}")
        except Exception as e:
            logger.warning(f"VectorBT operation failed: {e}, using pandas fallback")
            return self._pandas_rolling_operation(data, operation, window, **kwargs)

    def _pandas_rolling_operation(self, data: pd.Series, operation: str,
                                 window: int, **kwargs) -> pd.Series:
        """Fallback rolling operation using pandas."""
        if operation == 'mean':
            return data.rolling(window=window).mean()
        elif operation == 'std':
            return data.rolling(window=window).std()
        elif operation == 'var':
            return data.rolling(window=window).var()
        elif operation == 'min':
            return data.rolling(window=window).min()
        elif operation == 'max':
            return data.rolling(window=window).max()
        elif operation == 'sum':
            return data.rolling(window=window).sum()
        else:
            raise ValueError(f"Unsupported operation: {operation}")

    def _vectorbt_apply_operation(self, data: pd.Series, func,
                                 window: int, **kwargs) -> pd.Series:
        """Perform VectorBT rolling apply operation with fallback to pandas."""
        if not self._should_use_vectorbt(data):
            return data.rolling(window=window).apply(func, **kwargs)

        try:
            return rolling_apply(data, func, window=window, **kwargs)
        except Exception as e:
            logger.warning(f"VectorBT rolling apply failed: {e}, using pandas fallback")
            return data.rolling(window=window).apply(func, **kwargs)
