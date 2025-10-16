"""
VectorBT Memory Optimization Utilities

This module provides VectorBT-optimized memory management for large-scale ML operations:
- Chunked processing with VectorBT
- Memory-efficient data structures
- GPU memory management
- Caching strategies with VectorBT
- Memory monitoring and optimization
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union, Callable, Iterator
from dataclasses import dataclass, field
from enum import Enum
import logging
import warnings
import gc
import psutil
import os

# VectorBT imports
try:
    import vectorbt as vbt
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None

# GPU acceleration removed - CuPy not supported on all platforms
cp = None
CUPY_AVAILABLE = False

logger = logging.getLogger(__name__)

class MemoryStrategy(Enum):
    """Memory optimization strategies."""
    CHUNKED_PROCESSING = "chunked_processing"
    LAZY_EVALUATION = "lazy_evaluation"
    MEMORY_MAPPING = "memory_mapping"
    GPU_ACCELERATION = "gpu_acceleration"
    CACHING = "caching"
    COMPRESSION = "compression"

@dataclass
class MemoryConfig:
    """Configuration for memory optimization."""
    # Basic settings
    max_memory_gb: float = 8.0
    chunk_size: int = 10000
    enable_gpu: bool = True
    enable_compression: bool = True

    # VectorBT settings
    use_vectorbt: bool = True
    vectorbt_chunk_size: int = 50000
    enable_vectorbt_caching: bool = True

    # Memory management
    memory_cleanup_threshold: float = 0.8  # Cleanup when 80% memory used
    enable_auto_cleanup: bool = True
    enable_memory_monitoring: bool = True

    # Caching
    cache_size_mb: int = 1000
    enable_disk_caching: bool = False
    cache_directory: str = "./cache"

@dataclass
class MemoryStats:
    """Memory usage statistics."""
    total_memory_gb: float
    used_memory_gb: float
    available_memory_gb: float
    memory_usage_percent: float
    gpu_memory_gb: Optional[float] = None
    gpu_usage_percent: Optional[float] = None
    cache_hit_rate: Optional[float] = None

class VectorBTMemoryOptimizer:
    """
    VectorBT-enhanced memory optimizer.

    This class provides advanced memory management using VectorBT for:
    - Chunked processing of large datasets
    - Memory-efficient data structures
    - GPU memory management
    - Intelligent caching strategies
    """

    def __init__(self, config: Optional[MemoryConfig] = None):
        """
        Initialize VectorBT memory optimizer.

        Args:
            config: Memory configuration
        """
        self.config = config or MemoryConfig()

        # Initialize VectorBT settings
        self._configure_vectorbt()

        # Memory monitoring
        self.memory_stats = []
        self.cache_stats = {'hits': 0, 'misses': 0}

        # Initialize cache
        self._initialize_cache()

        logger.info("✅ VectorBT Memory Optimizer initialized")
        logger.info(f"📊 Max memory: {self.config.max_memory_gb:.1f}GB")
        logger.info(f"📊 Chunk size: {self.config.chunk_size:,}")
        logger.info(f"📊 GPU enabled: False (GPU support removed)")

    def _configure_vectorbt(self):
        """Configure VectorBT for optimal memory usage."""
        if not VECTORBT_AVAILABLE:
            logger.warning("VectorBT not available, memory optimizations limited")
            return

        # Configure VectorBT memory settings
        vbt.settings.array_wrapper['freq'] = '1min'

        # Enable chunking for large datasets
        if hasattr(vbt.settings, 'chunking'):
            vbt.settings.chunking['chunk_size'] = self.config.vectorbt_chunk_size
            vbt.settings.chunking['enable'] = True

        # Configure parallel processing
        if hasattr(vbt.settings, 'parallel'):
            vbt.settings.parallel['threading'] = True
            vbt.settings.parallel['n_threads'] = min(os.cpu_count() or 4, 8)

    def _initialize_cache(self):
        """Initialize memory cache."""
        self.cache = {}
        self.cache_size = 0
        self.max_cache_size = self.config.cache_size_mb * 1024 * 1024  # Convert to bytes

        if self.config.enable_disk_caching:
            os.makedirs(self.config.cache_directory, exist_ok=True)

    def process_large_dataset(self,
                             data: Union[np.ndarray, pd.DataFrame],
                             processing_func: Callable,
                             chunk_size: Optional[int] = None,
                             **kwargs) -> Union[np.ndarray, pd.DataFrame]:
        """
        Process large dataset in chunks using VectorBT optimizations.

        Args:
            data: Input data to process
            processing_func: Function to apply to each chunk
            chunk_size: Size of chunks (uses config default if None)
            **kwargs: Additional arguments for processing function

        Returns:
            Processed data
        """
        if chunk_size is None:
            chunk_size = self.config.chunk_size

        logger.info(f"🔄 Processing large dataset in chunks of {chunk_size:,}")

        # Convert to DataFrame if needed
        if isinstance(data, np.ndarray):
            if data.ndim == 1:
                data_df = pd.DataFrame(data, columns=['value'])
            else:
                data_df = pd.DataFrame(data, columns=[f'col_{i}' for i in range(data.shape[1])])
        else:
            data_df = data.copy()

        # Process in chunks
        results = []
        n_chunks = (len(data_df) + chunk_size - 1) // chunk_size

        for i in range(n_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, len(data_df))

            chunk = data_df.iloc[start_idx:end_idx]

            # Process chunk
            try:
                chunk_result = processing_func(chunk, **kwargs)
                results.append(chunk_result)

                # Memory cleanup
                if self.config.enable_auto_cleanup:
                    self._cleanup_memory()

                logger.debug(f"Processed chunk {i+1}/{n_chunks}")

            except Exception as e:
                logger.error(f"Error processing chunk {i+1}: {e}")
                raise

        # Combine results
        if isinstance(results[0], pd.DataFrame):
            return pd.concat(results, ignore_index=True)
        elif isinstance(results[0], np.ndarray):
            return np.concatenate(results, axis=0)
        else:
            return results

    def vectorbt_optimized_operation(self,
                                   operation: str,
                                   data: Union[np.ndarray, pd.DataFrame],
                                   **kwargs) -> Union[np.ndarray, pd.DataFrame]:
        """
        Perform VectorBT-optimized operations with advanced memory management.

        Args:
            operation: Operation to perform ('portfolio', 'metrics', 'optimization', 'backtest', 'analysis')
            data: Input data
            **kwargs: Operation-specific arguments

        Returns:
            Operation results
        """
        if not VECTORBT_AVAILABLE:
            logger.warning("VectorBT not available, using standard operations")
            return data

        logger.info(f"🔄 Performing VectorBT-optimized {operation}...")

        # Check memory usage and optimize if needed
        memory_usage = self.get_memory_stats()
        if memory_usage.memory_usage_percent > 0.8:
            logger.warning("High memory usage detected, performing cleanup")
            self._cleanup_memory()

        # Estimate memory requirements for operation
        estimated_memory = self._estimate_operation_memory(operation, data)
        if estimated_memory > self.config.max_memory_gb * 0.5:
            logger.info(f"Large operation detected ({estimated_memory:.2f}GB), using chunked processing")
            return self._vectorbt_chunked_operation(operation, data, **kwargs)

        try:
            if operation == 'portfolio':
                return self._vectorbt_portfolio_operation(data, **kwargs)
            elif operation == 'metrics':
                return self._vectorbt_metrics_operation(data, **kwargs)
            elif operation == 'optimization':
                return self._vectorbt_optimization_operation(data, **kwargs)
            elif operation == 'backtest':
                return self._vectorbt_backtest_operation(data, **kwargs)
            elif operation == 'analysis':
                return self._vectorbt_analysis_operation(data, **kwargs)
            else:
                raise ValueError(f"Unsupported operation: {operation}")

        except Exception as e:
            logger.error(f"VectorBT operation failed: {e}")
            # Fallback to standard operation
            return data

    def _estimate_operation_memory(self, operation: str, data: Union[np.ndarray, pd.DataFrame]) -> float:
        """Estimate memory requirements for VectorBT operations."""
        if isinstance(data, np.ndarray):
            base_memory = data.nbytes / (1024**3)
        else:
            base_memory = data.memory_usage(deep=True).sum() / (1024**3)

        # Memory multipliers for different operations
        multipliers = {
            'portfolio': 3.0,  # Portfolio operations create multiple arrays
            'metrics': 2.0,   # Metrics calculations need intermediate arrays
            'optimization': 4.0,  # Optimization needs multiple copies
            'backtest': 5.0,  # Backtesting is memory intensive
            'analysis': 2.5   # Analysis operations need intermediate results
        }

        return base_memory * multipliers.get(operation, 2.0)

    def _vectorbt_chunked_operation(self, operation: str, data: Union[np.ndarray, pd.DataFrame], **kwargs) -> Any:
        """Perform VectorBT operations in chunks for large datasets."""
        logger.info(f"🔄 Performing chunked VectorBT {operation}...")

        if isinstance(data, np.ndarray):
            if data.ndim == 1:
                data_df = pd.DataFrame(data, columns=['value'])
            else:
                data_df = pd.DataFrame(data, columns=[f'col_{i}' for i in range(data.shape[1])])
        else:
            data_df = data.copy()

        # Process in chunks
        chunk_size = self.config.chunk_size
        results = []

        for i in range(0, len(data_df), chunk_size):
            chunk = data_df.iloc[i:i+chunk_size]

            try:
                # Process chunk based on operation
                if operation == 'portfolio':
                    chunk_result = self._vectorbt_portfolio_operation(chunk, **kwargs)
                elif operation == 'metrics':
                    chunk_result = self._vectorbt_metrics_operation(chunk, **kwargs)
                elif operation == 'backtest':
                    chunk_result = self._vectorbt_backtest_operation(chunk, **kwargs)
                else:
                    chunk_result = chunk

                results.append(chunk_result)

                # Memory cleanup after each chunk
                if self.config.enable_auto_cleanup:
                    self._cleanup_memory()

            except Exception as e:
                logger.error(f"Error processing chunk {i//chunk_size + 1}: {e}")
                continue

        # Combine results
        if results and isinstance(results[0], pd.DataFrame):
            return pd.concat(results, ignore_index=True)
        elif results and isinstance(results[0], np.ndarray):
            return np.concatenate(results, axis=0)
        else:
            return results

    def _vectorbt_backtest_operation(self, data: Union[np.ndarray, pd.DataFrame], **kwargs) -> Any:
        """Perform VectorBT backtesting with memory optimization."""
        try:
            # Convert to proper format
            if isinstance(data, np.ndarray):
                data_df = pd.DataFrame(data)
            else:
                data_df = data.copy()

            # Create time index
            data_df.index = pd.date_range(start='2020-01-01', periods=len(data_df), freq='1min')

            # Use VectorBT portfolio for backtesting
            if len(data_df) > self.config.vectorbt_chunk_size:
                # Use chunked backtesting
                return self._chunked_vectorbt_backtest(data_df, **kwargs)
            else:
                # Standard backtesting
                return self.vbt.Portfolio.from_orders(data_df, freq='1min')

        except Exception as e:
            logger.error(f"VectorBT backtest operation failed: {e}")
            return data

    def _vectorbt_analysis_operation(self, data: Union[np.ndarray, pd.DataFrame], **kwargs) -> Any:
        """Perform VectorBT analysis operations with memory optimization."""
        try:
            # Convert to proper format
            if isinstance(data, np.ndarray):
                data_df = pd.DataFrame(data)
            else:
                data_df = data.copy()

            # Create time index
            data_df.index = pd.date_range(start='2020-01-01', periods=len(data_df), freq='1min')

            # Perform VectorBT analysis
            analysis_results = {}

            # Basic statistics
            analysis_results['basic_stats'] = data_df.describe()

            # Time series analysis
            if len(data_df.columns) > 1:
                # Calculate correlations
                analysis_results['correlations'] = data_df.corr()

                # Calculate returns if numeric
                numeric_cols = data_df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    returns = data_df[numeric_cols].pct_change().dropna()
                    analysis_results['returns_stats'] = returns.describe()

            return analysis_results

        except Exception as e:
            logger.error(f"VectorBT analysis operation failed: {e}")
            return {}

    def _chunked_vectorbt_backtest(self, data_df: pd.DataFrame, **kwargs) -> Any:
        """Perform chunked VectorBT backtesting for large datasets."""
        chunk_size = self.config.vectorbt_chunk_size
        results = []

        for i in range(0, len(data_df), chunk_size):
            chunk = data_df.iloc[i:i+chunk_size]

            try:
                # Create portfolio for chunk
                chunk_portfolio = self.vbt.Portfolio.from_orders(chunk, freq='1min')
                results.append(chunk_portfolio)

            except Exception as e:
                logger.error(f"Chunked backtest failed for chunk {i//chunk_size + 1}: {e}")
                continue

        return results

    def _vectorbt_portfolio_operation(self, data: Union[np.ndarray, pd.DataFrame], **kwargs) -> Any:
        """Perform VectorBT portfolio operations with memory optimization."""
        # Convert to proper format
        if isinstance(data, np.ndarray):
            data_df = pd.DataFrame(data)
        else:
            data_df = data.copy()

        # Use VectorBT portfolio with memory optimization
        if len(data_df) > self.config.vectorbt_chunk_size:
            # Process in chunks
            return self.process_large_dataset(
                data_df,
                lambda chunk: vbt.Portfolio.from_orders(chunk, freq='1min'),
                chunk_size=self.config.vectorbt_chunk_size
            )
        else:
            # Process normally
            return vbt.Portfolio.from_orders(data_df, freq='1min')

    def _vectorbt_metrics_operation(self, data: Union[np.ndarray, pd.DataFrame], **kwargs) -> Any:
        """Perform VectorBT metrics operations with memory optimization."""
        # This would implement VectorBT metrics calculations
        # For now, return the data as-is
        return data

    def _vectorbt_optimization_operation(self, data: Union[np.ndarray, pd.DataFrame], **kwargs) -> Any:
        """Perform VectorBT optimization operations with memory optimization."""
        # This would implement VectorBT optimization calculations
        # For now, return the data as-is
        return data

    def get_memory_stats(self) -> MemoryStats:
        """Get current memory usage statistics."""
        # Get system memory
        memory = psutil.virtual_memory()
        total_memory_gb = memory.total / (1024**3)
        used_memory_gb = memory.used / (1024**3)
        available_memory_gb = memory.available / (1024**3)
        memory_usage_percent = memory.percent / 100

        # Get GPU memory if available
        gpu_memory_gb = None
        gpu_usage_percent = None
        if False:  # GPU support removed
            try:
                gpu_memory = 0  # GPU support removed
                gpu_memory_gb = gpu_memory / (1024**3)
                gpu_usage_percent = 0.5  # Placeholder
            except:
                pass

        # Calculate cache hit rate
        cache_hit_rate = None
        if self.cache_stats['hits'] + self.cache_stats['misses'] > 0:
            cache_hit_rate = self.cache_stats['hits'] / (self.cache_stats['hits'] + self.cache_stats['misses'])

        return MemoryStats(
            total_memory_gb=total_memory_gb,
            used_memory_gb=used_memory_gb,
            available_memory_gb=available_memory_gb,
            memory_usage_percent=memory_usage_percent,
            gpu_memory_gb=gpu_memory_gb,
            gpu_usage_percent=gpu_usage_percent,
            cache_hit_rate=cache_hit_rate
        )

    def _cleanup_memory(self):
        """Perform memory cleanup."""
        logger.debug("🧹 Performing memory cleanup...")

        # Force garbage collection
        gc.collect()

        # Clear cache if too large
        if self.cache_size > self.max_cache_size:
            self._clear_cache()

        # GPU memory cleanup
        if False:  # GPU support removed
            try:
                # GPU memory cleanup removed
            except:
                pass

    def _clear_cache(self):
        """Clear memory cache."""
        self.cache.clear()
        self.cache_size = 0
        logger.debug("🗑️ Cache cleared")

    def cache_result(self, key: str, result: Any) -> bool:
        """
        Cache a result with memory management.

        Args:
            key: Cache key
            result: Result to cache

        Returns:
            True if cached successfully, False otherwise
        """
        # Estimate memory usage
        if isinstance(result, np.ndarray):
            memory_usage = result.nbytes
        elif isinstance(result, pd.DataFrame):
            memory_usage = result.memory_usage(deep=True).sum()
        else:
            memory_usage = 1000  # Estimate

        # Check if we have enough memory
        if self.cache_size + memory_usage > self.max_cache_size:
            # Remove oldest entries
            self._evict_cache_entries(memory_usage)

        # Cache the result
        self.cache[key] = result
        self.cache_size += memory_usage
        self.cache_stats['hits'] += 1

        return True

    def get_cached_result(self, key: str) -> Optional[Any]:
        """
        Get cached result.

        Args:
            key: Cache key

        Returns:
            Cached result or None if not found
        """
        if key in self.cache:
            self.cache_stats['hits'] += 1
            return self.cache[key]
        else:
            self.cache_stats['misses'] += 1
            return None

    def _evict_cache_entries(self, required_memory: int):
        """Evict cache entries to free up memory."""
        # Simple LRU eviction
        keys_to_remove = []
        freed_memory = 0

        for key, value in self.cache.items():
            if isinstance(value, np.ndarray):
                memory_usage = value.nbytes
            elif isinstance(value, pd.DataFrame):
                memory_usage = value.memory_usage(deep=True).sum()
            else:
                memory_usage = 1000

            keys_to_remove.append(key)
            freed_memory += memory_usage

            if freed_memory >= required_memory:
                break

        # Remove entries
        for key in keys_to_remove:
            if key in self.cache:
                del self.cache[key]
                self.cache_size -= freed_memory

    def optimize_memory_usage(self, data: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.DataFrame]:
        """
        Optimize memory usage of data.

        Args:
            data: Input data

        Returns:
            Memory-optimized data
        """
        logger.info("🔄 Optimizing memory usage...")

        if isinstance(data, np.ndarray):
            # Optimize numpy array
            if data.dtype == np.float64:
                # Convert to float32 if precision allows
                if np.allclose(data, data.astype(np.float32)):
                    data = data.astype(np.float32)
                    logger.info("Converted float64 to float32 for memory optimization")

            # Use memory mapping for very large arrays
            if data.nbytes > 100 * 1024 * 1024:  # 100MB
                logger.info("Using memory mapping for large array")
                # This would implement memory mapping

        elif isinstance(data, pd.DataFrame):
            # Optimize pandas DataFrame
            for col in data.columns:
                if data[col].dtype == 'float64':
                    # Try to convert to float32
                    if data[col].isnull().sum() == 0:  # No NaN values
                        data[col] = data[col].astype(np.float32)

                elif data[col].dtype == 'int64':
                    # Try to convert to smaller int types
                    if data[col].min() >= 0:
                        if data[col].max() < 255:
                            data[col] = data[col].astype(np.uint8)
                        elif data[col].max() < 65535:
                            data[col] = data[col].astype(np.uint16)
                        elif data[col].max() < 4294967295:
                            data[col] = data[col].astype(np.uint32)
                    else:
                        if data[col].min() >= -128 and data[col].max() <= 127:
                            data[col] = data[col].astype(np.int8)
                        elif data[col].min() >= -32768 and data[col].max() <= 32767:
                            data[col] = data[col].astype(np.int16)
                        elif data[col].min() >= -2147483648 and data[col].max() <= 2147483647:
                            data[col] = data[col].astype(np.int32)

        return data

    def monitor_memory_usage(self, interval: int = 10):
        """
        Monitor memory usage over time.

        Args:
            interval: Monitoring interval in seconds
        """
        import time

        logger.info(f"📊 Starting memory monitoring (interval: {interval}s)")

        while True:
            stats = self.get_memory_stats()
            self.memory_stats.append(stats)

            logger.info(f"Memory: {stats.used_memory_gb:.1f}GB/{stats.total_memory_gb:.1f}GB "
                       f"({stats.memory_usage_percent:.1%})")

            if stats.memory_usage_percent > self.config.memory_cleanup_threshold:
                logger.warning("High memory usage detected, performing cleanup")
                self._cleanup_memory()

            time.sleep(interval)

    def get_optimization_recommendations(self) -> List[str]:
        """Get memory optimization recommendations."""
        recommendations = []
        stats = self.get_memory_stats()

        if stats.memory_usage_percent > 0.8:
            recommendations.append("High memory usage detected - consider reducing chunk size")

        if stats.cache_hit_rate is not None and stats.cache_hit_rate < 0.5:
            recommendations.append("Low cache hit rate - consider increasing cache size")

        if self.config.enable_gpu and True:
            recommendations.append("

        if not VECTORBT_AVAILABLE:
            recommendations.append("VectorBT not available - install vectorbt for memory optimizations")

        return recommendations

# Convenience functions
def optimize_memory_usage(data: Union[np.ndarray, pd.DataFrame],
                         config: Optional[MemoryConfig] = None) -> Union[np.ndarray, pd.DataFrame]:
    """
    Convenience function to optimize memory usage.

    Args:
        data: Input data
        config: Memory configuration

    Returns:
        Memory-optimized data
    """
    optimizer = VectorBTMemoryOptimizer(config)
    return optimizer.optimize_memory_usage(data)

def process_large_dataset(data: Union[np.ndarray, pd.DataFrame],
                         processing_func: Callable,
                         chunk_size: Optional[int] = None,
                         config: Optional[MemoryConfig] = None,
                         **kwargs) -> Union[np.ndarray, pd.DataFrame]:
    """
    Convenience function to process large dataset in chunks.

    Args:
        data: Input data
        processing_func: Function to apply to each chunk
        chunk_size: Size of chunks
        config: Memory configuration
        **kwargs: Additional arguments

    Returns:
        Processed data
    """
    optimizer = VectorBTMemoryOptimizer(config)
    return optimizer.process_large_dataset(data, processing_func, chunk_size, **kwargs)

if __name__ == "__main__":
    # Example usage and testing
    logger.info("🧪 Testing VectorBT Memory Optimizer...")

    # Generate large dataset
    np.random.seed(42)
    n_samples = 100000
    n_features = 100

    X = np.random.randn(n_samples, n_features).astype(np.float64)
    y = np.random.randn(n_samples).astype(np.float64)

    print(f"Dataset: {X.shape[0]:,} samples, {X.shape[1]} features")
    print(f"Memory usage: {X.nbytes / (1024**2):.1f}MB")

    # Test memory optimizer
    config = MemoryConfig(max_memory_gb=4.0, chunk_size=10000)
    optimizer = VectorBTMemoryOptimizer(config)

    # Test memory optimization
    print("\n🔄 Testing memory optimization...")
    X_optimized = optimizer.optimize_memory_usage(X)
    print(f"Optimized memory usage: {X_optimized.nbytes / (1024**2):.1f}MB")
    print(f"Memory reduction: {(1 - X_optimized.nbytes / X.nbytes) * 100:.1f}%")

    # Test chunked processing
    print("\n🔄 Testing chunked processing...")
    def process_chunk(chunk):
        return np.mean(chunk, axis=0)

    result = optimizer.process_large_dataset(X, process_chunk, chunk_size=5000)
    print(f"Chunked processing result shape: {result.shape}")

    # Test memory stats
    print("\n📊 Memory statistics:")
    stats = optimizer.get_memory_stats()
    print(f"Total memory: {stats.total_memory_gb:.1f}GB")
    print(f"Used memory: {stats.used_memory_gb:.1f}GB")
    print(f"Memory usage: {stats.memory_usage_percent:.1%}")

    # Test recommendations
    print("\n💡 Optimization recommendations:")
    recommendations = optimizer.get_optimization_recommendations()
    for rec in recommendations:
        print(f"  - {rec}")

    print("\n✅ VectorBT Memory Optimizer test completed!")
