"""
Memory Optimization Utilities for Large Datasets

This module provides comprehensive memory optimization utilities for processing
large datasets efficiently using VectorBT and other optimization techniques.

Key Features:
- Chunked processing for large datasets
- Memory-efficient data type optimization
- VectorBT memory management
- GPU memory optimization
- Memory monitoring and cleanup
"""

import numpy as np
import pandas as pd
import logging
import gc
import psutil
import warnings
from typing import Any, Dict, List, Optional, Union, Tuple, Iterator, Callable
from dataclasses import dataclass
import time

# VectorBT imports for optimization
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

@dataclass
class MemoryConfig:
    """Configuration for memory optimization."""
    # Memory limits
    max_memory_gb: float = 8.0
    chunk_size: int = 10000
    max_chunk_size: int = 50000

    # Optimization settings
    enable_dtype_optimization: bool = True
    enable_chunked_processing: bool = True
    enable_memory_monitoring: bool = True
    enable_garbage_collection: bool = True

    # VectorBT settings
    vectorbt_memory_limit_gb: float = 6.0
    vectorbt_chunk_size: int = 20000

    # GPU settings
    gpu_memory_limit_gb: float = 4.0
    enable_gpu_memory_management: bool = True

    # Cleanup settings
    cleanup_frequency: int = 10  # Cleanup every N operations
    aggressive_cleanup_threshold: float = 0.8  # Cleanup when memory usage > 80%

class MemoryOptimizer:
    """
    Memory optimization utility for large dataset processing.

    Provides intelligent memory management, chunked processing, and optimization
    techniques to handle large datasets efficiently.
    """

    def __init__(self, config: Optional[MemoryConfig] = None):
        """
        Initialize memory optimizer.

        Args:
            config: Memory optimization configuration
        """
        self.config = config or MemoryConfig()
        self.memory_stats = {
            'peak_memory_usage': 0.0,
            'current_memory_usage': 0.0,
            'chunks_processed': 0,
            'memory_cleanups': 0,
            'dtype_optimizations': 0,
            'total_operations': 0
        }

        # Configure VectorBT memory settings
        if VECTORBT_AVAILABLE:
            vbt.settings.memory['limit'] = self.config.vectorbt_memory_limit_gb * 1024**3
            vbt.settings.array_wrapper['freq'] = '1min'

        logger.info(f"MemoryOptimizer initialized: Max memory={self.config.max_memory_gb}GB, Chunk size={self.config.chunk_size}")

    def optimize_dataframe_dtypes(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Optimize DataFrame dtypes for memory efficiency.

        Args:
            data: Input DataFrame

        Returns:
            Optimized DataFrame with memory-efficient dtypes
        """
        if not self.config.enable_dtype_optimization:
            return data

        try:
            optimized_data = data.copy()

            # Optimize numeric columns
            for col in optimized_data.select_dtypes(include=['float64']):
                # Try to downcast to float32
                if optimized_data[col].min() >= np.finfo(np.float32).min and \
                   optimized_data[col].max() <= np.finfo(np.float32).max:
                    optimized_data[col] = pd.to_numeric(optimized_data[col], downcast='float')

            for col in optimized_data.select_dtypes(include=['int64']):
                # Try to downcast to smaller integer types
                if optimized_data[col].min() >= np.iinfo(np.int32).min and \
                   optimized_data[col].max() <= np.iinfo(np.int32).max:
                    optimized_data[col] = pd.to_numeric(optimized_data[col], downcast='integer')

            # Optimize object columns
            for col in optimized_data.select_dtypes(include=['object']):
                if optimized_data[col].dtype == 'object':
                    # Try to convert to category if it has few unique values
                    unique_ratio = optimized_data[col].nunique() / len(optimized_data)
                    if unique_ratio < 0.5:  # Less than 50% unique values
                        optimized_data[col] = optimized_data[col].astype('category')

            self.memory_stats['dtype_optimizations'] += 1

            # Calculate memory savings
            original_memory = data.memory_usage(deep=True).sum()
            optimized_memory = optimized_data.memory_usage(deep=True).sum()
            memory_savings = (original_memory - optimized_memory) / original_memory * 100

            if memory_savings > 5:  # Only log if significant savings
                logger.info(f"DataFrame dtype optimization: {memory_savings:.1f}% memory reduction")

            return optimized_data

        except Exception as e:
            logger.warning(f"DataFrame dtype optimization failed: {e}")
            return data

    def should_use_chunked_processing(self, data_size: int) -> bool:
        """
        Determine if chunked processing should be used.

        Args:
            data_size: Size of the dataset

        Returns:
            True if chunked processing should be used
        """
        if not self.config.enable_chunked_processing:
            return False

        # Use chunked processing for large datasets
        return data_size > self.config.chunk_size

    def get_optimal_chunk_size(self, data_size: int, available_memory_gb: float) -> int:
        """
        Calculate optimal chunk size based on available memory.

        Args:
            data_size: Size of the dataset
            available_memory_gb: Available memory in GB

        Returns:
            Optimal chunk size
        """
        # Calculate chunk size based on available memory
        memory_based_chunk = int(available_memory_gb * 1024**3 / (data_size * 8))  # Rough estimate

        # Use the smaller of memory-based or configured chunk size
        optimal_chunk = min(
            memory_based_chunk,
            self.config.chunk_size,
            self.config.max_chunk_size
        )

        # Ensure minimum chunk size
        return max(optimal_chunk, 1000)

    def process_in_chunks(
        self,
        data: pd.DataFrame,
        processor_func: Callable[[pd.DataFrame], pd.DataFrame],
        **kwargs
    ) -> pd.DataFrame:
        """
        Process large dataset in memory-efficient chunks.

        Args:
            data: Input DataFrame
            processor_func: Function to process each chunk
            **kwargs: Additional arguments for processor function

        Returns:
            Processed DataFrame
        """
        if not self.should_use_chunked_processing(len(data)):
            return processor_func(data, **kwargs)

        try:
            # Get optimal chunk size
            available_memory = psutil.virtual_memory().available / 1024**3
            chunk_size = self.get_optimal_chunk_size(len(data), available_memory)

            logger.info(f"Processing {len(data)} rows in chunks of {chunk_size}")

            results = []
            total_chunks = (len(data) + chunk_size - 1) // chunk_size

            for i in range(0, len(data), chunk_size):
                chunk = data.iloc[i:i + chunk_size]

                # Process chunk
                chunk_result = processor_func(chunk, **kwargs)
                results.append(chunk_result)

                # Memory cleanup
                if i % (chunk_size * self.config.cleanup_frequency) == 0:
                    self._cleanup_memory()

                # Progress logging
                chunk_num = i // chunk_size + 1
                if chunk_num % 10 == 0:
                    logger.info(f"Processed chunk {chunk_num}/{total_chunks}")

                self.memory_stats['chunks_processed'] += 1

            # Combine results
            if results:
                combined_result = pd.concat(results, ignore_index=False)
                logger.info(f"Chunked processing completed: {len(combined_result)} rows processed")
                return combined_result
            else:
                logger.warning("No results from chunked processing")
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"Chunked processing failed: {e}")
            # Fallback to single processing
            return processor_func(data, **kwargs)

    def process_with_memory_monitoring(
        self,
        data: pd.DataFrame,
        processor_func: Callable[[pd.DataFrame], pd.DataFrame],
        **kwargs
    ) -> pd.DataFrame:
        """
        Process data with memory monitoring and optimization.

        Args:
            data: Input DataFrame
            processor_func: Function to process the data
            **kwargs: Additional arguments for processor function

        Returns:
            Processed DataFrame
        """
        if not self.config.enable_memory_monitoring:
            return processor_func(data, **kwargs)

        try:
            # Monitor initial memory
            initial_memory = self._get_memory_usage()
            logger.info(f"Initial memory usage: {initial_memory:.2f}GB")

            # Optimize data types
            optimized_data = self.optimize_dataframe_dtypes(data)

            # Process data
            start_time = time.time()
            result = processor_func(optimized_data, **kwargs)
            processing_time = time.time() - start_time

            # Monitor final memory
            final_memory = self._get_memory_usage()
            peak_memory = max(initial_memory, final_memory)

            # Update stats
            self.memory_stats['current_memory_usage'] = final_memory
            self.memory_stats['peak_memory_usage'] = max(self.memory_stats['peak_memory_usage'], peak_memory)
            self.memory_stats['total_operations'] += 1

            # Log memory usage
            memory_increase = final_memory - initial_memory
            logger.info(f"Memory usage: {initial_memory:.2f}GB -> {final_memory:.2f}GB "
                       f"(+{memory_increase:.2f}GB, Peak: {peak_memory:.2f}GB)")
            logger.info(f"Processing time: {processing_time:.2f}s")

            # Cleanup if memory usage is high
            if final_memory > self.config.aggressive_cleanup_threshold * psutil.virtual_memory().total / 1024**3:
                self._aggressive_cleanup()

            return result

        except Exception as e:
            logger.error(f"Memory-monitored processing failed: {e}")
            return processor_func(data, **kwargs)

    def _get_memory_usage(self) -> float:
        """Get current memory usage in GB."""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            return memory_info.rss / 1024**3
        except Exception:
            return 0.0

    def _cleanup_memory(self):
        """Perform memory cleanup."""
        if self.config.enable_garbage_collection:
            gc.collect()
            self.memory_stats['memory_cleanups'] += 1

    def _aggressive_cleanup(self):
        """Perform aggressive memory cleanup."""
        logger.info("Performing aggressive memory cleanup...")

        # Force garbage collection
        for _ in range(3):
            gc.collect()

        # Clear VectorBT cache if available
        if VECTORBT_AVAILABLE:
            try:
                vbt.settings.caching['enabled'] = False
                vbt.settings.caching['enabled'] = True
            except Exception as e:
                logger.warning(f"VectorBT cache cleanup failed: {e}")

        # Clear GPU memory if available (GPU support removed)
        if False:  # GPU support removed
            try:
                # GPU memory cleanup removed
            except Exception as e:
                logger.warning(f"GPU memory cleanup failed: {e}")

        self.memory_stats['memory_cleanups'] += 1
        logger.info("Aggressive memory cleanup completed")

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        current_memory = self._get_memory_usage()
        self.memory_stats['current_memory_usage'] = current_memory

        return {
            **self.memory_stats,
            'available_memory_gb': psutil.virtual_memory().available / 1024**3,
            'total_memory_gb': psutil.virtual_memory().total / 1024**3,
            'memory_usage_percent': psutil.virtual_memory().percent
        }

    def reset_stats(self):
        """Reset memory statistics."""
        self.memory_stats = {
            'peak_memory_usage': 0.0,
            'current_memory_usage': 0.0,
            'chunks_processed': 0,
            'memory_cleanups': 0,
            'dtype_optimizations': 0,
            'total_operations': 0
        }

class VectorBTMemoryOptimizer(MemoryOptimizer):
    """
    VectorBT-specific memory optimizer with advanced features.

    Extends the base memory optimizer with VectorBT-specific optimizations
    and memory management techniques.
    """

    def __init__(self, config: Optional[MemoryConfig] = None):
        """Initialize VectorBT memory optimizer."""
        super().__init__(config)

        if not VECTORBT_AVAILABLE:
            logger.warning("VectorBT not available, falling back to base memory optimizer")

    def process_vectorbt_operations(
        self,
        data: pd.DataFrame,
        operations: List[Callable],
        **kwargs
    ) -> pd.DataFrame:
        """
        Process VectorBT operations with memory optimization.

        Args:
            data: Input DataFrame
            operations: List of VectorBT operations to apply
            **kwargs: Additional arguments

        Returns:
            Processed DataFrame
        """
        if not VECTORBT_AVAILABLE:
            logger.warning("VectorBT not available, using fallback processing")
            return self.process_with_memory_monitoring(data, lambda x: x, **kwargs)

        try:
            # Configure VectorBT for memory efficiency
            original_parallel = vbt.settings.parallel['enabled']
            vbt.settings.parallel['enabled'] = True  # Enable parallel processing

            # Process with memory monitoring
            def vectorbt_processor(df):
                result = df.copy()
                for operation in operations:
                    result = operation(result, **kwargs)
                return result

            return self.process_with_memory_monitoring(data, vectorbt_processor, **kwargs)

        except Exception as e:
            logger.error(f"VectorBT memory optimization failed: {e}")
            return self.process_with_memory_monitoring(data, lambda x: x, **kwargs)

        finally:
            # Restore original settings
            if VECTORBT_AVAILABLE:
                vbt.settings.parallel['enabled'] = original_parallel

# Global optimizer instances
_global_memory_optimizer = None
_global_vectorbt_optimizer = None

def get_memory_optimizer(config: Optional[MemoryConfig] = None) -> MemoryOptimizer:
    """Get global memory optimizer instance."""
    global _global_memory_optimizer
    if _global_memory_optimizer is None:
        _global_memory_optimizer = MemoryOptimizer(config)
    return _global_memory_optimizer

def get_vectorbt_memory_optimizer(config: Optional[MemoryConfig] = None) -> VectorBTMemoryOptimizer:
    """Get global VectorBT memory optimizer instance."""
    global _global_vectorbt_optimizer
    if _global_vectorbt_optimizer is None:
        _global_vectorbt_optimizer = VectorBTMemoryOptimizer(config)
    return _global_vectorbt_optimizer

def optimize_dataframe_memory(data: pd.DataFrame, config: Optional[MemoryConfig] = None) -> pd.DataFrame:
    """Optimize DataFrame memory usage."""
    optimizer = get_memory_optimizer(config)
    return optimizer.optimize_dataframe_dtypes(data)

def process_large_dataset_chunked(
    data: pd.DataFrame,
    processor_func: Callable[[pd.DataFrame], pd.DataFrame],
    config: Optional[MemoryConfig] = None,
    **kwargs
) -> pd.DataFrame:
    """Process large dataset in memory-efficient chunks."""
    optimizer = get_memory_optimizer(config)
    return optimizer.process_in_chunks(data, processor_func, **kwargs)

def process_with_memory_monitoring(
    data: pd.DataFrame,
    processor_func: Callable[[pd.DataFrame], pd.DataFrame],
    config: Optional[MemoryConfig] = None,
    **kwargs
) -> pd.DataFrame:
    """Process data with memory monitoring and optimization."""
    optimizer = get_memory_optimizer(config)
    return optimizer.process_with_memory_monitoring(data, processor_func, **kwargs)

# Example usage and testing
if __name__ == "__main__":
    # Create sample large dataset
    np.random.seed(42)
    n_rows = 100000
    data = pd.DataFrame({
        'close': np.random.randn(n_rows).cumsum() + 100,
        'volume': np.random.lognormal(10, 1, n_rows),
        'high': np.random.randn(n_rows).cumsum() + 100 + np.abs(np.random.randn(n_rows)),
        'low': np.random.randn(n_rows).cumsum() + 100 - np.abs(np.random.randn(n_rows))
    })

    print(f"Original data shape: {data.shape}")
    print(f"Original memory usage: {data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    # Test memory optimization
    config = MemoryConfig(
        max_memory_gb=2.0,
        chunk_size=5000,
        enable_dtype_optimization=True
    )

    optimizer = MemoryOptimizer(config)

    # Test dtype optimization
    optimized_data = optimizer.optimize_dataframe_dtypes(data)
    print(f"Optimized memory usage: {optimized_data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    # Test chunked processing
    def test_processor(df):
        return df.rolling(window=20).mean()

    result = optimizer.process_in_chunks(data, test_processor)
    print(f"Chunked processing result shape: {result.shape}")

    # Print memory stats
    stats = optimizer.get_memory_stats()
    print(f"Memory stats: {stats}")
