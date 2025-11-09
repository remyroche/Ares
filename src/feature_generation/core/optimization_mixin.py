"""
Optimization Mixin

This mixin provides common optimization functionality for feature generators,
including data preprocessing, memory optimization, and performance tuning.

Usage:
    class MyFeatureGenerator(VectorizedFeatureGenerator, OptimizationMixin):
        def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
            # Use self.optimize_dataframe_processing() for data optimization
            optimized_data = self.optimize_dataframe_processing(data)
            return self._calculate_feature(optimized_data)
"""

import numpy as np
import pandas as pd
import logging
from typing import Any, Dict, List, Optional, Union
from functools import wraps
import time
import gc

class OptimizationMixin:
    """Mixin class that provides optimization capabilities for feature generators."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Initialize optimization settings
        self.enable_memory_optimization = getattr(self, 'enable_memory_optimization', True)
        self.enable_data_compression = getattr(self, 'enable_data_compression', True)
        self.enable_chunked_processing = getattr(self, 'enable_chunked_processing', True)
        self.chunk_size = getattr(self, 'chunk_size', 10000)
        self.memory_threshold_mb = getattr(self, 'memory_threshold_mb', 100)

        # Performance tracking
        self.optimization_stats = {
            'memory_optimizations': 0,
            'data_compressions': 0,
            'chunked_operations': 0,
            'total_optimization_time': 0.0,
            'memory_saved_mb': 0.0
        }

        # Setup logger
        self.logger = logging.getLogger(self.__class__.__name__)

    def optimize_dataframe_processing(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Optimize DataFrame for efficient processing.

        Args:
            data: Input DataFrame

        Returns:
            Optimized DataFrame
        """
        start_time = time.time()

        if not self.enable_memory_optimization:
            return data

        try:
            # Check memory usage
            memory_usage_mb = data.memory_usage(deep=True).sum() / (1024**2)

            if memory_usage_mb > self.memory_threshold_mb:
                self.logger.debug(f"Optimizing DataFrame: {memory_usage_mb:.2f}MB > {self.memory_threshold_mb}MB")

                # Optimize dtypes
                optimized_data = self._optimize_dtypes(data)

                # Compress data if enabled
                if self.enable_data_compression:
                    optimized_data = self._compress_data(optimized_data)

                # Update stats
                self.optimization_stats['memory_optimizations'] += 1
                memory_saved = memory_usage_mb - (optimized_data.memory_usage(deep=True).sum() / (1024**2))
                self.optimization_stats['memory_saved_mb'] += memory_saved

                self.logger.debug(f"Memory optimization saved {memory_saved:.2f}MB")

                return optimized_data
            else:
                return data

        except Exception as e:
            self.logger.warning(f"DataFrame optimization failed: {e}")
            return data
        finally:
            self.optimization_stats['total_optimization_time'] += time.time() - start_time

    def _optimize_dtypes(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame dtypes for memory efficiency with early downcasting."""
        # Always work on a copy to avoid modifying the original data
        optimized_data = data.copy()

        for column in optimized_data.columns:
            col_data = optimized_data[column]

            # Skip optimization for columns that might lose data integrity
            if col_data.dtype == 'object':
                # Only try to convert if all values are actually numeric strings
                try:
                    # Check if all non-null values can be converted to numeric
                    non_null_values = col_data.dropna()
                    if len(non_null_values) > 0:
                        # Test conversion on a sample to avoid errors
                        sample = non_null_values.head(10) if len(non_null_values) > 10 else non_null_values
                        pd.to_numeric(sample, errors='coerce')
                        # If no NaN introduced, proceed with conversion
                        converted = pd.to_numeric(non_null_values, errors='coerce')
                        if not pd.isna(converted).any():
                            # Use copy=False when possible for string→numeric conversions
                            # DEBUG: Log before conversion
                            self.logger.debug(f"OPTIMIZATION DEBUG: Converting column '{column}' from object to numeric with downcast='integer'")
                            self.logger.debug(f"OPTIMIZATION DEBUG: Before conversion - non-null count: {len(non_null_values)}, NaN count: {col_data.isna().sum()}")
                            optimized_data[column] = pd.to_numeric(col_data, downcast='integer')
                            # DEBUG: Log after conversion
                            self.logger.debug(f"OPTIMIZATION DEBUG: After integer conversion - NaN count: {optimized_data[column].isna().sum()}")
                        else:
                            # Try float conversion if integer fails
                            converted_float = pd.to_numeric(non_null_values, errors='coerce')
                            if not pd.isna(converted_float).any():
                                # DEBUG: Log before conversion
                                self.logger.debug(f"OPTIMIZATION DEBUG: Converting column '{column}' from object to numeric with downcast='float'")
                                self.logger.debug(f"OPTIMIZATION DEBUG: Before conversion - non-null count: {len(non_null_values)}, NaN count: {col_data.isna().sum()}")
                                optimized_data[column] = pd.to_numeric(col_data, downcast='float')
                                # DEBUG: Log after conversion
                                self.logger.debug(f"OPTIMIZATION DEBUG: After float conversion - NaN count: {optimized_data[column].isna().sum()}")
                            else:
                                # FIX: Don't convert if both integer and float fail - keep original
                                self.logger.warning(f"OPTIMIZATION: Skipping conversion for column '{column}' - both integer and float downcast failed, keeping original data")
                except (ValueError, TypeError):
                    # Keep as object if conversion fails
                    self.logger.debug(f"OPTIMIZATION DEBUG: Keeping column '{column}' as object due to conversion error")
                    pass

            elif col_data.dtype == 'int64':
                # Downcast integers with copy=False, but only if no data loss
                try:
                    if col_data.min() >= np.iinfo(np.int8).min and col_data.max() <= np.iinfo(np.int8).max:
                        optimized_data[column] = col_data.astype(np.int8, copy=False)
                    elif col_data.min() >= np.iinfo(np.int16).min and col_data.max() <= np.iinfo(np.int16).max:
                        optimized_data[column] = col_data.astype(np.int16, copy=False)
                    elif col_data.min() >= np.iinfo(np.int32).min and col_data.max() <= np.iinfo(np.int32).max:
                        optimized_data[column] = col_data.astype(np.int32, copy=False)
                except (ValueError, TypeError, OverflowError):
                    # Keep original dtype if downcasting fails
                    pass

            elif col_data.dtype == 'float64':
                # Downcast floats with copy=False, but only if no precision loss
                try:
                    if col_data.min() >= np.finfo(np.float32).min and col_data.max() <= np.finfo(np.float32).max:
                        optimized_data[column] = col_data.astype(np.float32, copy=False)
                except (ValueError, TypeError, OverflowError):
                    # Keep original dtype if downcasting fails
                    pass

        return optimized_data

    def _compress_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Compress DataFrame data for memory efficiency."""
        if not self.enable_data_compression:
            return data

        try:
            # Use pandas compression for categorical data
            compressed_data = data.copy()

            for column in compressed_data.columns:
                if compressed_data[column].dtype == 'object':
                    # Convert to category if beneficial
                    data_length = len(compressed_data)
                    if data_length > 0 and compressed_data[column].nunique() / data_length < 0.5:
                        compressed_data[column] = compressed_data[column].astype('category')

            self.optimization_stats['data_compressions'] += 1
            return compressed_data

        except Exception as e:
            self.logger.warning(f"Data compression failed: {e}")
            return data

    def chunked_processing(self, data: pd.DataFrame, func: callable,
                          chunk_size: Optional[int] = None, **kwargs) -> pd.DataFrame:
        """
        Process DataFrame in chunks for memory efficiency.

        Args:
            data: Input DataFrame
            func: Function to apply to each chunk
            chunk_size: Size of each chunk (uses default if None)
            **kwargs: Additional arguments for func

        Returns:
            Processed DataFrame
        """
        if not self.enable_chunked_processing:
            return func(data, **kwargs)

        chunk_size = chunk_size or self.chunk_size

        if len(data) <= chunk_size:
            return func(data, **kwargs)

        try:
            results = []

            for i in range(0, len(data), chunk_size):
                chunk = data.iloc[i:i + chunk_size]
                chunk_result = func(chunk, **kwargs)
                results.append(chunk_result)

                # Force garbage collection for large chunks
                if i % (chunk_size * 5) == 0:
                    gc.collect()

            self.optimization_stats['chunked_operations'] += 1

            if results:
                return pd.concat(results, ignore_index=False)
            else:
                return data

        except Exception as e:
            self.logger.warning(f"Chunked processing failed: {e}")
            return func(data, **kwargs)

    def optimize_memory_usage(self) -> None:
        """Optimize memory usage by forcing garbage collection."""
        if self.enable_memory_optimization:
            gc.collect()

    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization statistics."""
        stats = self.optimization_stats.copy()

        if stats['total_optimization_time'] > 0:
            denominator = stats['memory_optimizations'] + stats['data_compressions'] + stats['chunked_operations']
            if denominator > 0:  # Check for division by zero
                stats['average_optimization_time'] = stats['total_optimization_time'] / denominator
            else:
                stats['average_optimization_time'] = 0
        else:
            stats['average_optimization_time'] = 0

        # Include VectorBT performance stats if available
        if hasattr(self, 'performance_stats'):
            vectorbt_stats = self.performance_stats.copy()
            # Add VectorBT stats to optimization stats
            stats.update({
                'vectorbt_operations': vectorbt_stats.get('vectorbt_operations', 0),
                'pandas_fallbacks': vectorbt_stats.get('pandas_fallbacks', 0),
                'gpu_accelerations': vectorbt_stats.get('gpu_accelerations', 0),
                'total_operations': vectorbt_stats.get('total_operations', 0),
                'total_time': vectorbt_stats.get('total_time', 0.0),
                'cache_hits': vectorbt_stats.get('cache_hits', 0),
                'cache_misses': vectorbt_stats.get('cache_misses', 0)
            })

        return stats

    def reset_optimization_stats(self) -> None:
        """Reset optimization statistics."""
        self.optimization_stats = {
            'memory_optimizations': 0,
            'data_compressions': 0,
            'chunked_operations': 0,
            'total_optimization_time': 0.0,
            'memory_saved_mb': 0.0
        }

def optimization_required(memory_threshold_mb: float = 100):
    """Decorator to automatically apply optimization based on memory usage."""
    def decorator(func):
        @wraps(func)
        def wrapper(self, data: pd.DataFrame, *args, **kwargs):
            if hasattr(self, 'optimize_dataframe_processing'):
                memory_usage_mb = data.memory_usage(deep=True).sum() / (1024**2)
                if memory_usage_mb > memory_threshold_mb:
                    data = self.optimize_dataframe_processing(data)
            return func(self, data, *args, **kwargs)
        return wrapper
    return decorator
