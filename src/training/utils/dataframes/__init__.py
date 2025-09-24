"""
DataFrame Memory Management Utilities

This module provides utilities for optimal DataFrame handling:
- Memory-efficient operations
- Automatic memory cleanup
- Streaming processing
- Chunked operations
- Memory monitoring

Usage:
    from src.training.utils.dataframes import (
        memory_optimized_dataframe,
        with_chunked_processing,
        cleanup_dataframe,
        DataFrameManager
    )

    # Use decorator for memory optimization
    @memory_optimized_dataframe(max_memory_mb=1024)
    def process_large_data(df: pd.DataFrame) -> pd.DataFrame:
        # Your processing here
        return df

    # Use chunked processing
    with with_chunked_processing(df, chunk_size=10000) as chunks:
        for chunk in chunks:
            process_chunk(chunk)
"""

from typing import Dict, Any, Optional, Generator, Callable, Union
import pandas as pd
import numpy as np
import gc
import psutil
import logging
from functools import wraps
from contextlib import contextmanager
from pathlib import Path

logger = logging.getLogger(__name__)

class DataFrameManager:
    """Manages DataFrame memory usage and optimization."""

    def __init__(self, max_memory_mb: float = 1024.0):
        self.max_memory_mb = max_memory_mb
        self._memory_usage_history = []
        self._cleanup_callbacks = []

    def get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024

    def should_optimize_memory(self, df: pd.DataFrame) -> bool:
        """Check if DataFrame needs memory optimization."""
        estimated_mb = self._estimate_dataframe_memory(df)
        return estimated_mb > self.max_memory_mb

    def _estimate_dataframe_memory(self, df: pd.DataFrame) -> float:
        """Estimate DataFrame memory usage in MB."""
        if df.empty:
            return 0.0

        # Rough estimation: memory = rows * columns * bytes_per_cell
        bytes_per_cell = 8  # Assume float64
        memory_bytes = df.shape[0] * df.shape[1] * bytes_per_cell
        memory_mb = memory_bytes / 1024 / 1024

        return memory_mb

    def optimize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame memory usage."""
        if df.empty:
            return df

        optimized_df = df.copy()

        # Optimize data types
        for col in optimized_df.columns:
            if optimized_df[col].dtype == 'float64':
                # Check if we can downcast to float32
                if not optimized_df[col].isna().any():
                    optimized_df[col] = optimized_df[col].astype('float32')
            elif optimized_df[col].dtype == 'int64':
                # Check if we can downcast to int32 or int16
                min_val, max_val = optimized_df[col].min(), optimized_df[col].max()
                if min_val >= -32768 and max_val <= 32767:
                    optimized_df[col] = optimized_df[col].astype('int16')
                elif min_val >= -2147483648 and max_val <= 2147483647:
                    optimized_df[col] = optimized_df[col].astype('int32')

        # Optimize categorical columns
        for col in optimized_df.columns:
            if optimized_df[col].dtype == 'object':
                if len(optimized_df[col].unique()) / len(optimized_df) < 0.5:
                    optimized_df[col] = optimized_df[col].astype('category')

        return optimized_df

    def cleanup_dataframe(self, df: pd.DataFrame):
        """Clean up DataFrame and free memory."""
        if df is not None:
            del df
            gc.collect()

    def register_cleanup_callback(self, callback: Callable):
        """Register a cleanup callback."""
        self._cleanup_callbacks.append(callback)

    def run_cleanup(self):
        """Run all registered cleanup callbacks."""
        for callback in self._cleanup_callbacks:
            try:
                callback()
            except Exception as e:
                logger.warning(f"Cleanup callback failed: {e}")

    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization statistics."""
        return {
            'current_memory_mb': self.get_memory_usage(),
            'max_memory_mb': self.max_memory_mb,
            'memory_history': self._memory_usage_history[-10:],  # Last 10 measurements
            'active_callbacks': len(self._cleanup_callbacks)
        }

# Global DataFrame manager
_dataframe_manager = DataFrameManager()

def get_dataframe_manager() -> DataFrameManager:
    """Get the global DataFrame manager."""
    return _dataframe_manager

def memory_optimized_dataframe(
    max_memory_mb: float = 1024.0,
    optimize_types: bool = True,
    cleanup_on_exit: bool = True
):
    """Decorator for memory-optimized DataFrame operations."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            manager = get_dataframe_manager()

            # Check input DataFrames
            input_dfs = []
            for arg in args:
                if isinstance(arg, pd.DataFrame):
                    input_dfs.append(arg)
            for value in kwargs.values():
                if isinstance(value, pd.DataFrame):
                    input_dfs.append(value)

            # Optimize input DataFrames if needed
            optimized_inputs = {}
            for i, df in enumerate(input_dfs):
                if manager.should_optimize_memory(df):
                    optimized_inputs[f'input_{i}'] = manager.optimize_dataframe(df)

            try:
                # Execute function
                result = func(*args, **kwargs)

                # Optimize output DataFrame if needed
                if isinstance(result, pd.DataFrame) and manager.should_optimize_memory(result):
                    result = manager.optimize_dataframe(result)

                return result

            finally:
                # Cleanup
                if cleanup_on_exit:
                    for key, df in optimized_inputs.items():
                        manager.cleanup_dataframe(df)

        return wrapper
    return decorator

@contextmanager
def with_chunked_processing(
    df: pd.DataFrame,
    chunk_size: int = 10000,
    max_memory_mb: float = 512.0
) -> Generator[pd.DataFrame, None, None]:
    """Context manager for chunked DataFrame processing."""
    manager = get_dataframe_manager()

    if len(df) <= chunk_size:
        # No chunking needed
        yield df
        return

    # Process in chunks
    for i in range(0, len(df), chunk_size):
        chunk = df.iloc[i:i + chunk_size].copy()

        # Optimize chunk if needed
        if manager.should_optimize_memory(chunk):
            chunk = manager.optimize_dataframe(chunk)

        try:
            yield chunk
        finally:
            # Cleanup chunk
            manager.cleanup_dataframe(chunk)

def cleanup_dataframe(df: pd.DataFrame):
    """Clean up DataFrame and free memory."""
    get_dataframe_manager().cleanup_dataframe(df)

def optimize_dataframe_memory(df: pd.DataFrame) -> pd.DataFrame:
    """Optimize DataFrame memory usage."""
    return get_dataframe_manager().optimize_dataframe(df)

def read_parquet_optimized(
    file_path: Union[str, Path],
    chunk_size: Optional[int] = None,
    max_memory_mb: float = 512.0
) -> pd.DataFrame:
    """Memory-optimized parquet reading."""
    manager = get_dataframe_manager()

    if chunk_size is None:
        # Read entire file if memory allows
        df = pd.read_parquet(file_path)

        if manager.should_optimize_memory(df):
            df = manager.optimize_dataframe(df)

        return df

    # Read in chunks for very large files
    chunks = []
    for chunk in pd.read_parquet(file_path, chunksize=chunk_size):
        if manager.should_optimize_memory(chunk):
            chunk = manager.optimize_dataframe(chunk)
        chunks.append(chunk)

    return pd.concat(chunks, ignore_index=True)

def write_parquet_optimized(
    df: pd.DataFrame,
    file_path: Union[str, Path],
    chunk_size: Optional[int] = None
) -> None:
    """Memory-optimized parquet writing."""
    if chunk_size is None or len(df) <= chunk_size:
        df.to_parquet(file_path, index=False)
        return

    # Write in chunks for very large DataFrames
    for i in range(0, len(df), chunk_size):
        chunk = df.iloc[i:i + chunk_size]
        chunk.to_parquet(f"{file_path}.part_{i//chunk_size:03d}", index=False)

    # Combine parts (this could be optimized further)
    # For now, just write the full DataFrame
    df.to_parquet(file_path, index=False)

class DataFrameProcessor:
    """High-level DataFrame processing with memory management."""

    def __init__(self, max_memory_mb: float = 1024.0):
        self.manager = DataFrameManager(max_memory_mb)

    def process_with_memory_management(
        self,
        df: pd.DataFrame,
        operations: List[Callable[[pd.DataFrame], pd.DataFrame]],
        chunk_size: Optional[int] = None
    ) -> pd.DataFrame:
        """Process DataFrame with memory management."""
        if chunk_size and len(df) > chunk_size:
            # Process in chunks
            result_chunks = []

            for i in range(0, len(df), chunk_size):
                chunk = df.iloc[i:i + chunk_size].copy()

                # Apply operations to chunk
                for operation in operations:
                    chunk = operation(chunk)

                result_chunks.append(chunk)

            return pd.concat(result_chunks, ignore_index=True)

        # Process entire DataFrame
        for operation in operations:
            df = operation(df)

        return df

    def apply_operation_with_cleanup(
        self,
        df: pd.DataFrame,
        operation: Callable[[pd.DataFrame], pd.DataFrame]
    ) -> pd.DataFrame:
        """Apply operation with automatic cleanup."""
        try:
            result = operation(df)
            return result
        finally:
            self.manager.cleanup_dataframe(df)

# Convenience functions
def get_memory_usage() -> float:
    """Get current memory usage in MB."""
    return get_dataframe_manager().get_memory_usage()

def log_memory_usage(operation: str = "current"):
    """Log current memory usage."""
    memory_mb = get_memory_usage()
    logger.info(f"Memory usage ({operation}): {memory_mb:.2f} MB")

# Export all functions and classes
__all__ = [
    'DataFrameManager', 'get_dataframe_manager',
    'memory_optimized_dataframe', 'with_chunked_processing',
    'cleanup_dataframe', 'optimize_dataframe_memory',
    'read_parquet_optimized', 'write_parquet_optimized',
    'DataFrameProcessor', 'get_memory_usage', 'log_memory_usage'
]