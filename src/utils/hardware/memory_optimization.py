"""
from .logger import system_logger
Enhanced Memory Management Utilities

This module provides memory monitoring and optimization capabilities for the training pipeline.
"""
import functools
import gc
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
import numpy as np

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
try:
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
try:
    from .logger import system_logger

except ImportError:
    system_logger = logging.getLogger('EnhancedMemoryManagement')

@dataclass
class MemoryConfig:
    """Configuration for memory management."""
    max_memory_mb: float = 2048.0  # Increased default memory limit
    warning_threshold: float = 0.75  # Earlier warning
    critical_threshold: float = 0.90  # Earlier critical threshold
    gc_threshold: float = 0.65  # More aggressive GC
    monitor_interval: float = 0.5  # More frequent monitoring
    chunk_size_multiplier: float = 0.5  # Smaller chunks when memory pressure

class MemoryMonitor:
    """Monitor memory usage during processing."""

    def __init__(self, config: Optional[MemoryConfig]=None) -> None:
        self.config = config or MemoryConfig()
        self.peak_usage = 0.0
        self.usage_history: List[Dict[str, float]] = []
        self.logger = system_logger.getChild('MemoryMonitor')
        self._last_gc_time = 0.0

    def get_usage_mb(self) -> float:
        """Get current memory usage in MB."""
        if not PSUTIL_AVAILABLE:
            return 0.0
        try:
            process = psutil.Process()
            usage_mb = process.memory_info().rss / 1024 / 1024
            self.peak_usage = max(self.peak_usage, usage_mb)
            self.usage_history.append({'timestamp': time.time(), 'usage_mb': usage_mb, 'peak_mb': self.peak_usage})
            if len(self.usage_history) > 1000:
                self.usage_history = self.usage_history[-1000:]
            return usage_mb
        except Exception as e:
            self.logger.warning(f'Error getting memory usage: {e}')
            return 0.0

    def get_peak_usage_mb(self) -> float:
        """Get peak memory usage in MB."""
        return self.peak_usage

    def get_usage_percentage(self) -> float:
        """Get current memory usage as percentage of max."""
        current_usage = self.get_usage_mb()
        return current_usage / self.config.max_memory_mb * 100 if self.config.max_memory_mb > 0 else 0

    def is_memory_pressure(self, threshold: Optional[float]=None) -> bool:
        """Check if memory usage is above threshold."""
        if threshold is None:
            threshold = self.config.warning_threshold
        current_usage = self.get_usage_mb()
        return current_usage > self.config.max_memory_mb * threshold

    def is_critical_memory(self) -> bool:
        """Check if memory usage is at critical levels."""
        return self.is_memory_pressure(self.config.critical_threshold)

    def should_trigger_gc(self) -> bool:
        """Check if garbage collection should be triggered."""
        if time.time() - self._last_gc_time < 10:
            return False
        return self.is_memory_pressure(self.config.gc_threshold)

    def trigger_gc(self) -> Dict[str, float]:
        """Trigger garbage collection and return memory stats."""
        if not self.should_trigger_gc():
            return {'before_mb': self.get_usage_mb(), 'after_mb': self.get_usage_mb(), 'freed_mb': 0.0}
        before_mb = self.get_usage_mb()
        self._last_gc_time = time.time()
        collected = gc.collect()
        after_mb = self.get_usage_mb()
        freed_mb = before_mb - after_mb
        self.logger.info(f'GC triggered: freed {freed_mb:.1f}MB, collected {collected} objects')
        return {'before_mb': before_mb, 'after_mb': after_mb, 'freed_mb': freed_mb, 'collected_objects': collected}

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics."""
        current_usage = self.get_usage_mb()
        return {'current_mb': current_usage, 'peak_mb': self.peak_usage, 'usage_percentage': self.get_usage_percentage(), 'max_mb': self.config.max_memory_mb, 'is_pressure': self.is_memory_pressure(), 'is_critical': self.is_critical_memory(), 'history_count': len(self.usage_history)}

    def log_memory_status(self, context: str='') -> None:
        """Log current memory status."""
        stats = self.get_memory_stats()
        status_msg = f"Memory {context}: {stats['current_mb']:.1f}MB/{stats['max_mb']:.1f}MB ({stats['usage_percentage']:.1f}%)"
        if stats['is_critical']:
            self.logger.error(f'🚨 CRITICAL {status_msg}')
        elif stats['is_pressure']:
            self.logger.warning(f'⚠️ PRESSURE {status_msg}')
        else:
            self.logger.info(f'💾 {status_msg}')

    def start_monitoring(self) -> None:
        """Start memory monitoring."""
        self.logger.info("🧠 Memory monitoring started")
        self.log_memory_status("initial")

def memory_efficient(max_memory_mb: float = 1024.0, optimize_dtypes: bool = True) -> None:
    """Decorator for memory-efficient processing."""

    def decorator(func: Callable) -> Callable:

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> None:
            config = MemoryConfig(max_memory_mb = max_memory_mb)
            monitor = MemoryMonitor(config)
            initial_memory = monitor.get_usage_mb()
            monitor.log_memory_status(f'before {func.__name__}')
            try:
                result = await func(*args, **kwargs)
                final_memory = monitor.get_usage_mb()
                peak_memory = monitor.get_peak_usage_mb()
                monitor.log_memory_status(f'after {func.__name__}')
                if peak_memory > max_memory_mb:
                    monitor.logger.warning(f'Peak memory usage ({peak_memory:.1f}MB) exceeded limit ({max_memory_mb:.1f}MB)')
                if optimize_dtypes and PANDAS_AVAILABLE and isinstance(result, pd.DataFrame):
                    result = optimize_dataframe_dtypes(result)
                return result
            except Exception as e:
                monitor.logger.error(f'Error in {func.__name__}: {e}')
                raise
        return async_wrapper
    return decorator

def optimize_dataframe_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Optimize DataFrame data types for memory efficiency with enhanced optimization."""
    if not PANDAS_AVAILABLE or df is None or df.empty:
        return df

    original_memory = df.memory_usage(deep=True).sum() / 1024 / 1024
    optimized_df = df.copy()

    # Optimize float columns
    for col in optimized_df.select_dtypes(include=['float64']).columns:
        try:
            # Check if float32 is sufficient
            col_min, col_max = optimized_df[col].min(), optimized_df[col].max()
            if col_min >= -3.4e38 and col_max <= 3.4e38:
                optimized_df[col] = pd.to_numeric(optimized_df[col], downcast='float')
        except Exception:
            pass

    # Optimize integer columns
    for col in optimized_df.select_dtypes(include=['int64']).columns:
        try:
            optimized_df[col] = pd.to_numeric(optimized_df[col], downcast='integer')
        except Exception:
            pass

    # Optimize object columns to categories
    for col in optimized_df.select_dtypes(include=['object']).columns:
        try:
            unique_ratio = optimized_df[col].nunique() / len(optimized_df[col])
            if unique_ratio < 0.5:  # Less than 50% unique values
                optimized_df[col] = optimized_df[col].astype('category')
        except Exception:
            pass

    # Optimize datetime columns
    for col in optimized_df.select_dtypes(include=['datetime64[ns]']).columns:
        try:
            # Convert to datetime64[s] if precision allows
            optimized_df[col] = optimized_df[col].astype('datetime64[s]')
        except Exception:
            pass

    optimized_memory = optimized_df.memory_usage(deep=True).sum() / 1024 / 1024
    savings = original_memory - optimized_memory

    if savings > 0:
        logging.info(f'DataFrame optimization: {original_memory:.1f}MB -> {optimized_memory:.1f}MB (saved {savings:.1f}MB)')

    return optimized_df

def chunk_dataframe(df: pd.DataFrame, chunk_size: int, memory_monitor: Optional[MemoryMonitor]=None) -> List[pd.DataFrame]:
    """Split DataFrame into chunks based on memory constraints."""
    if df is None or df.empty:
        return []
    if memory_monitor is None:
        memory_monitor = MemoryMonitor()
    chunks = []
    total_rows = len(df)
    for start_idx in range(0, total_rows, chunk_size):
        end_idx = min(start_idx + chunk_size, total_rows)
        chunk = df.iloc[start_idx:end_idx].copy()
        if memory_monitor.is_memory_pressure():
            memory_monitor.trigger_gc()
        chunks.append(chunk)
    return chunks

class MemoryOptimizedProcessor:
    """Memory-optimized data processor."""

    def __init__(self, config: MemoryConfig) -> None:
        self.config = config
        self.monitor = MemoryMonitor(config)
        self.logger = system_logger.getChild('MemoryOptimizedProcessor')

    def process_in_chunks(self, df: pd.DataFrame, processor_func: Callable, chunk_size: int = 10000) -> pd.DataFrame:
        """Process DataFrame in chunks to manage memory usage with adaptive chunk sizing."""
        if df is None or df.empty:
            return df

        # Adaptive chunk sizing based on memory pressure
        if self.monitor.is_memory_pressure():
            chunk_size = int(chunk_size * self.config.chunk_size_multiplier)
            self.logger.info(f'Memory pressure detected, reducing chunk size to {chunk_size}')

        self.logger.info(f'Processing DataFrame of shape {df.shape} in chunks of {chunk_size}')
        chunks = chunk_dataframe(df, chunk_size, self.monitor)
        processed_chunks = []

        for i, chunk in enumerate(chunks):
            self.logger.debug(f'Processing chunk {i + 1}/{len(chunks)}')

            # Check memory before processing each chunk
            if self.monitor.is_critical_memory():
                self.logger.warning('Critical memory usage, triggering aggressive cleanup')
                self.monitor.trigger_gc()
                # Further reduce chunk size if still critical
                if self.monitor.is_critical_memory():
                    chunk_size = max(1000, int(chunk_size * 0.5))
                    self.logger.warning(f'Further reducing chunk size to {chunk_size}')

            processed_chunk = processor_func(chunk)
            processed_chunks.append(processed_chunk)

            # Adaptive memory management
            if self.monitor.is_memory_pressure():
                self.monitor.trigger_gc()
            if (i + 1) % 10 == 0:
                self.monitor.log_memory_status(f'chunk {i + 1}/{len(chunks)}')

        if processed_chunks:
            result = pd.concat(processed_chunks, ignore_index = True)
            self.logger.info(f'Completed processing: {len(processed_chunks)} chunks -> {result.shape}')
            return result
        else:
            return pd.DataFrame()

    def stream_process(self, file_path: str, processor_func: Callable, chunk_size: int = 10000) -> pd.DataFrame:
        """Stream process a file to manage memory usage."""
        if not PANDAS_AVAILABLE:
            raise ImportError('pandas is required for stream processing')
        self.logger.info(f'Stream processing file: {file_path}')
        chunks = []
        chunk_count = 0
        try:
            for chunk in pd.read_parquet(file_path, chunksize = chunk_size):
                chunk_count += 1
                self.logger.debug(f'Processing stream chunk {chunk_count}')
                processed_chunk = processor_func(chunk)
                chunks.append(processed_chunk)
                if self.monitor.is_memory_pressure():
                    self.monitor.trigger_gc()
                if chunk_count % 10 == 0:
                    self.monitor.log_memory_status(f'stream chunk {chunk_count}')
                if self.monitor.is_critical_memory():
                    self.logger.warning('Critical memory usage, stopping stream processing')
                    break
        except Exception as e:
            self.logger.error(f'Error during stream processing: {e}')
            raise
        if chunks:
            result = pd.concat(chunks, ignore_index = True)
            self.logger.info(f'Stream processing completed: {chunk_count} chunks -> {result.shape}')
            return result
        else:
            self.logger.warning('No chunks processed')
            return pd.DataFrame()

def get_memory_usage_mb() -> float:
    """Get current memory usage in MB."""
    monitor = MemoryMonitor()
    return monitor.get_usage_mb()

def log_memory_status(context: str='') -> None:
    """Log current memory status."""
    monitor = MemoryMonitor()
    monitor.log_memory_status(context)

def trigger_gc_if_needed(max_memory_mb: float = 1024.0) -> Dict[str, float]:
    """Trigger garbage collection if memory usage is high."""
    config = MemoryConfig(max_memory_mb = max_memory_mb)
    monitor = MemoryMonitor(config)
    return monitor.trigger_gc()

# Global memory manager instance
_global_memory_manager = None

def get_memory_manager(config: Optional[MemoryConfig] = None) -> MemoryMonitor:
    """Get or create a global memory manager instance."""
    global _global_memory_manager
    if _global_memory_manager is None:
        _global_memory_manager = MemoryMonitor(config)
    return _global_memory_manager

class MemoryContext:
    """Context manager for memory monitoring."""
    def __init__(self, config: Optional[MemoryConfig] = None, context_name: str = ""):
        self.config = config or MemoryConfig()
        self.monitor = MemoryMonitor(self.config)
        self.context_name = context_name
        self.initial_memory = 0.0

    def __enter__(self):
        self.initial_memory = self.monitor.get_usage_mb()
        self.monitor.log_memory_status(f"entering {self.context_name}")
        return self.monitor

    def __exit__(self, exc_type, exc_val, exc_tb):
        final_memory = self.monitor.get_usage_mb()
        delta = final_memory - self.initial_memory
        self.monitor.log_memory_status(f"exiting {self.context_name} (delta: {delta:+.1f}MB)")
        if exc_type:
            self.monitor.logger.error(f"Exception in {self.context_name}: {exc_val}")

def memory_context(context_name: str = "", max_memory_mb: float = 1024.0):
    """Create a memory context manager."""
    config = MemoryConfig(max_memory_mb=max_memory_mb)

    return MemoryContext(config, context_name)
