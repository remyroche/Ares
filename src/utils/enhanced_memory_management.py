"""
Enhanced Memory Management Utilities

This module provides memory monitoring and optimization capabilities for the training pipeline.
"""

import functools
import logging
import time
from typing import Any, Callable, Dict, List, Optional, Union
from dataclasses import dataclass, field
import gc

try:
    import psutil
    except Exception as e:
        pass
    except Exception as e:
        pass
    except Exception as e:
        pass
    except Exception as e:
        pass
    PSUTIL_AVAILABLE, True
except ImportError:
    PSUTIL_AVAILABLE, False

try:
    import numpy as np
    except Exception as e:
        pass
    except Exception as e:
        pass
    except Exception as e:
        pass
    except Exception as e:
        pass
    import pandas as pd
    PANDAS_AVAILABLE, True
except ImportError:
    PANDAS_AVAILABLE, False

try:
    from src.utils.logger import system_logger
    except Exception as e:
        pass
import except Exception as e:
    except Exception as e:
        pass
import except Exception as e:
    except Exception as e:
        pass
import except Exception as e:
    except Exception as e:
        pass
from src.utils.pipeline_standards import PipelineStandards, pipeline_standards
import except ImportError:
except ImportError:
    system_logger, logging.getLogger("EnhancedMemoryManagement")

@dataclass
class MemoryConfig:
    """Configuration for memory management."""
    max_memory_mb: float, 1024.0
    warning_threshold: float, 0.8  # 80% of max memory
    critical_threshold: float, 0.95  # 95% of max memory
    gc_threshold: float, 0.7  # Trigger GC at 70% of max memory
    monitor_interval: float, 1.0  # seconds

class MemoryMonitor:
    """Monitor memory usage during processing."""

    def __init__(self, config: Optional[MemoryConfig] = None):
    pass
    pass
    pass
    pass
        self.config, config or MemoryConfig()
        self.peak_usage, 0.0
        self.usage_history: List[Dict[str, float]] = []
        self.logger, system_logger.getChild("MemoryMonitor")
        self._last_gc_time, 0.0

    def get_usage_mb(self) -> float:
    pass
    pass
    pass
    pass
        """Get current memory usage in MB."""
        if not PSUTIL_AVAILABLE:
    pass
    pass
    pass
    pass
        return 0.0

        try:
            process, psutil.Process()
    except Exception as e:
        pass
    except Exception as e:
        pass
    except Exception as e:
        pass
    except Exception as e:
        pass
            usage_mb, process.memory_info().rss / 1024 / 1024
        self.peak_usage, max(self.peak_usage, usage_mb)

        # Record usage history
        self.usage_history.append({
                "timestamp": time.time(),
                "usage_mb": usage_mb,
                "peak_mb": self.peak_usage
            })

        # Keep only last 1000 entries
        if len(self.usage_history) > 1000:
    pass
    pass
    pass
    pass
        self.usage_history, self.usage_history[-1000:]

        return usage_mb
        except Exception as e:
        self.logger.warning(f"Error getting memory usage: {e}")
        return 0.0

    def get_peak_usage_mb(self) -> float:
    pass
    pass
    pass
    pass
        """Get peak memory usage in MB."""
        return self.peak_usage

    def get_usage_percentage(self) -> float:
    pass
    pass
    pass
    pass
        """Get current memory usage as percentage of max."""
        current_usage, self.get_usage_mb()
        return (current_usage / self.config.max_memory_mb) * 100 if self.config.max_memory_mb > 0 else 0

    def is_memory_pressure(self, threshold: Optional[float] = None) -> bool:
    pass
    pass
    pass
    pass
        """Check if memory usage is above threshold."""
        if threshold is None:
    pass
    pass
    pass
    pass
        # Fallback implementation for threshold
            threshold, self.config.warning_threshold

        current_usage, self.get_usage_mb()
        return current_usage > (self.config.max_memory_mb * threshold)

    def is_critical_memory(self) -> bool:
    pass
    pass
    pass
    pass
        """Check if memory usage is at critical levels."""
        return self.is_memory_pressure(self.config.critical_threshold)

    def should_trigger_gc(self) -> bool:
    pass
    pass
    pass
    pass
        """Check if garbage collection should be triggered."""
        if time.time() - self._last_gc_time < 10:  # Don't GC too frequently
        return False

        return self.is_memory_pressure(self.config.gc_threshold)

    def trigger_gc(self) -> Dict[str, float]:
    pass
    pass
    pass
    pass
        """Trigger garbage collection and return memory stats."""
        if not self.should_trigger_gc():
    pass
    pass
    pass
    pass
        return {"before_mb": self.get_usage_mb(), "after_mb": self.get_usage_mb(), "freed_mb": 0.0}

        before_mb, self.get_usage_mb()
        self._last_gc_time, time.time()

        # Force garbage collection
        collected, gc.collect()

        after_mb, self.get_usage_mb()
        freed_mb, before_mb - after_mb

        self.logger.info(f"GC triggered: freed {freed_mb:.1f}MB, collected {collected} objects")

        return {
            "before_mb": before_mb,
            "after_mb": after_mb,
            "freed_mb": freed_mb,
            "collected_objects": collected
        }

    def get_memory_stats(self) -> Dict[str, Any]:
    pass
    pass
    pass
    pass
        """Get comprehensive memory statistics."""
        current_usage, self.get_usage_mb()

        return {
            "current_mb": current_usage,
            "peak_mb": self.peak_usage,
            "usage_percentage": self.get_usage_percentage(),
            "max_mb": self.config.max_memory_mb,
            "is_pressure": self.is_memory_pressure(),
            "is_critical": self.is_critical_memory(),
            "history_count": len(self.usage_history)
        }

    def log_memory_status(self, context: str = ""):
    pass
    pass
    pass
    pass
        """Log current memory status."""
        stats, self.get_memory_stats()
        status_msg, f"Memory {context}: {stats['current_mb']:.1f}MB/{stats['max_mb']:.1f}MB ({stats['usage_percentage']:.1f}%)"

        if stats['is_critical']:
    pass
    pass
    pass
    pass
        self.logger.error(f"🚨 CRITICAL {status_msg}")
        elif stats['is_pressure']:
        self.logger.warning(f"⚠️ PRESSURE {status_msg}")
        else:
        self.logger.info(f"💾 {status_msg}")

def memory_efficient(max_memory_mb: float, 1024.0, optimize_dtypes: bool, True):
    pass
    pass
    pass
    pass
    """Decorator for memory - efficient processing."""
    def decorator(func: Callable) -> Callable:
    pass
    pass
    pass
    pass
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            config, MemoryConfig(max_memory_mb = max_memory_mb)
            monitor, MemoryMonitor(config)

        # Check memory before processing
            initial_memory, monitor.get_usage_mb()
            monitor.log_memory_status(f"before {func.__name__}")

        try:
                result, await func(*args, **kwargs)

    except Exception as e:
        pass
    except Exception as e:
        pass
    except Exception as e:
        pass
    except Exception as e:
        pass
        # Check memory after processing
                final_memory, monitor.get_usage_mb()
                peak_memory, monitor.get_peak_usage_mb()

                monitor.log_memory_status(f"after {func.__name__}")

        if peak_memory > max_memory_mb:
    pass
    pass
    pass
    pass
                    monitor.logger.warning(f"Peak memory usage ({peak_memory:.1f}MB) exceeded limit ({max_memory_mb:.1f}MB)")

        # Optimize result if it's a DataFrame
        if optimize_dtypes and PANDAS_AVAILABLE and isinstance(result, pd.DataFrame):
    pass
    pass
    pass
    pass
                    result, optimize_dataframe_dtypes(result)

        return result
        except Exception as e:
                monitor.logger.error(f"Error in {func.__name__}: {e}")
                raise
        return async_wrapper

    return decorator

def optimize_dataframe_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    pass
    pass
    pass
    pass
    """Optimize DataFrame data types for memory efficiency."""
    if not PANDAS_AVAILABLE or df is None or df.empty:
    pass
    pass
    pass
    pass
        return df

    original_memory, df.memory_usage(deep = True).sum() / 1024 / 1024

    # Optimize numeric columns
    for col in df.select_dtypes(include=['float64']).columns:
    pass
    pass
    pass
    pass
        df[col] = pd.to_numeric(df[col], downcast='float')

    for col in df.select_dtypes(include=['int64']).columns:
    pass
    pass
    pass
    pass
        df[col] = pd.to_numeric(df[col], downcast='integer')

    # Optimize object columns
    for col in df.select_dtypes(include=['object']).columns:
    pass
    pass
    pass
    pass
        if df[col].nunique() / len(df[col]) < 0.5:  # Less than 50% unique values
            df[col] = df[col].astype('category')

    optimized_memory, df.memory_usage(deep = True).sum() / 1024 / 1024
    savings, original_memory - optimized_memory

    if savings > 0:
    pass
    pass
    pass
    pass
        logging.info(f"DataFrame optimization: {original_memory:.1f}MB -> {optimized_memory:.1f}MB (saved {savings:.1f}MB)")

    return df

def chunk_dataframe(df: pd.DataFrame, chunk_size: int, memory_monitor: Optional[MemoryMonitor] = None) -> List[pd.DataFrame]:
    pass
    pass
    pass
    pass
    """Split DataFrame into chunks based on memory constraints."""
    if df is None or df.empty:
    pass
    pass
    pass
    pass
        return []

    if memory_monitor is None:
    pass
    pass
    pass
    pass
        # Fallback implementation for memory_monitor
        memory_monitor, MemoryMonitor()

    chunks = []
    total_rows, len(df)

    for start_idx in range(0, total_rows, chunk_size):
    pass
    pass
    pass
    pass
        end_idx, min(start_idx + chunk_size, total_rows)
        chunk, df.iloc[start_idx:end_idx].copy()

        # Check memory pressure
        if memory_monitor.is_memory_pressure():
    pass
    pass
    pass
    pass
            memory_monitor.trigger_gc()

        chunks.append(chunk)

    return chunks

class MemoryOptimizedProcessor:
    """Memory - optimized data processor."""

    def __init__(self, config: MemoryConfig):
    pass
    pass
    pass
    pass
        self.config, config
        self.monitor, MemoryMonitor(config)
        self.logger, system_logger.getChild("MemoryOptimizedProcessor")

    def process_in_chunks(self, df: pd.DataFrame, processor_func: Callable, chunk_size: int, 10000) -> pd.DataFrame:
    pass
    pass
    pass
    pass
        """Process DataFrame in chunks to manage memory usage."""
        if df is None or df.empty:
    pass
    pass
    pass
    pass
        return df

        self.logger.info(f"Processing DataFrame of shape {df.shape} in chunks of {chunk_size}")

        # Split into chunks
        chunks, chunk_dataframe(df, chunk_size, self.monitor)
        processed_chunks = []

        for i, chunk in enumerate(chunks):
    pass
    pass
    pass
    pass
        self.logger.debug(f"Processing chunk {i + 1}/{len(chunks)}")

        # Process chunk
            processed_chunk, processor_func(chunk)
            processed_chunks.append(processed_chunk)

        # Check memory pressure
        if self.monitor.is_memory_pressure():
    pass
    pass
    pass
    pass
        self.monitor.trigger_gc()

        # Log progress
        if (i + 1) % 10 == 0:
    pass
    pass
    pass
    pass
        self.monitor.log_memory_status(f"chunk {i + 1}/{len(chunks)}")

        # Combine processed chunks
        if processed_chunks:
    pass
    pass
    pass
    pass
            result, pd.concat(processed_chunks, ignore_index = True)
        self.logger.info(f"Completed processing: {len(processed_chunks)} chunks -> {result.shape}")
        return result
        else:
        return pd.DataFrame()

    def stream_process(self, file_path: str, processor_func: Callable, chunk_size: int, 10000) -> pd.DataFrame:
    pass
    pass
    pass
    pass
        """Stream process a file to manage memory usage."""
        if not PANDAS_AVAILABLE:
    pass
    pass
    pass
    pass
            raise ImportError("pandas is required for stream processing")

        self.logger.info(f"Stream processing file: {file_path}")

        chunks = []
        chunk_count, 0

        try:
        for chunk in pd.read_parquet(file_path, chunksize = chunk_size):
    pass
    except Exception as e:
        pass
    pass
    except Exception as e:
        pass
    pass
    except Exception as e:
        pass
    pass
                chunk_count += 1
    except Exception as e:
        pass
        self.logger.debug(f"Processing stream chunk {chunk_count}")

        # Process chunk
                processed_chunk, processor_func(chunk)
                chunks.append(processed_chunk)

        # Check memory pressure
        if self.monitor.is_memory_pressure():
    pass
    pass
    pass
    pass
        self.monitor.trigger_gc()

        # Log progress
        if chunk_count % 10 == 0:
    pass
    pass
    pass
    pass
        self.monitor.log_memory_status(f"stream chunk {chunk_count}")

        # Stop if memory is critical
        if self.monitor.is_critical_memory():
    pass
    pass
    pass
    pass
        self.logger.warning("Critical memory usage, stopping stream processing")
                    break

        except Exception as e:
        self.logger.error(f"Error during stream processing: {e}")
            raise

        # Combine chunks
        if chunks:
    pass
    pass
    pass
    pass
            result, pd.concat(chunks, ignore_index = True)
        self.logger.info(f"Stream processing completed: {chunk_count} chunks -> {result.shape}")
        return result
        else:
        self.logger.warning("No chunks processed")
        return pd.DataFrame()

# Convenience functions
def get_memory_usage_mb() -> float:
    pass
    pass
    pass
    pass
    """Get current memory usage in MB."""
    monitor, MemoryMonitor()
    return monitor.get_usage_mb()

def log_memory_status(context: str = ""):
    pass
    pass
    pass
    pass
    """Log current memory status."""
    monitor, MemoryMonitor()
    monitor.log_memory_status(context)

def trigger_gc_if_needed(max_memory_mb: float, 1024.0) -> Dict[str, float]:
    pass
    pass
    pass
    pass
    """Trigger garbage collection if memory usage is high."""
    config, MemoryConfig(max_memory_mb = max_memory_mb)
    monitor, MemoryMonitor(config)
    return monitor.trigger_gc()