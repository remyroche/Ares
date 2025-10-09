"""
Memory Management Utilities for Pre-Training Pipeline

This module provides standardized memory management utilities to prevent memory leaks
and optimize memory usage across all pre-training components.
"""

import gc
import logging
import psutil
import tracemalloc
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Union, Callable
import warnings

import numpy as np
import pandas as pd
from pathlib import Path

# Import tprint utilities
try:
    from src.utils.tprint import tprint, tprint_info, tprint_warning, tprint_error, tprint_success
    TPRINT_AVAILABLE = True
except ImportError:
    TPRINT_AVAILABLE = False
    def tprint(*args, **kwargs): print(*args, **kwargs)
    def tprint_info(*args, **kwargs): print("INFO:", *args, **kwargs)
    def tprint_warning(*args, **kwargs): print("WARNING:", *args, **kwargs)
    def tprint_error(*args, **kwargs): print("ERROR:", *args, **kwargs)
    def tprint_success(*args, **kwargs): print("SUCCESS:", *args, **kwargs)

logger = logging.getLogger(__name__)


class MemoryManager:
    """Centralized memory management for pre-training pipeline."""
    
    def __init__(self, memory_limit_gb: float = 8.0, enable_monitoring: bool = True):
        """
        Initialize memory manager.
        
        Args:
            memory_limit_gb: Memory limit in GB
            enable_monitoring: Whether to enable memory monitoring
        """
        self.memory_limit_gb = memory_limit_gb
        self.memory_limit_bytes = memory_limit_gb * 1024**3
        self.enable_monitoring = enable_monitoring
        self.memory_checkpoints: List[Dict[str, Any]] = []
        
        if enable_monitoring:
            tracemalloc.start()
            tprint_info(f"🧠 Memory manager initialized with {memory_limit_gb}GB limit")
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            
            return {
                'rss_mb': memory_info.rss / 1024 / 1024,  # Resident Set Size
                'vms_mb': memory_info.vms / 1024 / 1024,  # Virtual Memory Size
                'percent': process.memory_percent(),
                'available_gb': psutil.virtual_memory().available / 1024**3,
                'total_gb': psutil.virtual_memory().total / 1024**3
            }
        except Exception as e:
            logger.warning(f"Failed to get memory usage: {e}")
            return {'rss_mb': 0, 'vms_mb': 0, 'percent': 0, 'available_gb': 0, 'total_gb': 0}
    
    def is_memory_pressure(self, threshold: float = 0.8) -> bool:
        """Check if memory usage is above threshold."""
        usage = self.get_memory_usage()
        return usage['percent'] > (threshold * 100)
    
    def optimize_dataframe(self, df: pd.DataFrame, target_dtype: str = 'float32') -> pd.DataFrame:
        """
        Optimize DataFrame memory usage.
        
        Args:
            df: DataFrame to optimize
            target_dtype: Target dtype for numeric columns
            
        Returns:
            Optimized DataFrame
        """
        if df.empty:
            return df
        
        original_memory = df.memory_usage(deep=True).sum()
        optimized_df = df.copy()
        
        # Optimize numeric columns
        for col in optimized_df.select_dtypes(include=[np.number]).columns:
            col_data = optimized_df[col]
            if col_data.dtype != target_dtype:
                try:
                    # Try to convert to target dtype
                    if target_dtype == 'float32':
                        optimized_df[col] = pd.to_numeric(col_data, downcast='float')
                    elif target_dtype == 'int32':
                        optimized_df[col] = pd.to_numeric(col_data, downcast='integer')
                except (ValueError, TypeError):
                    # Keep original dtype if conversion fails
                    pass
        
        # Optimize categorical columns
        for col in optimized_df.select_dtypes(include=['object']).columns:
            if optimized_df[col].nunique() / len(optimized_df) < 0.5:  # Low cardinality
                optimized_df[col] = optimized_df[col].astype('category')
        
        optimized_memory = optimized_df.memory_usage(deep=True).sum()
        memory_saved = original_memory - optimized_memory
        
        if memory_saved > 0:
            tprint_info(f"💾 DataFrame optimized: {memory_saved / 1024 / 1024:.1f}MB saved")
        
        return optimized_df
    
    def chunk_dataframe(self, df: pd.DataFrame, chunk_size: int = 10000) -> List[pd.DataFrame]:
        """
        Split DataFrame into memory-efficient chunks.
        
        Args:
            df: DataFrame to chunk
            chunk_size: Size of each chunk
            
        Returns:
            List of DataFrame chunks
        """
        chunks = []
        for i in range(0, len(df), chunk_size):
            chunk = df.iloc[i:i + chunk_size].copy()
            chunk = self.optimize_dataframe(chunk)
            chunks.append(chunk)
        
        tprint_info(f"📦 DataFrame split into {len(chunks)} chunks of max {chunk_size} rows")
        return chunks
    
    def cleanup_memory(self, force_gc: bool = True) -> Dict[str, Any]:
        """
        Clean up memory by running garbage collection.
        
        Args:
            force_gc: Whether to force garbage collection
            
        Returns:
            Memory cleanup statistics
        """
        before_usage = self.get_memory_usage()
        
        if force_gc:
            collected = gc.collect()
            tprint_info(f"🗑️ Garbage collection freed {collected} objects")
        
        after_usage = self.get_memory_usage()
        
        memory_freed = before_usage['rss_mb'] - after_usage['rss_mb']
        
        return {
            'memory_freed_mb': memory_freed,
            'before_usage': before_usage,
            'after_usage': after_usage,
            'objects_collected': collected if force_gc else 0
        }
    
    @contextmanager
    def memory_checkpoint(self, name: str = "checkpoint"):
        """
        Context manager for memory checkpointing.
        
        Args:
            name: Name of the checkpoint
        """
        before_usage = self.get_memory_usage()
        checkpoint = {
            'name': name,
            'before': before_usage,
            'timestamp': pd.Timestamp.now()
        }
        
        try:
            yield checkpoint
        finally:
            after_usage = self.get_memory_usage()
            checkpoint['after'] = after_usage
            checkpoint['memory_delta_mb'] = after_usage['rss_mb'] - before_usage['rss_mb']
            
            self.memory_checkpoints.append(checkpoint)
            
            if checkpoint['memory_delta_mb'] > 100:  # More than 100MB increase
                tprint_warning(f"⚠️ Memory checkpoint '{name}' used {checkpoint['memory_delta_mb']:.1f}MB")
    
    def safe_dataframe_operation(self, df: pd.DataFrame, operation: Callable, 
                                *args, **kwargs) -> Any:
        """
        Safely perform DataFrame operation with memory management.
        
        Args:
            df: DataFrame to operate on
            operation: Operation to perform
            *args: Arguments for operation
            **kwargs: Keyword arguments for operation
            
        Returns:
            Result of operation
        """
        with self.memory_checkpoint(f"operation_{operation.__name__}"):
            # Check memory pressure before operation
            if self.is_memory_pressure():
                tprint_warning("⚠️ High memory pressure detected, cleaning up before operation")
                self.cleanup_memory()
            
            # Perform operation
            result = operation(df, *args, **kwargs)
            
            # Clean up if result is large
            if hasattr(result, 'memory_usage'):
                result_memory = result.memory_usage(deep=True).sum()
                if result_memory > 100 * 1024 * 1024:  # More than 100MB
                    tprint_info(f"📊 Large result generated: {result_memory / 1024 / 1024:.1f}MB")
            
            return result
    
    def monitor_memory_usage(self, operation_name: str, 
                           operation: Callable, *args, **kwargs) -> Any:
        """
        Monitor memory usage during operation execution.
        
        Args:
            operation_name: Name of the operation
            operation: Operation to execute
            *args: Arguments for operation
            **kwargs: Keyword arguments for operation
            
        Returns:
            Result of operation
        """
        with self.memory_checkpoint(operation_name):
            try:
                result = operation(*args, **kwargs)
                tprint_success(f"✅ Operation '{operation_name}' completed successfully")
                return result
            except MemoryError as e:
                tprint_error(f"❌ Memory error in '{operation_name}': {e}")
                self.cleanup_memory(force_gc=True)
                raise
            except Exception as e:
                tprint_error(f"❌ Error in '{operation_name}': {e}")
                raise
    
    def get_memory_report(self) -> Dict[str, Any]:
        """Get comprehensive memory usage report."""
        current_usage = self.get_memory_usage()
        
        report = {
            'current_usage': current_usage,
            'memory_limit_gb': self.memory_limit_gb,
            'memory_pressure': self.is_memory_pressure(),
            'checkpoints': self.memory_checkpoints[-10:],  # Last 10 checkpoints
            'total_checkpoints': len(self.memory_checkpoints)
        }
        
        if self.enable_monitoring:
            try:
                current, peak = tracemalloc.get_traced_memory()
                report['traced_memory'] = {
                    'current_mb': current / 1024 / 1024,
                    'peak_mb': peak / 1024 / 1024
                }
            except Exception:
                pass
        
        return report


# Global memory manager instance
_memory_manager: Optional[MemoryManager] = None


def get_memory_manager() -> MemoryManager:
    """Get the global memory manager instance."""
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = MemoryManager()
    return _memory_manager


def optimize_dataframe_memory(df: pd.DataFrame, target_dtype: str = 'float32') -> pd.DataFrame:
    """Optimize DataFrame memory usage using the global memory manager."""
    return get_memory_manager().optimize_dataframe(df, target_dtype)


def safe_dataframe_operation(df: pd.DataFrame, operation: Callable, 
                           *args, **kwargs) -> Any:
    """Safely perform DataFrame operation with memory management."""
    return get_memory_manager().safe_dataframe_operation(df, operation, *args, **kwargs)


@contextmanager
def memory_checkpoint(name: str = "checkpoint"):
    """Context manager for memory checkpointing."""
    with get_memory_manager().memory_checkpoint(name) as checkpoint:
        yield checkpoint


def cleanup_memory(force_gc: bool = True) -> Dict[str, Any]:
    """Clean up memory using the global memory manager."""
    return get_memory_manager().cleanup_memory(force_gc)


def monitor_memory_usage(operation_name: str, operation: Callable, 
                        *args, **kwargs) -> Any:
    """Monitor memory usage during operation execution."""
    return get_memory_manager().monitor_memory_usage(operation_name, operation, *args, **kwargs)


def get_memory_report() -> Dict[str, Any]:
    """Get comprehensive memory usage report."""
    return get_memory_manager().get_memory_report()


# Memory leak detection utilities
class MemoryLeakDetector:
    """Detect potential memory leaks in operations."""
    
    def __init__(self, threshold_mb: float = 50.0):
        """
        Initialize memory leak detector.
        
        Args:
            threshold_mb: Memory increase threshold in MB to flag as potential leak
        """
        self.threshold_mb = threshold_mb
        self.baseline_usage = None
    
    def set_baseline(self):
        """Set baseline memory usage."""
        self.baseline_usage = get_memory_manager().get_memory_usage()
        tprint_info(f"📊 Memory baseline set: {self.baseline_usage['rss_mb']:.1f}MB")
    
    def check_for_leaks(self) -> bool:
        """Check for potential memory leaks."""
        if self.baseline_usage is None:
            tprint_warning("⚠️ No baseline set, cannot detect leaks")
            return False
        
        current_usage = get_memory_manager().get_memory_usage()
        memory_increase = current_usage['rss_mb'] - self.baseline_usage['rss_mb']
        
        if memory_increase > self.threshold_mb:
            tprint_warning(f"🚨 Potential memory leak detected: {memory_increase:.1f}MB increase")
            return True
        
        return False


# Export main utilities
__all__ = [
    'MemoryManager',
    'get_memory_manager',
    'optimize_dataframe_memory',
    'safe_dataframe_operation',
    'memory_checkpoint',
    'cleanup_memory',
    'monitor_memory_usage',
    'get_memory_report',
    'MemoryLeakDetector'
]