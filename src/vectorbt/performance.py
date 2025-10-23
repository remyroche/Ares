"""
VectorBT performance monitoring and optimization.

This module provides performance monitoring, profiling, and optimization
utilities for VectorBT operations in the Ares trading system.
"""

import time
import logging
import psutil
import gc
from typing import Dict, Any, Optional, Callable, List, Tuple
from dataclasses import dataclass, field
from contextlib import contextmanager
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Performance metrics for VectorBT operations."""
    
    operation_name: str
    execution_time: float
    memory_used: float
    memory_peak: float
    cpu_percent: float
    data_size: int
    success: bool
    error_message: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            'operation_name': self.operation_name,
            'execution_time': self.execution_time,
            'memory_used': self.memory_used,
            'memory_peak': self.memory_peak,
            'cpu_percent': self.cpu_percent,
            'data_size': self.data_size,
            'success': self.success,
            'error_message': self.error_message,
            'timestamp': self.timestamp
        }

class VectorBTPerformanceMonitor:
    """Performance monitor for VectorBT operations."""
    
    def __init__(self, max_history: int = 1000):
        """
        Initialize performance monitor.
        
        Args:
            max_history: Maximum number of metrics to keep in history
        """
        self.max_history = max_history
        self.metrics_history: List[PerformanceMetrics] = []
        self.current_operations: Dict[str, Dict[str, Any]] = {}
        
    def start_operation(self, operation_name: str, data_size: int = 0) -> str:
        """
        Start monitoring an operation.
        
        Args:
            operation_name: Name of the operation
            data_size: Size of the data being processed
            
        Returns:
            str: Operation ID for tracking
        """
        operation_id = f"{operation_name}_{int(time.time() * 1000)}"
        
        # Get initial system state
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024**2  # MB
        initial_cpu = process.cpu_percent()
        
        self.current_operations[operation_id] = {
            'operation_name': operation_name,
            'data_size': data_size,
            'start_time': time.time(),
            'initial_memory': initial_memory,
            'initial_cpu': initial_cpu,
            'peak_memory': initial_memory
        }
        
        logger.debug(f"Started monitoring operation: {operation_name} (ID: {operation_id})")
        return operation_id
    
    def end_operation(self, operation_id: str, success: bool = True, error_message: str = None) -> PerformanceMetrics:
        """
        End monitoring an operation and return metrics.
        
        Args:
            operation_id: Operation ID returned by start_operation
            success: Whether the operation succeeded
            error_message: Error message if operation failed
            
        Returns:
            PerformanceMetrics: Performance metrics for the operation
        """
        if operation_id not in self.current_operations:
            raise ValueError(f"Unknown operation ID: {operation_id}")
        
        operation_info = self.current_operations.pop(operation_id)
        
        # Get final system state
        process = psutil.Process()
        final_memory = process.memory_info().rss / 1024**2  # MB
        final_cpu = process.cpu_percent()
        
        # Calculate metrics
        execution_time = time.time() - operation_info['start_time']
        memory_used = final_memory - operation_info['initial_memory']
        memory_peak = operation_info['peak_memory'] - operation_info['initial_memory']
        cpu_percent = (operation_info['initial_cpu'] + final_cpu) / 2
        
        metrics = PerformanceMetrics(
            operation_name=operation_info['operation_name'],
            execution_time=execution_time,
            memory_used=memory_used,
            memory_peak=memory_peak,
            cpu_percent=cpu_percent,
            data_size=operation_info['data_size'],
            success=success,
            error_message=error_message
        )
        
        # Add to history
        self.metrics_history.append(metrics)
        if len(self.metrics_history) > self.max_history:
            self.metrics_history.pop(0)
        
        logger.debug(f"Completed operation: {operation_info['operation_name']} "
                    f"(time: {execution_time:.4f}s, memory: {memory_used:.2f}MB)")
        
        return metrics
    
    def update_peak_memory(self, operation_id: str) -> None:
        """Update peak memory usage for an operation."""
        if operation_id in self.current_operations:
            process = psutil.Process()
            current_memory = process.memory_info().rss / 1024**2  # MB
            operation_info = self.current_operations[operation_id]
            operation_info['peak_memory'] = max(
                operation_info['peak_memory'], 
                current_memory
            )
    
    def get_operation_stats(self, operation_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get statistics for operations.
        
        Args:
            operation_name: Specific operation name to filter by
            
        Returns:
            Dict containing operation statistics
        """
        if operation_name:
            metrics = [m for m in self.metrics_history if m.operation_name == operation_name]
        else:
            metrics = self.metrics_history
        
        if not metrics:
            return {}
        
        execution_times = [m.execution_time for m in metrics]
        memory_used = [m.memory_used for m in metrics]
        success_rate = sum(1 for m in metrics if m.success) / len(metrics)
        
        return {
            'operation_name': operation_name or 'all',
            'total_operations': len(metrics),
            'success_rate': success_rate,
            'avg_execution_time': np.mean(execution_times),
            'min_execution_time': np.min(execution_times),
            'max_execution_time': np.max(execution_times),
            'std_execution_time': np.std(execution_times),
            'avg_memory_used': np.mean(memory_used),
            'max_memory_used': np.max(memory_used),
            'total_memory_used': np.sum(memory_used)
        }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get overall performance summary."""
        if not self.metrics_history:
            return {'message': 'No performance data available'}
        
        all_stats = self.get_operation_stats()
        operation_names = list(set(m.operation_name for m in self.metrics_history))
        operation_stats = {name: self.get_operation_stats(name) for name in operation_names}
        
        return {
            'overall': all_stats,
            'by_operation': operation_stats,
            'total_operations': len(self.metrics_history),
            'monitoring_duration': self.metrics_history[-1].timestamp - self.metrics_history[0].timestamp
        }
    
    def clear_history(self) -> None:
        """Clear performance history."""
        self.metrics_history.clear()
        logger.info("Performance history cleared")

# Global performance monitor instance
_performance_monitor = VectorBTPerformanceMonitor()

def get_performance_monitor() -> VectorBTPerformanceMonitor:
    """Get the global performance monitor instance."""
    return _performance_monitor

@contextmanager
def monitor_operation(operation_name: str, data_size: int = 0):
    """
    Context manager for monitoring VectorBT operations.
    
    Args:
        operation_name: Name of the operation
        data_size: Size of the data being processed
        
    Yields:
        str: Operation ID for tracking
    """
    operation_id = _performance_monitor.start_operation(operation_name, data_size)
    
    try:
        yield operation_id
        _performance_monitor.end_operation(operation_id, success=True)
    except Exception as e:
        _performance_monitor.end_operation(operation_id, success=False, error_message=str(e))
        raise

def profile_operation(func: Callable) -> Callable:
    """
    Decorator to profile VectorBT operations.
    
    Args:
        func: Function to profile
        
    Returns:
        Decorated function with profiling
    """
    def wrapper(*args, **kwargs):
        operation_name = f"{func.__module__}.{func.__name__}"
        
        # Estimate data size from arguments
        data_size = 0
        for arg in args:
            if isinstance(arg, (pd.Series, pd.DataFrame)):
                data_size += arg.size
            elif isinstance(arg, np.ndarray):
                data_size += arg.size
        
        with monitor_operation(operation_name, data_size) as operation_id:
            return func(*args, **kwargs)
    
    return wrapper

class MemoryOptimizer:
    """Memory optimization utilities for VectorBT operations."""
    
    @staticmethod
    def optimize_dataframe_memory(df: pd.DataFrame) -> pd.DataFrame:
        """
        Optimize DataFrame memory usage.
        
        Args:
            df: DataFrame to optimize
            
        Returns:
            Optimized DataFrame
        """
        original_memory = df.memory_usage(deep=True).sum() / 1024**2  # MB
        
        # Optimize numeric columns
        for col in df.select_dtypes(include=[np.number]).columns:
            col_min = df[col].min()
            col_max = df[col].max()
            
            if col_min >= 0:
                if col_max < 255:
                    df[col] = df[col].astype(np.uint8)
                elif col_max < 65535:
                    df[col] = df[col].astype(np.uint16)
                elif col_max < 4294967295:
                    df[col] = df[col].astype(np.uint32)
            else:
                if col_min > np.iinfo(np.int8).min and col_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif col_min > np.iinfo(np.int16).min and col_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif col_min > np.iinfo(np.int32).min and col_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
        
        # Optimize object columns
        for col in df.select_dtypes(include=['object']).columns:
            if df[col].dtype == 'object':
                try:
                    df[col] = df[col].astype('category')
                except (ValueError, TypeError):
                    pass
        
        optimized_memory = df.memory_usage(deep=True).sum() / 1024**2  # MB
        reduction = (original_memory - optimized_memory) / original_memory * 100
        
        logger.debug(f"Memory optimization: {original_memory:.2f}MB -> {optimized_memory:.2f}MB "
                    f"({reduction:.1f}% reduction)")
        
        return df
    
    @staticmethod
    def chunk_processing(data: pd.DataFrame, chunk_size: int, 
                        func: Callable, *args, **kwargs) -> pd.DataFrame:
        """
        Process large DataFrame in chunks to manage memory.
        
        Args:
            data: DataFrame to process
            chunk_size: Size of each chunk
            func: Function to apply to each chunk
            *args: Arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            Processed DataFrame
        """
        results = []
        
        for i in range(0, len(data), chunk_size):
            chunk = data.iloc[i:i + chunk_size]
            
            # Process chunk
            with monitor_operation(f"chunk_processing_{func.__name__}", len(chunk)):
                chunk_result = func(chunk, *args, **kwargs)
                results.append(chunk_result)
            
            # Force garbage collection
            gc.collect()
        
        # Combine results
        if results:
            if isinstance(results[0], pd.DataFrame):
                return pd.concat(results, ignore_index=True)
            elif isinstance(results[0], pd.Series):
                return pd.concat(results, ignore_index=True)
            else:
                return np.concatenate(results)
        else:
            return data

def get_memory_usage() -> Dict[str, float]:
    """
    Get current memory usage information.
    
    Returns:
        Dict containing memory usage statistics
    """
    process = psutil.Process()
    memory_info = process.memory_info()
    virtual_memory = psutil.virtual_memory()
    
    return {
        'process_memory_mb': memory_info.rss / 1024**2,
        'process_memory_percent': process.memory_percent(),
        'system_memory_total_gb': virtual_memory.total / 1024**3,
        'system_memory_available_gb': virtual_memory.available / 1024**3,
        'system_memory_used_percent': virtual_memory.percent
    }

def optimize_vectorbt_performance() -> None:
    """Apply performance optimizations to VectorBT."""
    try:
        import vectorbt as vbt
        from vectorbt.utils.config import configure
        
        # Configure for optimal performance
        configure(
            memory_efficient=True,
            parallel=True,
            validate_data=True,
            raise_on_warning=False
        )
        
        # Set optimal settings
        vbt.settings['memory_efficient'] = True
        vbt.settings['parallel'] = True
        vbt.settings['validate_data'] = True
        vbt.settings['raise_on_warning'] = False
        
        logger.info("VectorBT performance optimizations applied")
        
    except ImportError:
        logger.warning("VectorBT not available for optimization")
    except Exception as e:
        logger.error(f"Failed to optimize VectorBT performance: {e}")

# Auto-optimize on import
optimize_vectorbt_performance()