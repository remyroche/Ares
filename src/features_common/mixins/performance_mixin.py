"""
Performance monitoring and optimization mixin.

This mixin provides comprehensive performance monitoring, profiling,
and optimization capabilities for all features_common components.
"""

import time
import psutil
import logging
from typing import Dict, Any, Optional, List, Callable, Union
import pandas as pd
import numpy as np
from contextlib import contextmanager
from functools import wraps

from ..config import get_unified_config

logger = logging.getLogger(__name__)

class PerformanceMixin:
    """
    Mixin class providing performance monitoring and optimization.
    
    This mixin can be added to any class to provide performance tracking,
    memory monitoring, and execution profiling capabilities.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize performance mixin."""
        super().__init__(*args, **kwargs)
        
        # Get unified configuration
        self.config = get_unified_config()
        
        # Performance tracking
        self._performance_stats = {
            'total_operations': 0,
            'total_execution_time': 0.0,
            'memory_usage': [],
            'execution_times': [],
            'peak_memory_usage': 0,
            'current_memory_usage': 0,
            'gpu_operations': 0,
            'cpu_operations': 0,
            'optimized_operations': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        # Profiling data
        self._profiling_data = []
        self._active_profiles = {}
        
        # Memory tracking
        self._memory_snapshots = []
        self._last_memory_check = 0
        self._memory_check_interval = 1.0  # Check every second
        
        # Performance thresholds
        self._performance_thresholds = {
            'slow_operation': 1.0,  # 1 second
            'memory_warning': 0.8,  # 80% of available memory
            'cpu_warning': 0.9,    # 90% CPU usage
        }
    
    def enable_performance_monitoring(self) -> None:
        """Enable performance monitoring."""
        self.config.optimization.enable_performance_monitoring = True
        logger.debug("Performance monitoring enabled")
    
    def disable_performance_monitoring(self) -> None:
        """Disable performance monitoring."""
        self.config.optimization.enable_performance_monitoring = False
        logger.debug("Performance monitoring disabled")
    
    def is_performance_monitoring_enabled(self) -> bool:
        """Check if performance monitoring is enabled."""
        return self.config.optimization.enable_performance_monitoring
    
    @contextmanager
    def profile_operation(self, operation_name: str, 
                         track_memory: bool = True,
                         track_cpu: bool = True):
        """
        Context manager for profiling operations.
        
        Args:
            operation_name: Name of the operation being profiled
            track_memory: Whether to track memory usage
            track_cpu: Whether to track CPU usage
        """
        if not self.is_performance_monitoring_enabled():
            yield
            return
        
        start_time = time.time()
        start_memory = self._get_memory_usage() if track_memory else 0
        start_cpu = psutil.cpu_percent() if track_cpu else 0
        
        profile_id = f"{operation_name}_{int(start_time * 1000)}"
        self._active_profiles[profile_id] = {
            'operation_name': operation_name,
            'start_time': start_time,
            'start_memory': start_memory,
            'start_cpu': start_cpu,
            'track_memory': track_memory,
            'track_cpu': track_cpu
        }
        
        try:
            yield profile_id
        finally:
            end_time = time.time()
            end_memory = self._get_memory_usage() if track_memory else start_memory
            end_cpu = psutil.cpu_percent() if track_cpu else start_cpu
            
            execution_time = end_time - start_time
            memory_delta = end_memory - start_memory
            cpu_delta = end_cpu - start_cpu
            
            # Record profiling data
            profile_data = {
                'profile_id': profile_id,
                'operation_name': operation_name,
                'execution_time': execution_time,
                'memory_delta': memory_delta,
                'cpu_delta': cpu_delta,
                'start_memory': start_memory,
                'end_memory': end_memory,
                'start_cpu': start_cpu,
                'end_cpu': end_cpu,
                'timestamp': start_time
            }
            
            self._profiling_data.append(profile_data)
            self._update_performance_stats(profile_data)
            
            # Remove from active profiles
            if profile_id in self._active_profiles:
                del self._active_profiles[profile_id]
            
            # Check for performance issues
            self._check_performance_thresholds(profile_data)
    
    def profile_method(self, track_memory: bool = True, track_cpu: bool = True):
        """
        Decorator for profiling methods.
        
        Args:
            track_memory: Whether to track memory usage
            track_cpu: Whether to track CPU usage
        """
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                operation_name = f"{func.__name__}"
                with self.profile_operation(operation_name, track_memory, track_cpu):
                    return func(*args, **kwargs)
            return wrapper
        return decorator
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # Convert to MB
        except Exception:
            return 0.0
    
    def _update_performance_stats(self, profile_data: Dict[str, Any]) -> None:
        """Update performance statistics with new profile data."""
        self._performance_stats['total_operations'] += 1
        self._performance_stats['total_execution_time'] += profile_data['execution_time']
        self._performance_stats['execution_times'].append(profile_data['execution_time'])
        
        # Update memory tracking
        if profile_data['memory_delta'] != 0:
            self._performance_stats['memory_usage'].append(profile_data['end_memory'])
            self._performance_stats['current_memory_usage'] = profile_data['end_memory']
            self._performance_stats['peak_memory_usage'] = max(
                self._performance_stats['peak_memory_usage'],
                profile_data['end_memory']
            )
        
        # Keep only recent execution times
        if len(self._performance_stats['execution_times']) > 1000:
            self._performance_stats['execution_times'] = self._performance_stats['execution_times'][-500:]
        
        # Keep only recent memory usage
        if len(self._performance_stats['memory_usage']) > 1000:
            self._performance_stats['memory_usage'] = self._performance_stats['memory_usage'][-500:]
    
    def _check_performance_thresholds(self, profile_data: Dict[str, Any]) -> None:
        """Check if performance thresholds are exceeded."""
        execution_time = profile_data['execution_time']
        memory_usage = profile_data['end_memory']
        
        # Check for slow operations
        if execution_time > self._performance_thresholds['slow_operation']:
            logger.warning(
                f"Slow operation detected: {profile_data['operation_name']} "
                f"took {execution_time:.3f}s"
            )
        
        # Check for high memory usage
        if memory_usage > self._performance_thresholds['memory_warning'] * psutil.virtual_memory().total / 1024 / 1024:
            logger.warning(
                f"High memory usage detected: {profile_data['operation_name']} "
                f"used {memory_usage:.1f}MB"
            )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        stats = self._performance_stats.copy()
        
        # Calculate derived metrics
        if stats['total_operations'] > 0:
            stats['avg_execution_time'] = stats['total_execution_time'] / stats['total_operations']
        else:
            stats['avg_execution_time'] = 0.0
        
        # Calculate execution time statistics
        if stats['execution_times']:
            times = np.array(stats['execution_times'])
            stats['min_execution_time'] = float(np.min(times))
            stats['max_execution_time'] = float(np.max(times))
            stats['std_execution_time'] = float(np.std(times))
            stats['median_execution_time'] = float(np.median(times))
        else:
            stats['min_execution_time'] = 0.0
            stats['max_execution_time'] = 0.0
            stats['std_execution_time'] = 0.0
            stats['median_execution_time'] = 0.0
        
        # Calculate memory statistics
        if stats['memory_usage']:
            memory_usage = np.array(stats['memory_usage'])
            stats['avg_memory_usage'] = float(np.mean(memory_usage))
            stats['std_memory_usage'] = float(np.std(memory_usage))
            stats['min_memory_usage'] = float(np.min(memory_usage))
            stats['max_memory_usage'] = float(np.max(memory_usage))
        else:
            stats['avg_memory_usage'] = 0.0
            stats['std_memory_usage'] = 0.0
            stats['min_memory_usage'] = 0.0
            stats['max_memory_usage'] = 0.0
        
        # Calculate cache statistics
        total_cache_operations = stats['cache_hits'] + stats['cache_misses']
        if total_cache_operations > 0:
            stats['cache_hit_rate'] = stats['cache_hits'] / total_cache_operations
        else:
            stats['cache_hit_rate'] = 0.0
        
        # Add system information
        stats['system_memory_total'] = psutil.virtual_memory().total / 1024 / 1024  # MB
        stats['system_memory_available'] = psutil.virtual_memory().available / 1024 / 1024  # MB
        stats['system_cpu_count'] = psutil.cpu_count()
        
        return stats
    
    def get_profiling_summary(self, operation_name: Optional[str] = None) -> Dict[str, Any]:
        """Get profiling summary for specific operation or all operations."""
        if operation_name:
            filtered_data = [d for d in self._profiling_data if d['operation_name'] == operation_name]
        else:
            filtered_data = self._profiling_data
        
        if not filtered_data:
            return {'message': 'No profiling data available'}
        
        # Calculate statistics
        execution_times = [d['execution_time'] for d in filtered_data]
        memory_deltas = [d['memory_delta'] for d in filtered_data]
        
        summary = {
            'operation_name': operation_name or 'all_operations',
            'total_operations': len(filtered_data),
            'total_execution_time': sum(execution_times),
            'avg_execution_time': np.mean(execution_times),
            'min_execution_time': np.min(execution_times),
            'max_execution_time': np.max(execution_times),
            'std_execution_time': np.std(execution_times),
            'total_memory_delta': sum(memory_deltas),
            'avg_memory_delta': np.mean(memory_deltas),
            'min_memory_delta': np.min(memory_deltas),
            'max_memory_delta': np.max(memory_deltas),
            'std_memory_delta': np.std(memory_deltas)
        }
        
        return summary
    
    def get_memory_usage_history(self) -> List[Dict[str, Any]]:
        """Get memory usage history."""
        return [
            {
                'timestamp': d['timestamp'],
                'memory_usage': d['end_memory'],
                'operation_name': d['operation_name']
            }
            for d in self._profiling_data
        ]
    
    def get_slow_operations(self, threshold: Optional[float] = None) -> List[Dict[str, Any]]:
        """Get operations that exceeded the performance threshold."""
        if threshold is None:
            threshold = self._performance_thresholds['slow_operation']
        
        return [
            d for d in self._profiling_data
            if d['execution_time'] > threshold
        ]
    
    def reset_performance_stats(self) -> None:
        """Reset all performance statistics."""
        self._performance_stats = {
            'total_operations': 0,
            'total_execution_time': 0.0,
            'memory_usage': [],
            'execution_times': [],
            'peak_memory_usage': 0,
            'current_memory_usage': 0,
            'gpu_operations': 0,
            'cpu_operations': 0,
            'optimized_operations': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        self._profiling_data = []
        self._active_profiles = {}
        self._memory_snapshots = []
    
    def set_performance_thresholds(self, **thresholds) -> None:
        """Set performance thresholds."""
        for key, value in thresholds.items():
            if key in self._performance_thresholds:
                self._performance_thresholds[key] = value
            else:
                logger.warning(f"Unknown performance threshold: {key}")
    
    def get_performance_recommendations(self) -> List[str]:
        """Get performance optimization recommendations."""
        recommendations = []
        stats = self.get_performance_stats()
        
        # Check execution time
        if stats['avg_execution_time'] > 0.5:  # 500ms
            recommendations.append("Consider using VectorBT optimization for faster execution")
        
        # Check memory usage
        if stats['peak_memory_usage'] > 1000:  # 1GB
            recommendations.append("Consider enabling memory optimization to reduce memory usage")
        
        # Check cache hit rate
        if stats['cache_hit_rate'] < 0.5:
            recommendations.append("Consider increasing cache size or improving cache strategy")
        
        # Check execution time variance
        if stats['std_execution_time'] > stats['avg_execution_time'] * 0.5:
            recommendations.append("High execution time variance detected - consider profiling for bottlenecks")
        
        return recommendations