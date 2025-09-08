from ..standardized_parquet_handler import standardized_parquet_handler
"""Performance Monitoring System for Step 7 Enhanced Matrix Operations.

from src.utils.comprehensive_function_logger import log_step_functions, log_important_calls, log_all_calls, log_internal_call, log_step_progress, log_data_operation

This module provides performance monitoring and resource usage tracking
for all functions and operations.
"""
import time
import gc
from typing import Any, Dict

# Optional dependencies with fallback handling
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None

class PerformanceMonitor:
    """Performance monitoring and resource usage tracking for all functions."""

    def __init__(self, logger):
        self.logger = logger
        self.performance_metrics = {}
        self.resource_usage = {}
        self.start_time = time.time()
        
        # Handle psutil availability
        if PSUTIL_AVAILABLE:
            self.process = psutil.Process()
            self.psutil_available = True
        else:
            self.process = None
            self.psutil_available = False
            self.logger.warning("⚠️ psutil not available - limited performance monitoring")
    
    def start_monitoring(self, function_name: str) -> Dict[str, Any]:
        """Start monitoring performance for a function."""
        if self.psutil_available:
            initial_memory = self.process.memory_info().rss / 1024 / 1024  # MB
            initial_cpu = self.process.cpu_percent()
        else:
            initial_memory = 0.0
            initial_cpu = 0.0
        
        metrics = {
            'function_name': function_name,
            'start_time': time.time(),
            'initial_memory_mb': initial_memory,
            'initial_cpu_percent': initial_cpu,
            'initial_gc_count': gc.get_count(),
            'psutil_available': self.psutil_available
        }
        
        self.performance_metrics[function_name] = metrics
        return metrics
    
    def stop_monitoring(self, function_name: str) -> Dict[str, Any]:
        """Stop monitoring and calculate performance metrics."""
        if function_name not in self.performance_metrics:
            self.logger.warning(f"⚠️ No monitoring data found for {function_name}")
            return {}
        
        metrics = self.performance_metrics[function_name]
        end_time = time.time()
        
        # Calculate performance metrics
        duration = end_time - metrics['start_time']
        
        if self.psutil_available:
            final_memory = self.process.memory_info().rss / 1024 / 1024  # MB
            final_cpu = self.process.cpu_percent()
        else:
            final_memory = 0.0
            final_cpu = 0.0
        
        final_gc_count = gc.get_count()
        
        # Update metrics
        metrics.update({
            'end_time': end_time,
            'duration_seconds': duration,
            'final_memory_mb': final_memory,
            'final_cpu_percent': final_cpu,
            'final_gc_count': final_gc_count,
            'memory_delta_mb': final_memory - metrics['initial_memory_mb'],
            'cpu_delta_percent': final_cpu - metrics['initial_cpu_percent'],
            'gc_delta': tuple(f - i for f, i in zip(final_gc_count, metrics['initial_gc_count']))
        })
        
        # Log performance summary
        self.logger.info(f"📊 Performance metrics for {function_name}:")
        self.logger.info(f"   Duration: {duration:.3f}s")
        if self.psutil_available:
            self.logger.info(f"   Memory delta: {metrics['memory_delta_mb']:.1f} MB")
            self.logger.info(f"   CPU delta: {metrics['cpu_delta_percent']:.1f}%")
        else:
            self.logger.info("   Memory/CPU monitoring: Not available (psutil missing)")
        self.logger.info(f"   GC delta: {metrics['gc_delta']}")
        
        return metrics
    
    def get_system_resources(self) -> Dict[str, Any]:
        """Get current system resource usage."""
        if self.psutil_available:
            return {
                'cpu_percent': psutil.cpu_percent(interval = 1),
                'memory_percent': psutil.virtual_memory().percent,
                'memory_available_gb': psutil.virtual_memory().available / 1024 / 1024 / 1024,
                'disk_usage_percent': psutil.disk_usage('/').percent,
                'process_memory_mb': self.process.memory_info().rss / 1024 / 1024,
                'process_cpu_percent': self.process.cpu_percent(),
                'open_files': len(self.process.open_files()),
                'threads': self.process.num_threads(),
                'psutil_available': True
            }
        else:
            return {
                'cpu_percent': 0.0,
                'memory_percent': 0.0,
                'memory_available_gb': 0.0,
                'disk_usage_percent': 0.0,
                'process_memory_mb': 0.0,
                'process_cpu_percent': 0.0,
                'open_files': 0,
                'threads': 0,
                'psutil_available': False
            }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        total_duration = sum(m.get('duration_seconds', 0) for m in self.performance_metrics.values())
        total_memory_delta = sum(m.get('memory_delta_mb', 0) for m in self.performance_metrics.values())
        
        return {
            'total_functions_monitored': len(self.performance_metrics),
            'total_duration_seconds': total_duration,
            'total_memory_delta_mb': total_memory_delta,
            'average_duration_seconds': total_duration / len(self.performance_metrics) if self.performance_metrics else 0,
            'average_memory_delta_mb': total_memory_delta / len(self.performance_metrics) if self.performance_metrics else 0,
            'session_duration_seconds': time.time() - self.start_time,
            'current_system_resources': self.get_system_resources(),
            'function_metrics': self.performance_metrics
        }

__all__ = ['PerformanceMonitor']