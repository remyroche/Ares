"""
Training Memory Management Utilities.

Provides context managers and utilities for managing memory during ML model training,
with automatic cleanup and monitoring.
"""

import gc
import psutil
import time
from contextlib import contextmanager
from typing import Optional, Dict, Any, Callable
from src.utils.logger import system_logger
from src.utils.tprint import tprint

logger = system_logger.getChild('MemoryManager')


class TrainingMemoryManager:
    """
    Memory manager for ML training with automatic cleanup and monitoring.
    
    Features:
    - Automatic garbage collection at key points
    - Memory usage monitoring and alerts
    - Hardware resource cleanup
    - Memory leak detection
    """
    
    def __init__(
        self,
        alert_threshold_percent: float = 85.0,
        cleanup_interval_seconds: float = 60.0,
        enable_monitoring: bool = True
    ):
        """
        Initialize the memory manager.
        
        Args:
            alert_threshold_percent: Memory usage threshold for alerts (default 85%)
            cleanup_interval_seconds: Interval for automatic cleanup (default 60s)
            enable_monitoring: Enable continuous monitoring (default True)
        """
        self.alert_threshold = alert_threshold_percent
        self.cleanup_interval = cleanup_interval_seconds
        self.enable_monitoring = enable_monitoring
        self.logger = logger
        
        self.initial_memory = None
        self.last_cleanup_time = None
        self.memory_snapshots = []
    
    def get_memory_info(self) -> Dict[str, Any]:
        """
        Get current memory information.
        
        Returns:
            Dictionary with memory statistics
        """
        memory = psutil.virtual_memory()
        return {
            'total_gb': memory.total / (1024 ** 3),
            'available_gb': memory.available / (1024 ** 3),
            'used_gb': memory.used / (1024 ** 3),
            'percent': memory.percent,
            'free_gb': memory.free / (1024 ** 3)
        }
    
    def monitor_memory(self, stage: str, verbose: bool = True) -> float:
        """
        Monitor memory usage at a specific stage.
        
        Args:
            stage: Name of the stage being monitored
            verbose: Whether to print monitoring information
            
        Returns:
            Current memory usage percentage
        """
        memory_info = self.get_memory_info()
        
        # Store snapshot
        self.memory_snapshots.append({
            'stage': stage,
            'timestamp': time.time(),
            'memory_info': memory_info
        })
        
        if verbose:
            tprint(
                f"🧠 [{stage}] Memory: {memory_info['percent']:.1f}% "
                f"({memory_info['used_gb']:.1f}GB / {memory_info['total_gb']:.1f}GB)",
                color="blue"
            )
        
        # Check alert threshold
        if memory_info['percent'] > self.alert_threshold:
            tprint(
                f"⚠️ [{stage}] High memory usage detected: {memory_info['percent']:.1f}% "
                f"(threshold: {self.alert_threshold}%)",
                color="yellow"
            )
            self.logger.warning(
                f"High memory usage at {stage}: {memory_info['percent']:.1f}%"
            )
        
        return memory_info['percent']
    
    def cleanup_memory(self, force: bool = False, verbose: bool = True) -> Dict[str, Any]:
        """
        Perform memory cleanup via garbage collection.
        
        Args:
            force: Force cleanup regardless of interval
            verbose: Whether to print cleanup information
            
        Returns:
            Dictionary with cleanup statistics
        """
        current_time = time.time()
        
        # Check cleanup interval unless forced
        if not force and self.last_cleanup_time is not None:
            time_since_cleanup = current_time - self.last_cleanup_time
            if time_since_cleanup < self.cleanup_interval:
                return {'skipped': True, 'reason': 'interval_not_reached'}
        
        # Get memory before cleanup
        before = self.get_memory_info()
        
        # Perform garbage collection
        collected = gc.collect()
        
        # Get memory after cleanup
        after = self.get_memory_info()
        
        # Calculate freed memory
        freed_gb = before['used_gb'] - after['used_gb']
        
        self.last_cleanup_time = current_time
        
        if verbose:
            tprint(
                f"🧹 Memory cleanup: freed {freed_gb:.2f}GB, "
                f"collected {collected} objects",
                color="green"
            )
        
        return {
            'skipped': False,
            'collected_objects': collected,
            'freed_gb': freed_gb,
            'before_percent': before['percent'],
            'after_percent': after['percent']
        }
    
    def start_monitoring(self, stage: str = "Initial"):
        """Start memory monitoring session."""
        self.initial_memory = self.get_memory_info()
        self.memory_snapshots = []
        self.last_cleanup_time = time.time()
        
        tprint(
            f"🔍 [{stage}] Memory monitoring started: "
            f"{self.initial_memory['percent']:.1f}% "
            f"({self.initial_memory['used_gb']:.1f}GB / {self.initial_memory['total_gb']:.1f}GB)",
            color="cyan"
        )
    
    def end_monitoring(self, stage: str = "Final") -> Dict[str, Any]:
        """
        End memory monitoring session and return statistics.
        
        Returns:
            Dictionary with monitoring statistics
        """
        final_memory = self.get_memory_info()
        
        if self.initial_memory is None:
            return {'error': 'Monitoring was not started'}
        
        # Calculate changes
        memory_change_gb = final_memory['used_gb'] - self.initial_memory['used_gb']
        percent_change = final_memory['percent'] - self.initial_memory['percent']
        
        tprint(
            f"📊 [{stage}] Memory monitoring ended: "
            f"{final_memory['percent']:.1f}% "
            f"(change: {memory_change_gb:+.1f}GB, {percent_change:+.1f}%)",
            color="blue"
        )
        
        # Check for memory leak
        if memory_change_gb > 1.0:  # More than 1GB increase
            tprint(
                f"⚠️ Potential memory leak detected: {memory_change_gb:.1f}GB increase",
                color="yellow"
            )
            self.logger.warning(f"Potential memory leak: {memory_change_gb:.1f}GB increase")
        
        return {
            'initial_memory': self.initial_memory,
            'final_memory': final_memory,
            'memory_change_gb': memory_change_gb,
            'percent_change': percent_change,
            'snapshots': self.memory_snapshots
        }
    
    def get_memory_report(self) -> str:
        """
        Generate a comprehensive memory usage report.
        
        Returns:
            Formatted memory report string
        """
        if not self.memory_snapshots:
            return "No memory snapshots available"
        
        report_lines = [
            "=" * 60,
            "MEMORY USAGE REPORT",
            "=" * 60,
            ""
        ]
        
        for snapshot in self.memory_snapshots:
            stage = snapshot['stage']
            mem = snapshot['memory_info']
            report_lines.append(
                f"{stage:30s}: {mem['percent']:5.1f}% ({mem['used_gb']:6.1f}GB used)"
            )
        
        if self.initial_memory:
            final = self.memory_snapshots[-1]['memory_info']
            change = final['used_gb'] - self.initial_memory['used_gb']
            report_lines.extend([
                "",
                f"{'Total Memory Change':30s}: {change:+6.1f}GB"
            ])
        
        report_lines.append("=" * 60)
        
        return "\n".join(report_lines)


@contextmanager
def managed_training(
    stage_name: str = "Training",
    auto_cleanup: bool = True,
    cleanup_on_error: bool = True,
    alert_threshold: float = 85.0,
    hardware_manager: Optional[Any] = None
):
    """
    Context manager for automatic memory management during training.
    
    Usage:
        with managed_training("Model Training", hardware_manager=hw_mgr):
            # Training code here
            model.fit(X, y)
        # Automatic cleanup happens here
    
    Args:
        stage_name: Name of the training stage
        auto_cleanup: Automatically cleanup on exit
        cleanup_on_error: Cleanup even if error occurs
        alert_threshold: Memory alert threshold percentage
        hardware_manager: Optional hardware manager to cleanup
        
    Yields:
        TrainingMemoryManager instance for manual control
    """
    memory_mgr = TrainingMemoryManager(
        alert_threshold_percent=alert_threshold,
        enable_monitoring=True
    )
    
    # Start monitoring
    memory_mgr.start_monitoring(stage_name)
    
    try:
        # Yield the manager for manual operations
        yield memory_mgr
        
    except Exception as e:
        # Handle errors
        tprint(f"❌ [{stage_name}] Error occurred: {e}", color="red")
        
        if cleanup_on_error:
            tprint(f"🧹 [{stage_name}] Performing cleanup after error", color="yellow")
            memory_mgr.cleanup_memory(force=True)
        
        raise
        
    finally:
        # Always end monitoring
        memory_mgr.end_monitoring(stage_name)
        
        # Automatic cleanup
        if auto_cleanup:
            memory_mgr.cleanup_memory(force=True)
        
        # Hardware manager cleanup
        if hardware_manager is not None:
            try:
                if hasattr(hardware_manager, 'cleanup'):
                    hardware_manager.cleanup()
                    tprint(f"✅ [{stage_name}] Hardware resources cleaned up", color="green")
            except Exception as e:
                tprint(f"⚠️ [{stage_name}] Hardware cleanup warning: {e}", color="yellow")


@contextmanager
def periodic_cleanup(interval_seconds: float = 60.0, verbose: bool = False):
    """
    Context manager for periodic memory cleanup during long-running operations.
    
    Usage:
        with periodic_cleanup(interval_seconds=30):
            for epoch in range(100):
                # Long training loop
                train_epoch(epoch)
    
    Args:
        interval_seconds: Cleanup interval in seconds
        verbose: Print cleanup messages
        
    Yields:
        Cleanup function that can be called manually
    """
    memory_mgr = TrainingMemoryManager(
        cleanup_interval_seconds=interval_seconds,
        enable_monitoring=True
    )
    
    last_cleanup = time.time()
    
    def maybe_cleanup(force: bool = False):
        """Cleanup if interval has passed or forced."""
        nonlocal last_cleanup
        current_time = time.time()
        
        if force or (current_time - last_cleanup) >= interval_seconds:
            result = memory_mgr.cleanup_memory(force=True, verbose=verbose)
            last_cleanup = current_time
            return result
        return None
    
    try:
        yield maybe_cleanup
    finally:
        # Final cleanup
        memory_mgr.cleanup_memory(force=True, verbose=True)


def monitor_function_memory(func: Callable) -> Callable:
    """
    Decorator to monitor memory usage of a function.
    
    Usage:
        @monitor_function_memory
        def train_model(X, y):
            # Training code
            pass
    
    Args:
        func: Function to wrap
        
    Returns:
        Wrapped function with memory monitoring
    """
    def wrapper(*args, **kwargs):
        func_name = func.__name__
        
        with managed_training(stage_name=func_name) as memory_mgr:
            result = func(*args, **kwargs)
            
            # Print memory report
            report = memory_mgr.get_memory_report()
            tprint(f"\n{report}", color="blue")
            
            return result
    
    return wrapper

