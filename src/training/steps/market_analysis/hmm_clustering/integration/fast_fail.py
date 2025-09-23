"""
Fast Fail Mechanism for HMM Clustering Pipeline

This module provides fast fail mechanisms for clustering operations including:
- Timeout management
- Memory usage monitoring
- Quality validation
- Resource exhaustion detection
"""

import time
import signal
import psutil
import logging
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass
from datetime import datetime
import threading
from contextlib import contextmanager

try:
    from src.utils.hardware import (
        get_hardware_accelerator,
        get_memory_manager,
        get_performance_monitor
    )
    HARDWARE_ACCELERATION_AVAILABLE = True
except ImportError:
    HARDWARE_ACCELERATION_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class FastFailConfig:
    """Configuration for fast fail mechanisms."""
    timeout_seconds: float = 300.0
    memory_limit_gb: float = 8.0
    cpu_limit_percent: float = 95.0
    quality_threshold: float = 0.3
    max_iterations: int = 100
    enable_timeout: bool = True
    enable_memory_check: bool = True
    enable_cpu_check: bool = True
    enable_quality_check: bool = True
    enable_iteration_limit: bool = True


@dataclass
class FastFailResult:
    """Result of fast fail check."""
    should_fail: bool = False
    fail_reason: str = ""
    resource_usage: Dict[str, Any] = None
    timestamp: str = None


class FastFailManager:
    """Manages fast fail mechanisms for clustering operations."""

    def __init__(self, config: Optional[FastFailConfig] = None):
        """Initialize the fast fail manager.

        Args:
            config: Fast fail configuration
        """
        self.config = config or FastFailConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize hardware monitoring if available
        self.memory_manager = None
        self.performance_monitor = None
        
        if HARDWARE_ACCELERATION_AVAILABLE:
            try:
                self.memory_manager = get_memory_manager()
                self.performance_monitor = get_performance_monitor()
                self.logger.info("✅ Hardware monitoring initialized for fast fail")
            except Exception as e:
                self.logger.warning(f"⚠️ Hardware monitoring not available for fast fail: {e}")
        
        # Track operation start times
        self.operation_start_times = {}
        
        # Track iteration counts
        self.iteration_counts = {}

    def start_operation(self, operation_id: str) -> None:
        """Start tracking an operation.

        Args:
            operation_id: Unique identifier for the operation
        """
        self.operation_start_times[operation_id] = time.time()
        self.iteration_counts[operation_id] = 0
        
        self.logger.debug(f"🚀 Started tracking operation: {operation_id}")

    def check_fast_fail(self, operation_id: str, iteration_count: Optional[int] = None) -> FastFailResult:
        """Check if operation should fail fast.

        Args:
            operation_id: Unique identifier for the operation
            iteration_count: Current iteration count (optional)

        Returns:
            FastFailResult indicating if operation should fail
        """
        try:
            # Update iteration count if provided
            if iteration_count is not None:
                self.iteration_counts[operation_id] = iteration_count
            
            # Check timeout
            if self.config.enable_timeout:
                timeout_result = self._check_timeout(operation_id)
                if timeout_result.should_fail:
                    return timeout_result
            
            # Check memory usage
            if self.config.enable_memory_check:
                memory_result = self._check_memory_usage()
                if memory_result.should_fail:
                    return memory_result
            
            # Check CPU usage
            if self.config.enable_cpu_check:
                cpu_result = self._check_cpu_usage()
                if cpu_result.should_fail:
                    return cpu_result
            
            # Check iteration limit
            if self.config.enable_iteration_limit:
                iteration_result = self._check_iteration_limit(operation_id)
                if iteration_result.should_fail:
                    return iteration_result
            
            # All checks passed
            return FastFailResult(
                should_fail=False,
                fail_reason="",
                resource_usage=self._get_current_resource_usage(),
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            self.logger.error(f"❌ Fast fail check failed: {e}")
            return FastFailResult(
                should_fail=True,
                fail_reason=f"Fast fail check error: {str(e)}",
                resource_usage={},
                timestamp=datetime.now().isoformat()
            )

    def _check_timeout(self, operation_id: str) -> FastFailResult:
        """Check if operation has exceeded timeout.

        Args:
            operation_id: Operation identifier

        Returns:
            FastFailResult for timeout check
        """
        try:
            if operation_id not in self.operation_start_times:
                return FastFailResult(should_fail=False)
            
            start_time = self.operation_start_times[operation_id]
            elapsed_time = time.time() - start_time
            
            if elapsed_time > self.config.timeout_seconds:
                return FastFailResult(
                    should_fail=True,
                    fail_reason=f"Operation timeout: {elapsed_time:.1f}s > {self.config.timeout_seconds}s",
                    resource_usage=self._get_current_resource_usage(),
                    timestamp=datetime.now().isoformat()
                )
            
            return FastFailResult(should_fail=False)
            
        except Exception as e:
            self.logger.warning(f"⚠️ Timeout check failed: {e}")
            return FastFailResult(should_fail=False)

    def _check_memory_usage(self) -> FastFailResult:
        """Check memory usage against limits.

        Returns:
            FastFailResult for memory check
        """
        try:
            # Get memory information
            memory_info = psutil.virtual_memory()
            memory_used_gb = memory_info.used / (1024**3)
            memory_percent = memory_info.percent
            
            # Check against limit
            if memory_used_gb > self.config.memory_limit_gb:
                return FastFailResult(
                    should_fail=True,
                    fail_reason=f"Memory usage exceeded: {memory_used_gb:.1f}GB > {self.config.memory_limit_gb}GB",
                    resource_usage={
                        'memory_used_gb': memory_used_gb,
                        'memory_percent': memory_percent,
                        'memory_limit_gb': self.config.memory_limit_gb
                    },
                    timestamp=datetime.now().isoformat()
                )
            
            return FastFailResult(should_fail=False)
            
        except Exception as e:
            self.logger.warning(f"⚠️ Memory check failed: {e}")
            return FastFailResult(should_fail=False)

    def _check_cpu_usage(self) -> FastFailResult:
        """Check CPU usage against limits.

        Returns:
            FastFailResult for CPU check
        """
        try:
            # Get CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Check against limit
            if cpu_percent > self.config.cpu_limit_percent:
                return FastFailResult(
                    should_fail=True,
                    fail_reason=f"CPU usage exceeded: {cpu_percent:.1f}% > {self.config.cpu_limit_percent}%",
                    resource_usage={
                        'cpu_percent': cpu_percent,
                        'cpu_limit_percent': self.config.cpu_limit_percent
                    },
                    timestamp=datetime.now().isoformat()
                )
            
            return FastFailResult(should_fail=False)
            
        except Exception as e:
            self.logger.warning(f"⚠️ CPU check failed: {e}")
            return FastFailResult(should_fail=False)

    def _check_iteration_limit(self, operation_id: str) -> FastFailResult:
        """Check iteration count against limits.

        Args:
            operation_id: Operation identifier

        Returns:
            FastFailResult for iteration check
        """
        try:
            if operation_id not in self.iteration_counts:
                return FastFailResult(should_fail=False)
            
            iteration_count = self.iteration_counts[operation_id]
            
            if iteration_count > self.config.max_iterations:
                return FastFailResult(
                    should_fail=True,
                    fail_reason=f"Iteration limit exceeded: {iteration_count} > {self.config.max_iterations}",
                    resource_usage=self._get_current_resource_usage(),
                    timestamp=datetime.now().isoformat()
                )
            
            return FastFailResult(should_fail=False)
            
        except Exception as e:
            self.logger.warning(f"⚠️ Iteration limit check failed: {e}")
            return FastFailResult(should_fail=False)

    def _get_current_resource_usage(self) -> Dict[str, Any]:
        """Get current resource usage information.

        Returns:
            Dictionary of resource usage
        """
        try:
            # Get system information
            memory_info = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=0.1)
            disk_info = psutil.disk_usage('/')
            
            resource_usage = {
                'memory_used_gb': memory_info.used / (1024**3),
                'memory_percent': memory_info.percent,
                'memory_available_gb': memory_info.available / (1024**3),
                'cpu_percent': cpu_percent,
                'disk_used_gb': disk_info.used / (1024**3),
                'disk_percent': (disk_info.used / disk_info.total) * 100
            }
            
            # Add hardware monitoring data if available
            if self.memory_manager:
                try:
                    hardware_memory = self.memory_manager.get_memory_usage()
                    resource_usage.update(hardware_memory)
                except Exception:
                    pass
            
            if self.performance_monitor:
                try:
                    performance_data = self.performance_monitor.get_current_metrics()
                    resource_usage.update(performance_data)
                except Exception:
                    pass
            
            return resource_usage
            
        except Exception as e:
            self.logger.warning(f"⚠️ Resource usage check failed: {e}")
            return {'error': str(e)}

    def validate_quality(self, quality_metrics: Dict[str, Any]) -> FastFailResult:
        """Validate clustering quality against thresholds.

        Args:
            quality_metrics: Quality metrics dictionary

        Returns:
            FastFailResult for quality validation
        """
        try:
            if not self.config.enable_quality_check:
                return FastFailResult(should_fail=False)
            
            # Check silhouette score
            silhouette = quality_metrics.get('silhouette', 0.0)
            if silhouette < self.config.quality_threshold:
                return FastFailResult(
                    should_fail=True,
                    fail_reason=f"Quality threshold not met: silhouette {silhouette:.3f} < {self.config.quality_threshold}",
                    resource_usage=self._get_current_resource_usage(),
                    timestamp=datetime.now().isoformat()
                )
            
            # Check cluster count
            n_clusters = quality_metrics.get('n_clusters', 0)
            if n_clusters < 2:
                return FastFailResult(
                    should_fail=True,
                    fail_reason=f"Insufficient clusters: {n_clusters} < 2",
                    resource_usage=self._get_current_resource_usage(),
                    timestamp=datetime.now().isoformat()
                )
            
            if n_clusters > 25:
                return FastFailResult(
                    should_fail=True,
                    fail_reason=f"Too many clusters: {n_clusters} > 25",
                    resource_usage=self._get_current_resource_usage(),
                    timestamp=datetime.now().isoformat()
                )
            
            return FastFailResult(should_fail=False)
            
        except Exception as e:
            self.logger.warning(f"⚠️ Quality validation failed: {e}")
            return FastFailResult(should_fail=False)

    def end_operation(self, operation_id: str) -> Dict[str, Any]:
        """End tracking an operation and return summary.

        Args:
            operation_id: Operation identifier

        Returns:
            Operation summary
        """
        try:
            if operation_id not in self.operation_start_times:
                return {'error': f'Operation {operation_id} not found'}
            
            start_time = self.operation_start_times[operation_id]
            end_time = time.time()
            execution_time = end_time - start_time
            
            summary = {
                'operation_id': operation_id,
                'execution_time': execution_time,
                'iterations': self.iteration_counts.get(operation_id, 0),
                'resource_usage': self._get_current_resource_usage(),
                'timestamp': datetime.now().isoformat()
            }
            
            # Clean up tracking data
            del self.operation_start_times[operation_id]
            if operation_id in self.iteration_counts:
                del self.iteration_counts[operation_id]
            
            self.logger.debug(f"✅ Ended tracking operation: {operation_id} (execution time: {execution_time:.2f}s)")
            return summary
            
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to end operation tracking: {e}")
            return {'error': str(e)}

    @contextmanager
    def operation_context(self, operation_id: str):
        """Context manager for operation tracking.

        Args:
            operation_id: Unique identifier for the operation

        Yields:
            FastFailManager instance for the operation
        """
        try:
            self.start_operation(operation_id)
            yield self
        finally:
            self.end_operation(operation_id)

    def create_timeout_handler(self, operation_id: str) -> Callable:
        """Create a timeout handler for an operation.

        Args:
            operation_id: Operation identifier

        Returns:
            Timeout handler function
        """
        def timeout_handler(signum, frame):
            self.logger.error(f"❌ Operation {operation_id} timed out after {self.config.timeout_seconds}s")
            raise TimeoutError(f"Operation {operation_id} timed out")
        
        return timeout_handler

    def set_timeout(self, operation_id: str) -> None:
        """Set timeout for an operation.

        Args:
            operation_id: Operation identifier
        """
        if self.config.enable_timeout:
            handler = self.create_timeout_handler(operation_id)
            signal.signal(signal.SIGALRM, handler)
            signal.alarm(int(self.config.timeout_seconds))

    def clear_timeout(self) -> None:
        """Clear timeout."""
        signal.alarm(0)


def create_fast_fail_manager(config: Optional[FastFailConfig] = None) -> FastFailManager:
    """Create a fast fail manager instance.

    Args:
        config: Fast fail configuration

    Returns:
        FastFailManager instance
    """
    return FastFailManager(config)