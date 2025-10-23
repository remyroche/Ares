"""
Memory Manager for Feature Matrices and Intermediate States.

This module provides memory management capabilities for tracking memory footprint
of feature matrices and intermediate states, streaming large datasets in batch mode,
and preventing OOM crashes.
"""

import logging
import gc
import threading
import time
import weakref
from typing import Any, Dict, List, Optional, Tuple, Set, Union, Callable
from contextlib import contextmanager
import psutil
import os

# Import from hardware optimization modules
try:
    from utils.hardware.m1_memory_optimizer import (
        get_m1_memory_optimizer, M1MemoryOptimizer,
        optimize_dataframe_memory, start_m1_memory_monitoring,
        stop_m1_memory_monitoring, get_memory_usage
    )
    M1_MEMORY_OPTIMIZATION_AVAILABLE = True
except ImportError:
    M1_MEMORY_OPTIMIZATION_AVAILABLE = False
    M1MemoryOptimizer = None

try:
    from utils.matrix_operations.vectorized_core import get_vectorized_processing_core
    VECTORIZED_CORE_AVAILABLE = True
except ImportError:
    VECTORIZED_CORE_AVAILABLE = False
    get_vectorized_processing_core = None

from src.utils.tprint import (
    tprint, tprint_debug, tprint_info, tprint_warning, tprint_error,
    tprint_success, tprint_progress, tprint_performance, tprint_structured,
    tprint_timer, tprint_logged, LogLevel, TimestampFormat
)

logger = logging.getLogger(__name__)

class MemoryManager:
    """Track memory footprint of feature matrices and intermediate states."""

    def __init__(self, memory_limit_gb: Optional[float] = None,
                 enable_monitoring: bool = True,
                 aggressive_cleanup: bool = False):
        """Initialize Memory Manager.

        Args:
            memory_limit_gb: Memory limit in GB (None for system default)
            enable_monitoring: Whether to enable continuous monitoring
            aggressive_cleanup: Enable aggressive memory cleanup strategies
        """
        self.logger = logger.getChild('MemoryManager')
        tprint(f"🚀 Memory Manager initialized (limit: {memory_limit_gb}GB, monitoring: {enable_monitoring})", "INFO")

        # Initialize M1 memory optimizer if available
        if M1_MEMORY_OPTIMIZATION_AVAILABLE:
            self.m1_memory_optimizer = get_m1_memory_optimizer(memory_limit_gb)
        else:
            self.m1_memory_optimizer = None

        # Configuration
        self.memory_limit_gb = memory_limit_gb
        self.enable_monitoring = enable_monitoring
        self.aggressive_cleanup = aggressive_cleanup

        # Memory tracking
        self.memory_snapshots = []
        self.feature_matrices = {}  # Track feature matrix memory usage
        self.intermediate_states = {}  # Track intermediate computation states
        self.memory_alerts = []

        # Batch processing configuration
        self.batch_size = 1000  # Default batch size for streaming
        self.chunk_size_mb = 100  # Default chunk size in MB

        # Cleanup strategies
        self.cleanup_thresholds = {
            'warning': 0.7,    # 70% memory usage
            'critical': 0.85,  # 85% memory usage
            'emergency': 0.95  # 95% memory usage
        }

        # Monitoring
        self.monitoring_active = False
        self.monitoring_thread = None

        # Weak references for automatic cleanup
        self.tracked_objects = weakref.WeakSet()

        # Start monitoring if enabled
        if self.enable_monitoring:
            self.start_monitoring()

        self.logger.info(f"Memory Manager initialized (limit: {memory_limit_gb}GB, monitoring: {enable_monitoring})")

    def start_monitoring(self):
        """Start continuous memory monitoring."""
        if self.monitoring_active:
            return

        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True,
            name="MemoryManager-Monitor"
        )
        self.monitoring_thread.start()
        self.logger.info("Memory monitoring started")

    def stop_monitoring(self):
        """Stop memory monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=2.0)
        self.logger.info("Memory monitoring stopped")

    def _monitoring_loop(self):
        """Main memory monitoring loop."""
        while self.monitoring_active:
            try:
                self._check_memory_pressure()
                self._apply_cleanup_strategies()
                time.sleep(3.0)  # Check every 3 seconds
            except Exception as e:
                self.logger.error(f"Memory monitoring error: {e}")
                time.sleep(5.0)

    def _check_memory_pressure(self):
        """Check current memory pressure and take action."""
        tprint("🔍 Checking memory pressure...", "DEBUG")
        try:
            memory_stats = self.get_memory_stats()
            memory_percent = memory_stats.get('memory_percent', 0)

            # Take action based on memory pressure
            if memory_percent > self.cleanup_thresholds['emergency']:
                self._emergency_cleanup()
                self._log_alert("EMERGENCY", f"Memory usage at {memory_percent:.1%}")
            elif memory_percent > self.cleanup_thresholds['critical']:
                self._critical_cleanup()
                self._log_alert("CRITICAL", f"Memory usage at {memory_percent:.1%}")
            elif memory_percent > self.cleanup_thresholds['warning']:
                self._warning_cleanup()
                self._log_alert("WARNING", f"Memory usage at {memory_percent:.1%}")

        except Exception as e:
            self.logger.error(f"Memory pressure check failed: {e}")

    def _apply_cleanup_strategies(self):
        """Apply cleanup strategies based on memory pressure."""
        if not self.m1_memory_optimizer:
            return

        try:
            if self.aggressive_cleanup:
                self.m1_memory_optimizer._aggressive_memory_cleanup()
            else:
                self.m1_memory_optimizer._apply_memory_optimizations()
        except Exception as e:
            self.logger.warning(f"Cleanup strategy application failed: {e}")

    def _emergency_cleanup(self):
        """Emergency memory cleanup."""
        tprint("🚨 EMERGENCY: Performing emergency memory cleanup", "ERROR")
        self.logger.warning("🚨 EMERGENCY: Performing emergency memory cleanup")

        try:
            # Force garbage collection multiple times
            for _ in range(5):
                gc.collect()

            # Clear all tracked objects that are not critical
            self._clear_non_critical_objects()

            # Clear caches aggressively
            self._clear_all_caches()

            # Force system memory cleanup if possible
            if self.m1_memory_optimizer:
                self.m1_memory_optimizer._free_unused_memory()

        except Exception as e:
            self.logger.error(f"Emergency cleanup failed: {e}")

    def _critical_cleanup(self):
        """Critical memory cleanup."""
        self.logger.warning("⚠️ CRITICAL: Performing critical memory cleanup")

        try:
            # Force garbage collection
            gc.collect()

            # Clear intermediate states that are not critical
            self._clear_expired_intermediate_states()

            # Optimize memory usage
            if self.m1_memory_optimizer:
                self.m1_memory_optimizer.optimize_memory_usage(aggressive=True)

        except Exception as e:
            self.logger.error(f"Critical cleanup failed: {e}")

    def _warning_cleanup(self):
        """Warning level memory cleanup."""
        self.logger.info("🧠 WARNING: Performing memory cleanup")

        try:
            # Light garbage collection
            gc.collect(0)  # Young generation only

            # Clear expired intermediate states
            self._clear_expired_intermediate_states()

        except Exception as e:
            self.logger.debug(f"Warning cleanup failed: {e}")

    def _clear_non_critical_objects(self):
        """Clear non-critical tracked objects."""
        current_objects = list(self.tracked_objects)
        cleared_count = 0

        for obj in current_objects:
            try:
                # Check if object is still alive and not critical
                if hasattr(obj, '_memory_critical') and obj._memory_critical:
                    continue

                # Remove from tracking
                self.tracked_objects.discard(obj)
                cleared_count += 1

            except (ReferenceError, AttributeError):
                # Object already garbage collected
                self.tracked_objects.discard(obj)
                cleared_count += 1

        if cleared_count > 0:
            self.logger.info(f"Cleared {cleared_count} non-critical objects")

    def _clear_expired_intermediate_states(self):
        """Clear expired intermediate computation states."""
        current_time = time.time()
        expired_keys = []

        for key, state_info in self.intermediate_states.items():
            expiry_time = state_info.get('expiry_time', 0)
            if current_time > expiry_time:
                expired_keys.append(key)

        for key in expired_keys:
            del self.intermediate_states[key]
            self.logger.debug(f"Cleared expired intermediate state: {key}")

        if expired_keys:
            self.logger.info(f"Cleared {len(expired_keys)} expired intermediate states")

    def _clear_all_caches(self):
        """Clear all system caches."""
        try:
            # Clear Python garbage
            gc.collect()

            # Clear M1 memory caches if available
            if self.m1_memory_optimizer:
                self.m1_memory_optimizer._clear_caches_aggressive()

        except Exception as e:
            self.logger.debug(f"Cache clearing failed: {e}")

    def _log_alert(self, level: str, message: str):
        """Log memory alert."""
        alert = {
            'level': level,
            'message': message,
            'timestamp': time.time(),
            'memory_stats': self.get_memory_stats()
        }

        self.memory_alerts.append(alert)

        # Keep only last 100 alerts
        if len(self.memory_alerts) > 100:
            self.memory_alerts = self.memory_alerts[-100:]

        # Log based on level
        if level == "EMERGENCY":
            self.logger.critical(f"🚨 {message}")
        elif level == "CRITICAL":
            self.logger.warning(f"⚠️ {message}")
        else:
            self.logger.info(f"🧠 {message}")

    def register_feature_matrix(self, name: str, matrix: Any, critical: bool = False):
        """Register a feature matrix for memory tracking.

        Args:
            name: Name of the feature matrix
            matrix: The matrix object (DataFrame, ndarray, etc.)
            critical: Whether this matrix is critical for processing
        """
        tprint(f"📊 Registering feature matrix '{name}' (critical: {critical})", "DEBUG")
        try:
            # Calculate memory usage
            memory_mb = self._calculate_object_memory(matrix)

            # Store matrix info
            matrix_info = {
                'name': name,
                'object': matrix,
                'memory_mb': memory_mb,
                'critical': critical,
                'registered_at': time.time(),
                'access_count': 0,
                'last_accessed': time.time()
            }

            self.feature_matrices[name] = matrix_info

            # Track object for automatic cleanup if not critical
            if not critical:
                self.tracked_objects.add(matrix)

            # Set critical flag on object if supported
            if hasattr(matrix, '_memory_critical'):
                matrix._memory_critical = critical

            self.logger.debug(f"Registered feature matrix '{name}': {memory_mb:.1f}MB")
            return True

        except Exception as e:
            self.logger.error(f"Failed to register feature matrix '{name}': {e}")
            return False

    def unregister_feature_matrix(self, name: str) -> bool:
        """Unregister a feature matrix."""
        try:
            if name in self.feature_matrices:
                matrix_info = self.feature_matrices[name]
                matrix = matrix_info['object']

                # Remove from tracking
                self.tracked_objects.discard(matrix)
                del self.feature_matrices[name]

                self.logger.debug(f"Unregistered feature matrix '{name}'")
                return True
            return False

        except Exception as e:
            self.logger.error(f"Failed to unregister feature matrix '{name}': {e}")
            return False

    def register_intermediate_state(self, name: str, state: Any,
                                 expiry_seconds: float = 300.0,
                                 critical: bool = False):
        """Register an intermediate computation state.

        Args:
            name: Name of the state
            state: The state object
            expiry_seconds: How long to keep the state (seconds)
            critical: Whether this state is critical
        """
        try:
            memory_mb = self._calculate_object_memory(state)

            state_info = {
                'name': name,
                'object': state,
                'memory_mb': memory_mb,
                'critical': critical,
                'registered_at': time.time(),
                'expiry_time': time.time() + expiry_seconds,
                'access_count': 0,
                'last_accessed': time.time()
            }

            self.intermediate_states[name] = state_info

            # Track for cleanup if not critical
            if not critical:
                self.tracked_objects.add(state)

            self.logger.debug(f"Registered intermediate state '{name}': {memory_mb:.1f}MB")
            return True

        except Exception as e:
            self.logger.error(f"Failed to register intermediate state '{name}': {e}")
            return False

    def get_intermediate_state(self, name: str) -> Any:
        """Get an intermediate state and update access tracking."""
        if name in self.intermediate_states:
            state_info = self.intermediate_states[name]
            state_info['access_count'] += 1
            state_info['last_accessed'] = time.time()
            return state_info['object']
        return None

    def _calculate_object_memory(self, obj: Any) -> float:
        """Calculate memory usage of an object in MB."""
        try:
            if hasattr(obj, 'memory_usage'):
                # Pandas DataFrame/Series
                return obj.memory_usage(deep=True).sum() / (1024 * 1024)
            elif hasattr(obj, 'nbytes'):
                # NumPy array
                return obj.nbytes / (1024 * 1024)
            elif hasattr(obj, '__sizeof__'):
                # General Python object
                return obj.__sizeof__() / (1024 * 1024)
            else:
                return 1.0  # Default estimate
        except Exception:
            return 1.0  # Fallback

    def optimize_dataframe(self, df: Any) -> Any:
        """Optimize a DataFrame or numpy array for memory usage.

        Args:
            df: DataFrame or numpy array to optimize

        Returns:
            Memory-optimized object
        """
        try:
            if M1_MEMORY_OPTIMIZATION_AVAILABLE and self.m1_memory_optimizer:
                return self.m1_memory_optimizer.optimize_dataframe_memory(df)
            else:
                return df
        except Exception as e:
            self.logger.warning(f"DataFrame optimization failed: {e}")
            return df

    def stream_large_dataset(self, data_loader: Callable,
                           batch_size: Optional[int] = None,
                           max_memory_mb: Optional[float] = None) -> Any:
        """Stream large dataset in batches to manage memory.

        Args:
            data_loader: Function that loads data in batches
            batch_size: Size of each batch
            max_memory_mb: Maximum memory to use for batch

        Returns:
            Generator yielding processed batches
        """
        if batch_size is None:
            batch_size = self.batch_size

        if max_memory_mb is None:
            max_memory_mb = self.chunk_size_mb

        try:
            for batch in data_loader(batch_size):
                batch_memory = self._calculate_object_memory(batch)

                # Check if batch fits in memory budget
                if batch_memory > max_memory_mb:
                    self.logger.warning(f"Batch too large: {batch_memory:.1f}MB > {max_memory_mb:.1f}MB")

                yield self.optimize_dataframe(batch)

                # Force cleanup after each batch
                gc.collect(0)

        except Exception as e:
            self.logger.error(f"Streaming failed: {e}")
            raise

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics."""
        try:
            # Get system memory stats
            if M1_MEMORY_OPTIMIZATION_AVAILABLE and self.m1_memory_optimizer:
                stats = self.m1_memory_optimizer.get_memory_stats()
            else:
                memory = psutil.virtual_memory()
                stats = {
                    'total_memory': memory.total,
                    'available_memory': memory.available,
                    'used_memory': memory.used,
                    'memory_percent': memory.percent,
                    'memory_pressure': memory.percent / 100.0
                }

            # Add our tracking information
            stats.update({
                'feature_matrices_count': len(self.feature_matrices),
                'intermediate_states_count': len(self.intermediate_states),
                'tracked_objects_count': len(self.tracked_objects),
                'total_feature_memory_mb': sum(
                    info['memory_mb'] for info in self.feature_matrices.values()
                ),
                'total_intermediate_memory_mb': sum(
                    info['memory_mb'] for info in self.intermediate_states.values()
                ),
                'alerts_count': len(self.memory_alerts),
                'monitoring_active': self.monitoring_active
            })

            return stats

        except Exception as e:
            self.logger.error(f"Failed to get memory stats: {e}")
            return {'error': str(e)}

    def force_cleanup(self, aggressive: bool = False) -> Dict[str, Any]:
        """Force memory cleanup.

        Args:
            aggressive: Whether to use aggressive cleanup

        Returns:
            Cleanup statistics
        """
        start_time = time.time()
        initial_stats = self.get_memory_stats()

        try:
            if aggressive and self.m1_memory_optimizer:
                self.m1_memory_optimizer._aggressive_memory_cleanup()
            else:
                gc.collect()

            # Clear expired states
            self._clear_expired_intermediate_states()

            final_stats = self.get_memory_stats()
            cleanup_time = time.time() - start_time

            return {
                'success': True,
                'cleanup_time': cleanup_time,
                'memory_before': initial_stats,
                'memory_after': final_stats,
                'aggressive': aggressive
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'cleanup_time': time.time() - start_time
            }

    def get_memory_report(self) -> Dict[str, Any]:
        """Get comprehensive memory report."""
        return {
            'memory_stats': self.get_memory_stats(),
            'feature_matrices': {
                name: {
                    'memory_mb': info['memory_mb'],
                    'critical': info['critical'],
                    'access_count': info['access_count']
                }
                for name, info in self.feature_matrices.items()
            },
            'intermediate_states': {
                name: {
                    'memory_mb': info['memory_mb'],
                    'critical': info['critical'],
                    'expires_in': info['expiry_time'] - time.time()
                }
                for name, info in self.intermediate_states.items()
            },
            'recent_alerts': self.memory_alerts[-10:],  # Last 10 alerts
            'configuration': {
                'memory_limit_gb': self.memory_limit_gb,
                'batch_size': self.batch_size,
                'chunk_size_mb': self.chunk_size_mb,
                'aggressive_cleanup': self.aggressive_cleanup
            }
        }

    def shutdown(self):
        """Shutdown memory manager and cleanup resources."""
        self.stop_monitoring()

        # Clear all tracked objects
        self.feature_matrices.clear()
        self.intermediate_states.clear()
        self.tracked_objects.clear()
        self.memory_alerts.clear()

        self.logger.info("Memory Manager shutdown complete")

# Global instance for easy access
_memory_manager_instance = None

def get_memory_manager(memory_limit_gb: Optional[float] = None,
                      enable_monitoring: bool = True) -> MemoryManager:
    """Get global memory manager instance."""
    global _memory_manager_instance

    if _memory_manager_instance is None:
        _memory_manager_instance = MemoryManager(
            memory_limit_gb=memory_limit_gb,
            enable_monitoring=enable_monitoring
        )

    return _memory_manager_instance

# Convenience functions
def register_feature_matrix(name: str, matrix: Any, critical: bool = False) -> bool:
    """Register a feature matrix for memory tracking."""
    manager = get_memory_manager()
    return manager.register_feature_matrix(name, matrix, critical)

def optimize_dataframe_memory(df: Any) -> Any:
    """Optimize DataFrame memory usage."""
    manager = get_memory_manager()
    return manager.optimize_dataframe(df)

def stream_large_dataset(data_loader: Callable, batch_size: Optional[int] = None,
                        max_memory_mb: Optional[float] = None):
    """Stream large dataset in batches."""
    manager = get_memory_manager()
    return manager.stream_large_dataset(data_loader, batch_size, max_memory_mb)

def get_memory_report() -> Dict[str, Any]:
    """Get comprehensive memory report."""
    manager = get_memory_manager()
    return manager.get_memory_report()
