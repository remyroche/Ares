"""
Automatic Memory Management System
Handles memory monitoring, cleanup, and automatic resource management
"""

import os
import gc
import psutil
import logging
from typing import Dict, Any, Optional, Callable
from contextlib import contextmanager
import threading
import time

logger = logging.getLogger(__name__)


class MemoryManager:
    """Automatic memory management and monitoring system."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.memory_threshold = self.config.get('memory_threshold', 0.85)  # 85% memory usage
        self.disk_threshold = self.config.get('disk_threshold', 0.90)  # 90% disk usage
        self.check_interval = self.config.get('check_interval', 30)  # Check every 30 seconds
        self.cleanup_enabled = self.config.get('cleanup_enabled', True)

        # Memory monitoring
        self.memory_monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.last_memory_check = 0
        self.memory_alert_count = 0

        # Ray-specific settings
        self.ray_memory_limit = self.config.get('ray_memory_limit', 2 * 1024 * 1024 * 1024)  # 2GB

        # Cleanup callbacks
        self.cleanup_callbacks: list[Callable] = []

        logger.info("🧠 MemoryManager initialized")

    def start_monitoring(self):
        """Start automatic memory monitoring."""
        if self.memory_monitoring:
            return

        self.memory_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("📊 Memory monitoring started")

    def stop_monitoring(self):
        """Stop automatic memory monitoring."""
        self.memory_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("📊 Memory monitoring stopped")

    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.memory_monitoring:
            try:
                self._check_memory_usage()
                self._check_disk_usage()
                time.sleep(self.check_interval)
            except Exception as e:
                logger.warning(f"Memory monitoring error: {e}")
                time.sleep(self.check_interval)

    def _check_memory_usage(self):
        """Check system memory usage and take action if needed."""
        try:
            memory = psutil.virtual_memory()
            memory_percent = memory.percent / 100.0

            if memory_percent > self.memory_threshold:
                self.memory_alert_count += 1
                logger.warning(f"🚨 High memory usage: {memory_percent:.1%} ({memory.used // (1024**3)}GB/{memory.total // (1024**3)}GB)")

                if self.cleanup_enabled:
                    self._perform_memory_cleanup()
            else:
                # Reset alert count when memory is normal
                if self.memory_alert_count > 0:
                    self.memory_alert_count = 0

        except Exception as e:
            logger.warning(f"Memory check failed: {e}")

    def _check_disk_usage(self):
        """Check disk usage and clean up if needed."""
        try:
            # Check main disk
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent / 100.0

            if disk_percent > self.disk_threshold:
                logger.warning(f"💾 High disk usage: {disk_percent:.1%} ({disk.used // (1024**3)}GB/{disk.total // (1024**3)}GB)")
                self._perform_disk_cleanup()

            # Check Ray temp directory specifically
            ray_temp = os.environ.get('RAY_TEMP_DIR', '/tmp/ray')
            if os.path.exists(ray_temp):
                ray_disk = psutil.disk_usage(ray_temp)
                if ray_disk.percent > 85:  # More aggressive for Ray
                    logger.warning(f"🚨 Ray temp directory nearly full: {ray_disk.percent:.1%}")
                    self._cleanup_ray_sessions()

        except Exception as e:
            logger.warning(f"Disk check failed: {e}")

    def _perform_memory_cleanup(self):
        """Perform memory cleanup operations."""
        try:
            logger.info("🧹 Performing memory cleanup...")

            # Force garbage collection
            collected = gc.collect()
            logger.info(f"🗑️  Garbage collected {collected} objects")

            # Clear any caches if available
            self._clear_caches()

            # Call cleanup callbacks
            for callback in self.cleanup_callbacks:
                try:
                    callback()
                except Exception as e:
                    logger.warning(f"Cleanup callback failed: {e}")

            # Check memory after cleanup
            memory = psutil.virtual_memory()
            memory_percent = memory.percent / 100.0
            logger.info(f"✅ Memory after cleanup: {memory_percent:.1%}")

        except Exception as e:
            logger.error(f"Memory cleanup failed: {e}")

    def _perform_disk_cleanup(self):
        """Perform disk cleanup operations."""
        try:
            logger.info("🧹 Performing disk cleanup...")

            # Clean up temporary files
            self._cleanup_temp_files()

            # Clean up Ray sessions
            self._cleanup_ray_sessions()

            # Clean up old log files
            self._cleanup_old_logs()

            logger.info("✅ Disk cleanup completed")

        except Exception as e:
            logger.error(f"Disk cleanup failed: {e}")

    def _clear_caches(self):
        """Clear any available caches."""
        try:
            # Clear Python caches
            import sys
            if hasattr(sys, '_clear_type_cache'):
                sys._clear_type_cache()

            # Try to clear any LRU caches
            for module_name, module in sys.modules.items():
                if hasattr(module, '_cache') and hasattr(module._cache, 'clear'):
                    try:
                        module._cache.clear()
                        logger.debug(f"Cleared cache for {module_name}")
                    except:
                        pass

        except Exception as e:
            logger.debug(f"Cache clearing failed: {e}")

    def _cleanup_temp_files(self):
        """Clean up temporary files."""
        try:
            import tempfile
            import glob

            # Clean old temp files (older than 1 hour)
            temp_dir = tempfile.gettempdir()
            current_time = time.time()

            # Clean Python temp files
            for pattern in ['*.pyc', '*.pyo', '__pycache__']:
                for path in glob.glob(os.path.join(temp_dir, '**', pattern), recursive=True):
                    try:
                        if os.path.isfile(path):
                            os.remove(path)
                    except:
                        pass

        except Exception as e:
            logger.debug(f"Temp file cleanup failed: {e}")

    def _cleanup_ray_sessions(self):
        """Clean up old Ray sessions."""
        try:
            ray_temp = os.environ.get('RAY_TEMP_DIR', '/tmp/ray')

            if os.path.exists(ray_temp):
                # Remove old session directories (keep latest)
                sessions = [d for d in os.listdir(ray_temp) if d.startswith('session_')]
                if len(sessions) > 1:
                    sessions.sort()
                    for old_session in sessions[:-1]:  # Keep the latest
                        session_path = os.path.join(ray_temp, old_session)
                        try:
                            import shutil
                            shutil.rmtree(session_path)
                            logger.info(f"🗑️  Removed old Ray session: {old_session}")
                        except Exception as e:
                            logger.debug(f"Failed to remove {session_path}: {e}")

        except Exception as e:
            logger.debug(f"Ray cleanup failed: {e}")

    def _cleanup_old_logs(self):
        """Clean up old log files."""
        try:
            # Clean logs older than 7 days
            import glob
            current_time = time.time()
            max_age = 7 * 24 * 60 * 60  # 7 days

            for log_pattern in ['*.log', '*.log.*']:
                for log_file in glob.glob(log_pattern):
                    try:
                        if os.path.getmtime(log_file) < (current_time - max_age):
                            os.remove(log_file)
                            logger.debug(f"Removed old log: {log_file}")
                    except:
                        pass

        except Exception as e:
            logger.debug(f"Log cleanup failed: {e}")

    def add_cleanup_callback(self, callback: Callable):
        """Add a cleanup callback function."""
        self.cleanup_callbacks.append(callback)

    def get_memory_status(self) -> Dict[str, Any]:
        """Get current memory status."""
        try:
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')

            return {
                'memory_percent': memory.percent / 100.0,
                'memory_used_gb': memory.used // (1024**3),
                'memory_total_gb': memory.total // (1024**3),
                'disk_percent': disk.percent / 100.0,
                'disk_used_gb': disk.used // (1024**3),
                'disk_total_gb': disk.total // (1024**3),
                'memory_alerts': self.memory_alert_count
            }
        except Exception as e:
            logger.warning(f"Failed to get memory status: {e}")
            return {}

    @contextmanager
    def memory_context(self, operation_name: str = "operation"):
        """Context manager for memory-intensive operations."""
        start_memory = psutil.virtual_memory()
        logger.debug(f"📊 Starting {operation_name} - Memory: {start_memory.percent:.1%}")

        try:
            yield
        finally:
            end_memory = psutil.virtual_memory()
            memory_diff = end_memory.used - start_memory.used

            if memory_diff > 0:
                logger.debug(f"📊 {operation_name} completed - Memory increased by {memory_diff // (1024**2)}MB")
            else:
                logger.debug(f"📊 {operation_name} completed - Memory decreased by {abs(memory_diff) // (1024**2)}MB")

            # Trigger cleanup if memory usage is high
            if end_memory.percent / 100.0 > self.memory_threshold:
                logger.info(f"🚨 High memory usage after {operation_name}, triggering cleanup")
                self._perform_memory_cleanup()


# Global memory manager instance
_memory_manager: Optional[MemoryManager] = None


def get_memory_manager(config: Optional[Dict[str, Any]] = None) -> MemoryManager:
    """Get or create the global memory manager instance."""
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = MemoryManager(config)
    return _memory_manager


def start_memory_monitoring(config: Optional[Dict[str, Any]] = None):
    """Start automatic memory monitoring."""
    manager = get_memory_manager(config)
    manager.start_monitoring()


def stop_memory_monitoring():
    """Stop automatic memory monitoring."""
    global _memory_manager
    if _memory_manager:
        _memory_manager.stop_monitoring()


def memory_context(operation_name: str = "operation"):
    """Context manager for memory monitoring."""
    manager = get_memory_manager()
    return manager.memory_context(operation_name)
