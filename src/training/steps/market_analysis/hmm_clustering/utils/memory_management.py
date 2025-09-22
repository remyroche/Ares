"""
Memory management utilities for HMM clustering.
"""

import gc
import logging
from typing import Dict, List, Any, Optional, Union
import time

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


class MemoryManager:
    """Memory management utilities for clustering operations."""

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the memory manager.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize hardware memory manager if available
        self.hardware_memory_manager = None
        
        if HARDWARE_ACCELERATION_AVAILABLE:
            try:
                self.hardware_memory_manager = get_memory_manager()
                self.logger.info("✅ Hardware memory manager initialized")
            except Exception as e:
                self.logger.warning(f"⚠️ Hardware memory manager not available: {e}")
        
        # Memory management settings
        self.memory_limit_gb = self.config.get('memory_limit_gb', 8.0)
        self.garbage_collection_enabled = self.config.get('garbage_collection_enabled', True)
        self.memory_monitoring_enabled = self.config.get('memory_monitoring_enabled', True)

    def get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage information.

        Returns:
            Dictionary of memory usage information
        """
        try:
            memory_info = {}
            
            if self.hardware_memory_manager:
                # Use hardware memory manager if available
                hardware_info = self.hardware_memory_manager.get_memory_usage()
                memory_info.update(hardware_info)
            else:
                # Use standard memory monitoring
                import psutil
                memory = psutil.virtual_memory()
                memory_info = {
                    'total_memory_gb': memory.total / (1024**3),
                    'available_memory_gb': memory.available / (1024**3),
                    'used_memory_gb': memory.used / (1024**3),
                    'memory_percent': memory.percent,
                    'free_memory_gb': memory.free / (1024**3)
                }
            
            # Add timestamp
            memory_info['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')
            
            return memory_info
            
        except Exception as e:
            self.logger.warning(f"⚠️ Memory usage check failed: {e}")
            return {
                'error': str(e),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }

    def check_memory_usage(self) -> bool:
        """Check if memory usage is within acceptable limits.

        Returns:
            True if memory usage is acceptable
        """
        try:
            memory_info = self.get_memory_usage()
            
            if 'error' in memory_info:
                return False
            
            used_memory_gb = memory_info.get('used_memory_gb', 0.0)
            
            if used_memory_gb > self.memory_limit_gb:
                self.logger.warning(f"⚠️ Memory usage ({used_memory_gb:.1f}GB) exceeds limit ({self.memory_limit_gb}GB)")
                return False
            
            return True
            
        except Exception as e:
            self.logger.warning(f"⚠️ Memory usage check failed: {e}")
            return False

    def optimize_memory(self) -> Dict[str, Any]:
        """Optimize memory usage.

        Returns:
            Memory optimization results
        """
        try:
            optimization_results = {
                'optimization_applied': True,
                'steps_taken': [],
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Force garbage collection
            if self.garbage_collection_enabled:
                collected = gc.collect()
                optimization_results['steps_taken'].append(f'Garbage collection: {collected} objects collected')
            
            # Use hardware memory manager if available
            if self.hardware_memory_manager:
                hardware_optimization = self.hardware_memory_manager.optimize_memory()
                optimization_results['hardware_optimization'] = hardware_optimization
                optimization_results['steps_taken'].append('Hardware memory optimization applied')
            
            # Get memory usage after optimization
            memory_after = self.get_memory_usage()
            optimization_results['memory_after_optimization'] = memory_after
            
            self.logger.info("✅ Memory optimization completed")
            return optimization_results
            
        except Exception as e:
            self.logger.warning(f"⚠️ Memory optimization failed: {e}")
            return {
                'optimization_applied': False,
                'error': str(e),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }

    def monitor_memory_usage(self, operation_name: str, interval_seconds: int = 5) -> Dict[str, Any]:
        """Monitor memory usage during an operation.

        Args:
            operation_name: Name of the operation being monitored
            interval_seconds: Monitoring interval in seconds

        Returns:
            Memory monitoring configuration
        """
        try:
            monitoring_config = {
                'operation_name': operation_name,
                'interval_seconds': interval_seconds,
                'monitoring_enabled': True,
                'start_time': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            if self.hardware_memory_manager:
                # Use hardware memory monitoring if available
                hardware_monitoring = self.hardware_memory_manager.monitor_memory_usage(
                    operation_name, interval_seconds
                )
                monitoring_config['hardware_monitoring'] = hardware_monitoring
            
            # Get initial memory usage
            initial_memory = self.get_memory_usage()
            monitoring_config['initial_memory_usage'] = initial_memory
            
            self.logger.info(f"✅ Memory monitoring started for operation: {operation_name}")
            return monitoring_config
            
        except Exception as e:
            self.logger.warning(f"⚠️ Memory monitoring setup failed: {e}")
            return {
                'operation_name': operation_name,
                'monitoring_enabled': False,
                'error': str(e),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }

    def get_memory_summary(self, operation_name: str) -> Dict[str, Any]:
        """Get memory usage summary for an operation.

        Args:
            operation_name: Name of the operation

        Returns:
            Memory usage summary
        """
        try:
            summary = {
                'operation_name': operation_name,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Get current memory usage
            current_memory = self.get_memory_usage()
            summary['current_memory_usage'] = current_memory
            
            if self.hardware_memory_manager:
                # Get hardware memory summary if available
                hardware_summary = self.hardware_memory_manager.get_memory_summary(operation_name)
                summary['hardware_memory_summary'] = hardware_summary
            
            return summary
            
        except Exception as e:
            self.logger.warning(f"⚠️ Memory summary generation failed: {e}")
            return {
                'operation_name': operation_name,
                'error': str(e),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }


def create_memory_manager(config: Dict[str, Any] = None) -> MemoryManager:
    """Create a memory manager instance.

    Args:
        config: Configuration dictionary

    Returns:
        MemoryManager instance
    """
    return MemoryManager(config)