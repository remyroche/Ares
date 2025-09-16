"""
Memory Tracking Utility

Provides memory usage tracking during training operations.
"""

import time
import gc
import logging
from typing import Dict, List, Optional, Any

try:
    import psutil
    import os
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

logger = logging.getLogger(__name__)


class MemoryTracker:
    """Utility class for tracking memory usage during training."""
    
    def __init__(self, enable_tracking: bool = True):
        """
        Initialize memory tracker.
        
        Args:
            enable_tracking: Whether to enable memory tracking
        """
        self.enable_tracking = enable_tracking and PSUTIL_AVAILABLE
        self.process = None
        self.initial_memory = 0.0
        self.peak_memory = 0.0
        self.memory_snapshots = []
        self.start_time = time.time()
        
        if self.enable_tracking:
            try:
                self.process = psutil.Process(os.getpid())
                self.initial_memory = self._get_memory_usage()
                self.peak_memory = self.initial_memory
            except Exception as e:
                logger.warning(f"Failed to initialize memory tracking: {e}")
                self.enable_tracking = False
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        if not self.enable_tracking or not self.process:
            return 0.0
        
        try:
            return self.process.memory_info().rss / 1024 / 1024
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return 0.0
    
    def take_snapshot(self, label: str = "", force_gc: bool = False) -> float:
        """
        Take a memory snapshot and return current usage.
        
        Args:
            label: Label for the snapshot
            force_gc: Whether to force garbage collection before snapshot
            
        Returns:
            Current memory usage in MB
        """
        if not self.enable_tracking:
            return 0.0
        
        if force_gc:
            gc.collect()
        
        current_memory = self._get_memory_usage()
        self.peak_memory = max(self.peak_memory, current_memory)
        
        snapshot = {
            'label': label,
            'memory_mb': current_memory,
            'timestamp': time.time(),
            'elapsed_time': time.time() - self.start_time
        }
        self.memory_snapshots.append(snapshot)
        
        return current_memory
    
    def get_peak_memory(self) -> float:
        """Get peak memory usage during tracking."""
        return self.peak_memory
    
    def get_memory_increase(self) -> float:
        """Get memory increase from initial state."""
        return self.peak_memory - self.initial_memory
    
    def get_current_memory(self) -> float:
        """Get current memory usage."""
        return self._get_memory_usage()
    
    def get_memory_snapshots(self) -> List[Dict[str, Any]]:
        """Get all memory snapshots."""
        return self.memory_snapshots.copy()
    
    def cleanup(self) -> float:
        """
        Force garbage collection and take final snapshot.
        
        Returns:
            Memory usage after cleanup
        """
        if not self.enable_tracking:
            return 0.0
        
        gc.collect()
        final_memory = self.take_snapshot("cleanup", force_gc=True)
        
        logger.debug(f"Memory cleanup completed. Final usage: {final_memory:.1f}MB")
        return final_memory
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive memory usage summary.
        
        Returns:
            Dictionary with memory statistics
        """
        if not self.enable_tracking:
            return {
                'tracking_enabled': False,
                'initial_memory_mb': 0.0,
                'peak_memory_mb': 0.0,
                'current_memory_mb': 0.0,
                'memory_increase_mb': 0.0,
                'snapshots_count': 0
            }
        
        current_memory = self.get_current_memory()
        
        return {
            'tracking_enabled': True,
            'initial_memory_mb': self.initial_memory,
            'peak_memory_mb': self.peak_memory,
            'current_memory_mb': current_memory,
            'memory_increase_mb': self.get_memory_increase(),
            'snapshots_count': len(self.memory_snapshots),
            'snapshots': self.memory_snapshots
        }
    
    def log_memory_usage(self, operation: str) -> None:
        """
        Log current memory usage for an operation.
        
        Args:
            operation: Name of the operation
        """
        if not self.enable_tracking:
            return
        
        current_memory = self.get_current_memory()
        memory_increase = current_memory - self.initial_memory
        
        logger.info(f"Memory usage for {operation}: {current_memory:.1f}MB "
                   f"(increase: {memory_increase:+.1f}MB)")
    
    def reset(self) -> None:
        """Reset memory tracker for new tracking session."""
        self.initial_memory = self._get_memory_usage()
        self.peak_memory = self.initial_memory
        self.memory_snapshots = []
        self.start_time = time.time()
        
        if self.enable_tracking:
            logger.debug("Memory tracker reset")
    
    def is_tracking_enabled(self) -> bool:
        """Check if memory tracking is enabled."""
        return self.enable_tracking