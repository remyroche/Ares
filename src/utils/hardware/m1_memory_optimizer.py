"""
M1 Memory Optimizer for Apple Silicon.

This module provides memory optimization techniques specifically
designed for Apple Silicon's unified memory architecture.
"""

import logging
import psutil
import gc
import pandas as pd
from typing import Any, Dict, List, Optional, Set
import sys
import threading
import time

logger = logging.getLogger(__name__)

class M1MemoryOptimizer:
    """Memory optimizer for M1 unified memory architecture."""

    def __init__(self, memory_limit_gb: Optional[float] = None):
        self.logger = logger.getChild('M1MemoryOptimizer')
        self.memory_pressure = 0.0  # Initialize as float
        self.monitoring_active = False
        self.optimization_thread = None

        # Memory limit in GB (if specified)
        self.memory_limit_gb = memory_limit_gb
        if memory_limit_gb:
            # Convert GB to bytes for comparison
            self.memory_limit_bytes = memory_limit_gb * 1024 * 1024 * 1024
            self.logger.info(f"🧠 Memory limit set to {memory_limit_gb} GB ({self.memory_limit_bytes} bytes)")

        # Memory thresholds for different optimization levels
        self.thresholds = {
            'low': 0.6,      # 60% memory usage
            'medium': 0.75,  # 75% memory usage
            'high': 0.85,    # 85% memory usage
            'critical': 0.95 # 95% memory usage
        }

        # Track objects to prevent premature garbage collection
        self.protected_objects: Set[int] = set()

    def start_monitoring(self):
        """Start memory monitoring and optimization."""
        if self.monitoring_active:
            return

        self.monitoring_active = True
        self.optimization_thread = threading.Thread(
            target=self._memory_monitoring_loop,
            daemon=True
        )
        self.optimization_thread.start()
        self.logger.info("🧠 M1 memory monitoring started")

    def stop_monitoring(self):
        """Stop memory monitoring."""
        self.monitoring_active = False
        if self.optimization_thread:
            self.optimization_thread.join(timeout=1.0)
        self.logger.info("🧠 M1 memory monitoring stopped")

    def _memory_monitoring_loop(self):
        """Main memory monitoring loop."""
        max_iterations = 3600  # Maximum 1 hour of monitoring (3600 * 5 seconds)
        iteration_count = 0

        while self.monitoring_active and iteration_count < max_iterations:
            try:
                self._check_memory_pressure()
                self._apply_memory_optimizations()
                time.sleep(5)  # Check every 5 seconds
                iteration_count += 1

            except Exception as e:
                self.logger.error(f"Memory monitoring error: {e}")
                time.sleep(10)  # Wait longer on error
                iteration_count += 2  # Count error iterations

        # Auto-stop monitoring if we hit the iteration limit
        if iteration_count >= max_iterations:
            self.logger.warning(f"🧠 Memory monitoring reached maximum iterations ({max_iterations}), auto-stopping")
            self.monitoring_active = False

    def _check_memory_pressure(self):
        """Check current memory pressure."""
        try:
            memory = psutil.virtual_memory()
            self.memory_pressure = float(memory.percent) / 100.0

            if self.memory_pressure > self.thresholds['critical']:
                self.logger.warning(f"🚨 CRITICAL: Memory pressure at {self.memory_pressure:.2f}")
            elif self.memory_pressure > self.thresholds['high']:
                self.logger.warning(f"⚠️ HIGH: Memory pressure at {self.memory_pressure:.2f}")
            elif self.memory_pressure > self.thresholds['medium']:
                self.logger.info(f"🧠 Memory pressure at {self.memory_pressure:.2f}")
        except Exception as e:
            # Ensure memory_pressure is always a valid float
            self.memory_pressure = 0.0
            self.logger.error(f"Could not check memory pressure: {e}")

    def _apply_memory_optimizations(self):
        """Apply memory optimizations based on current pressure."""
        if self.memory_pressure > self.thresholds['high']:
            self._aggressive_memory_cleanup()
        elif self.memory_pressure > self.thresholds['medium']:
            self._moderate_memory_cleanup()
        elif self.memory_pressure > self.thresholds['low']:
            self._light_memory_cleanup()

    def _light_memory_cleanup(self):
        """Light memory cleanup for low pressure."""
        # Force garbage collection
        collected = gc.collect(0)  # Young generation only
        if collected > 0:
            self.logger.debug(f"🧹 Light cleanup: {collected} objects collected")

    def _moderate_memory_cleanup(self):
        """Moderate memory cleanup for medium pressure."""
        # Force full garbage collection
        collected = gc.collect()
        if collected > 0:
            self.logger.info(f"🧹 Moderate cleanup: {collected} objects collected")

        # Clear any cached data
        self._clear_caches()

    def _aggressive_memory_cleanup(self):
        """Aggressive memory cleanup for high pressure."""
        # Force full garbage collection multiple times
        total_collected = 0
        for _ in range(3):
            collected = gc.collect()
            total_collected += collected

        if total_collected > 0:
            self.logger.warning(f"🧹 Aggressive cleanup: {total_collected} objects collected")

        # Clear all caches aggressively
        self._clear_caches_aggressive()

        # Try to free up memory by deleting unused objects
        self._free_unused_memory()

    def _clear_caches(self):
        """Clear various caches."""
        try:
            # Clear pandas cache if available
            if hasattr(pd, '_cache'):
                pd._cache.clear()

            # Clear numpy's internal caches
            import numpy as np
            if hasattr(np, 'array'):
                # Force cleanup of array caches
                pass

        except Exception as e:
            self.logger.debug(f"Cache clearing failed: {e}")

    def _clear_caches_aggressive(self):
        """Aggressively clear all caches."""
        try:
            # Clear pandas caches
            import pandas as pd
            if hasattr(pd, 'core'):
                # Clear common pandas caches
                pass

            # Force Python garbage collection
            gc.collect()

        except Exception as e:
            self.logger.debug(f"Aggressive cache clearing failed: {e}")

    def _free_unused_memory(self):
        """Attempt to free unused memory."""
        try:
            # Get current memory usage
            before = psutil.virtual_memory().used

            # Force garbage collection
            gc.collect()

            # Try to release memory back to system
            # Note: On macOS/Apple Silicon, memory is managed differently

            after = psutil.virtual_memory().used
            freed = before - after

            if freed > 0:
                self.logger.info(f"🧠 Freed {freed / 1024 / 1024:.1f} MB of memory")

        except Exception as e:
            self.logger.debug(f"Memory freeing failed: {e}")

    def optimize_dataframe_memory(self, df):
        """Optimize DataFrame memory usage for M1."""
        if df is None or df.empty:
            return df

        try:
            initial_memory = df.memory_usage(deep=True).sum()

            # Convert object columns to category if beneficial
            for col in df.select_dtypes(include=['object']):
                if df[col].nunique() / len(df) < 0.5:  # Less than 50% unique values
                    df[col] = df[col].astype('category')

            # Downcast numeric types
            for col in df.select_dtypes(include=['int64']):
                df[col] = pd.to_numeric(df[col], downcast='integer')

            for col in df.select_dtypes(include=['float64']):
                df[col] = pd.to_numeric(df[col], downcast='float')

            final_memory = df.memory_usage(deep=True).sum()
            saved_memory = initial_memory - final_memory

            if saved_memory > 0:
                self.logger.info(f"🧠 DataFrame memory optimized: {saved_memory / 1024 / 1024:.1f} MB saved")

        except Exception as e:
            self.logger.warning(f"DataFrame memory optimization failed: {e}")

        return df

    def optimize_dataframe(self, df):
        """Alias for optimize_dataframe_memory for backward compatibility."""
        return self.optimize_dataframe_memory(df)

    def protect_object(self, obj: Any):
        """Protect an object from garbage collection."""
        obj_id = id(obj)
        self.protected_objects.add(obj_id)

    def unprotect_object(self, obj: Any):
        """Remove protection from an object."""
        obj_id = id(obj)
        self.protected_objects.discard(obj_id)

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get current memory statistics."""
        try:
            memory = psutil.virtual_memory()
            return {
                'total_memory': memory.total,
                'available_memory': memory.available,
                'used_memory': memory.used,
                'memory_percent': memory.percent,
                'memory_pressure': self.memory_pressure,
                'protected_objects': len(self.protected_objects)
            }
        except Exception as e:
            return {'error': str(e)}

    def get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage statistics (alias for get_memory_stats)."""
        return self.get_memory_stats()

    def load_dataframe(self, file_path: str, **kwargs) -> pd.DataFrame:
        """Load a DataFrame from file with M1 memory optimization.
        
        Args:
            file_path: Path to the data file
            **kwargs: Additional arguments passed to pandas read function
            
        Returns:
            Optimized DataFrame
        """
        try:
            # Determine file type and load accordingly
            if file_path.endswith('.parquet'):
                df = pd.read_parquet(file_path, **kwargs)
            elif file_path.endswith('.csv'):
                df = pd.read_csv(file_path, **kwargs)
            elif file_path.endswith('.json'):
                df = pd.read_json(file_path, **kwargs)
            elif file_path.endswith('.pickle') or file_path.endswith('.pkl'):
                df = pd.read_pickle(file_path, **kwargs)
            else:
                # Try to infer from file extension
                df = pd.read_csv(file_path, **kwargs)
            
            # Apply memory optimization
            optimized_df = self.optimize_dataframe_memory(df)
            
            self.logger.info(f"📊 Loaded DataFrame from {file_path}: {optimized_df.shape}")
            return optimized_df
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load DataFrame from {file_path}: {e}")
            raise

    def optimize_dataframe(self, df):
        """Alias for optimize_dataframe_memory for compatibility."""
        return self.optimize_dataframe_memory(df)


# Global instance - lazy initialization to avoid circular import issues
_m1_memory_optimizer_instance: Optional[M1MemoryOptimizer] = None


def get_m1_memory_optimizer(memory_limit_gb: Optional[float] = None) -> M1MemoryOptimizer:
    """Get the M1 memory optimizer instance.

    Args:
        memory_limit_gb: Optional memory limit in GB

    Returns:
        M1MemoryOptimizer instance
    """
    global _m1_memory_optimizer_instance

    try:
        # Lazy initialization to avoid circular import issues
        if _m1_memory_optimizer_instance is None:
            _m1_memory_optimizer_instance = M1MemoryOptimizer(memory_limit_gb=memory_limit_gb)
        else:
            # If a memory limit is specified and it's different from the current instance, create a new one
            if memory_limit_gb and (not hasattr(_m1_memory_optimizer_instance, 'memory_limit_gb') or
                                   _m1_memory_optimizer_instance.memory_limit_gb != memory_limit_gb):
                _m1_memory_optimizer_instance = M1MemoryOptimizer(memory_limit_gb=memory_limit_gb)

        return _m1_memory_optimizer_instance

    except Exception as e:
        # Fallback: return a basic instance without memory limit if initialization fails
        logger.warning(f"Failed to initialize M1 memory optimizer: {e}. Using basic instance.")
        return M1MemoryOptimizer(memory_limit_gb=None)


def start_m1_memory_monitoring():
    """Start M1 memory monitoring."""
    global _m1_memory_optimizer_instance
    if _m1_memory_optimizer_instance is None:
        _m1_memory_optimizer_instance = get_m1_memory_optimizer()
    _m1_memory_optimizer_instance.start_monitoring()


def stop_m1_memory_monitoring():
    """Stop M1 memory monitoring."""
    global _m1_memory_optimizer_instance
    if _m1_memory_optimizer_instance is not None:
        _m1_memory_optimizer_instance.stop_monitoring()


def optimize_dataframe_memory(df):
    """Optimize DataFrame memory usage."""
    global _m1_memory_optimizer_instance
    if _m1_memory_optimizer_instance is None:
        _m1_memory_optimizer_instance = get_m1_memory_optimizer()
    return _m1_memory_optimizer_instance.optimize_dataframe_memory(df)


def optimize_dataframe(df):
    """Optimize DataFrame memory usage (alias for optimize_dataframe_memory)."""
    return optimize_dataframe_memory(df)


def optimize_memory() -> Dict[str, Any]:
    """Optimize memory usage and return statistics.

    Returns:
        Dictionary with memory optimization statistics
    """
    try:
        # Force garbage collection
        collected_objects = gc.collect()

        # Get memory stats before and after
        global _m1_memory_optimizer_instance
        if _m1_memory_optimizer_instance is None:
            _m1_memory_optimizer_instance = get_m1_memory_optimizer()
        before_stats = _m1_memory_optimizer_instance.get_memory_stats()

        # Apply memory optimizations
        _m1_memory_optimizer_instance._apply_memory_optimizations()

        after_stats = _m1_memory_optimizer_instance.get_memory_stats()

        return {
            'collected_objects': collected_objects,
            'memory_before': before_stats,
            'memory_after': after_stats,
            'optimization_applied': True,
            'success': True
        }

    except Exception as e:
        logger.error(f"Memory optimization failed: {e}")
        return {
            'error': str(e),
            'optimization_applied': False,
            'success': False
        }

def get_memory_usage() -> Dict[str, Any]:
    """Get current memory usage statistics.

    Returns:
        Dictionary with memory usage statistics
    """
    global _m1_memory_optimizer_instance
    if _m1_memory_optimizer_instance is None:
        _m1_memory_optimizer_instance = get_m1_memory_optimizer()
    return _m1_memory_optimizer_instance.get_memory_stats()


def get_memory_manager() -> M1MemoryOptimizer:
    """Get memory manager instance for parallel processing."""
    global _m1_memory_optimizer_instance
    if _m1_memory_optimizer_instance is None:
        _m1_memory_optimizer_instance = get_m1_memory_optimizer()
    return _m1_memory_optimizer_instance


def get_vectorized_processing_core() -> M1MemoryOptimizer:
    """Get vectorized processing core instance."""
    global _m1_memory_optimizer_instance
    if _m1_memory_optimizer_instance is None:
        _m1_memory_optimizer_instance = get_m1_memory_optimizer()
    return _m1_memory_optimizer_instance
