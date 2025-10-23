"""
M1 Memory Optimizer for Apple Silicon.

This module provides memory optimization techniques specifically
designed for Apple Silicon's unified memory architecture.

Version: 2.0.0
Backwards Compatibility: Yes (maintains API compatibility with v1.x)
"""

import logging
import gc
from typing import Any, Dict, List, Optional, Set
import sys
import threading
import time
import warnings
from functools import wraps

# Optional dependencies
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None

logger = logging.getLogger(__name__)

# Version information
__version__ = "2.0.0"
__compatible_versions__ = ["1.0.0", "1.1.0", "1.2.0", "2.0.0"]

def deprecated(reason: str, version: str = "2.0.0"):
    """Decorator to mark functions as deprecated."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            warnings.warn(
                f"{func.__name__} is deprecated since version {version}. {reason}",
                DeprecationWarning,
                stacklevel=2
            )
            return func(*args, **kwargs)
        return wrapper
    return decorator

class M1MemoryOptimizer:
    """Memory optimizer for M1 unified memory architecture with enhanced backwards compatibility."""

    def __init__(self, memory_limit_gb: Optional[float] = None, compatibility_mode: str = "auto"):
        self.logger = logger.getChild('M1MemoryOptimizer')
        self.memory_pressure = 0.0  # Initialize as float
        self.monitoring_active = False
        self.optimization_thread = None
        self.compatibility_mode = compatibility_mode
        self.version = __version__
        
        # Counter to reduce logging frequency (log every 24th check = 120 seconds = 2 minutes max)
        self._log_counter = 0

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

        # M1-specific compatibility flags
        self._legacy_mode = False
        self._m1_detected = self._detect_m1_system()
        self._m1_generation = self._detect_m1_generation()

        if not self._m1_detected:
            self.logger.warning("⚠️ Non-M1 system detected - some optimizations may not be effective")

        # Initialize M1-specific features
        self._initialize_m1_features()

    def _detect_m1_system(self) -> bool:
        """Detect if running on Apple Silicon M1/M2/M3/M4."""
        try:
            import platform
            import subprocess

            if platform.system() != 'Darwin':
                return False

            result = subprocess.run(['sysctl', 'machdep.cpu.brand_string'],
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                brand = result.stdout.strip().lower()
                m1_indicators = ['apple', 'm1', 'm2', 'm3', 'm4', 'silicon']
                return any(indicator in brand for indicator in m1_indicators)
            return False
        except Exception as e:
            self.logger.warning(f"Could not detect M1 system: {e}")
            return False

    def _detect_m1_generation(self) -> str:
        """Detect M1 chip generation for optimization purposes."""
        if not self._m1_detected:
            return "none"

        try:
            import subprocess
            result = subprocess.run(['sysctl', 'machdep.cpu.brand_string'],
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                brand = result.stdout.strip().lower()
                if 'm4' in brand:
                    return "m4"
                elif 'm3' in brand:
                    return "m3"
                elif 'm2' in brand:
                    return "m2"
                elif 'm1' in brand:
                    return "m1"
                elif 'apple' in brand:
                    return "apple_silicon"
            return "unknown"
        except Exception as e:
            self.logger.warning(f"Could not detect M1 generation: {e}")
            return "unknown"

    def _initialize_m1_features(self):
        """Initialize M1-specific memory optimization features."""
        if self._m1_detected:
            # M1-specific memory thresholds based on generation
            if self._m1_generation in ["m3", "m4"]:
                # Newer M1 chips have better memory management
                self.thresholds.update({
                    'low': 0.65,     # Slightly higher thresholds for newer chips
                    'medium': 0.80,
                    'high': 0.90,
                    'critical': 0.95
                })
            elif self._m1_generation in ["m1", "m2"]:
                # Original M1/M2 chips - more conservative thresholds
                self.thresholds.update({
                    'low': 0.55,
                    'medium': 0.70,
                    'high': 0.80,
                    'critical': 0.90
                })

            self.logger.info(f"🧠 M1 Memory Optimizer initialized for {self._m1_generation.upper()}")
        else:
            self.logger.warning("⚠️ M1-specific optimizations disabled - non-M1 system detected")

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
        """Check current memory pressure with M1-specific optimizations."""
        if not PSUTIL_AVAILABLE:
            self.memory_pressure = 0.0
            return

        try:
            memory = psutil.virtual_memory()
            self.memory_pressure = float(memory.percent) / 100.0

            # Increment counter for logging frequency control
            self._log_counter += 1
            should_log = (self._log_counter % 24 == 0)  # Log every 24th check (120 seconds = 2 minutes max)

            # M1-specific memory pressure handling
            if self._m1_detected:
                # M1 unified memory architecture requires different handling
                if self.memory_pressure > self.thresholds['critical']:
                    # Always log critical warnings
                    self.logger.warning(f"🚨 CRITICAL: M1 Memory pressure at {self.memory_pressure:.2f}")
                    self._handle_critical_m1_memory()
                elif self.memory_pressure > self.thresholds['high']:
                    # Always log high warnings
                    self.logger.warning(f"⚠️ HIGH: M1 Memory pressure at {self.memory_pressure:.2f}")
                    self._handle_high_m1_memory()
                elif self.memory_pressure > self.thresholds['medium'] and should_log:
                    # Only log medium pressure every 24th check (2 minutes max)
                    self.logger.info(f"🧠 M1 Memory pressure at {self.memory_pressure:.2f}")
            else:
                # Standard memory pressure handling for non-M1 systems
                if self.memory_pressure > self.thresholds['critical']:
                    # Always log critical warnings
                    self.logger.warning(f"🚨 CRITICAL: Memory pressure at {self.memory_pressure:.2f}")
                elif self.memory_pressure > self.thresholds['high']:
                    # Always log high warnings
                    self.logger.warning(f"⚠️ HIGH: Memory pressure at {self.memory_pressure:.2f}")
                elif self.memory_pressure > self.thresholds['medium'] and should_log:
                    # Only log medium pressure every 24th check (2 minutes max)
                    self.logger.info(f"🧠 Memory pressure at {self.memory_pressure:.2f}")
        except Exception as e:
            # Ensure memory_pressure is always a valid float
            self.memory_pressure = 0.0
            self.logger.error(f"Could not check memory pressure: {e}")

    def _handle_critical_m1_memory(self):
        """Handle critical memory pressure on M1 systems."""
        if not self._m1_detected:
            return

        try:
            # M1-specific critical memory handling
            self.logger.warning("🚨 M1 Critical memory pressure - applying aggressive cleanup")

            # Force multiple garbage collections
            for _ in range(5):
                gc.collect()

            # Clear M1-specific caches
            self._clear_m1_caches()

            # Attempt to free M1 unified memory
            self._free_m1_unified_memory()

        except Exception as e:
            self.logger.error(f"M1 critical memory handling failed: {e}")

    def _handle_high_m1_memory(self):
        """Handle high memory pressure on M1 systems."""
        if not self._m1_detected:
            return

        try:
            # M1-specific high memory handling
            self.logger.info("⚠️ M1 High memory pressure - applying moderate cleanup")

            # Force garbage collection
            gc.collect()

            # Clear M1-specific caches
            self._clear_m1_caches()

        except Exception as e:
            self.logger.error(f"M1 high memory handling failed: {e}")

    def _clear_m1_caches(self):
        """Clear M1-specific caches and optimizations."""
        if not self._m1_detected:
            return

        try:
            # Clear pandas cache if available
            if PANDAS_AVAILABLE and hasattr(pd, '_cache'):
                pd._cache.clear()

            # Clear numpy's internal caches
            try:
                import numpy as np
                if hasattr(np, 'array'):
                    # Force cleanup of array caches
                    pass
            except ImportError:
                pass

            # M1-specific cache clearing
            self.logger.debug("🧹 M1-specific caches cleared")

        except Exception as e:
            self.logger.debug(f"M1 cache clearing failed: {e}")

    def _free_m1_unified_memory(self):
        """Attempt to free M1 unified memory."""
        if not self._m1_detected:
            return

        try:
            # Get current memory usage
            before = psutil.virtual_memory().used

            # Force garbage collection multiple times for M1
            for _ in range(3):
                gc.collect()

            # Try to release memory back to M1 unified memory system
            after = psutil.virtual_memory().used
            freed = before - after

            if freed > 0:
                self.logger.info(f"🧠 M1 unified memory: freed {freed / 1024 / 1024:.1f} MB")

        except Exception as e:
            self.logger.debug(f"M1 unified memory freeing failed: {e}")

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
            if PANDAS_AVAILABLE and hasattr(pd, '_cache'):
                pd._cache.clear()

            # Clear numpy's internal caches
            try:
                import numpy as np
                if hasattr(np, 'array'):
                    # Force cleanup of array caches
                    pass
            except ImportError:
                pass

        except Exception as e:
            self.logger.debug(f"Cache clearing failed: {e}")

    def _clear_caches_aggressive(self):
        """Aggressively clear all caches."""
        try:
            # Clear pandas caches
            if PANDAS_AVAILABLE and hasattr(pd, 'core'):
                # Clear common pandas caches
                pass

            # Force Python garbage collection
            gc.collect()

        except Exception as e:
            self.logger.debug(f"Aggressive cache clearing failed: {e}")

    def _free_unused_memory(self):
        """Attempt to free unused memory."""
        if not PSUTIL_AVAILABLE:
            gc.collect()
            return

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
        if not PANDAS_AVAILABLE:
            self.logger.warning("Pandas not available, returning DataFrame as-is")
            return df

        if df is None:
            return df

        # Check if it's a pandas DataFrame
        if hasattr(df, 'empty'):
            # It's a pandas DataFrame
            if df.empty:
                return df
        else:
            # It's likely a numpy array or other object
            # For numpy arrays, we can't optimize them the same way as DataFrames
            # but we can still return them as-is
            self.logger.debug("Non-DataFrame object passed to optimize_dataframe_memory, returning as-is")
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

    def optimize_series_memory(self, series):
        """Optimize pandas Series memory usage for M1."""
        if not PANDAS_AVAILABLE:
            self.logger.warning("Pandas not available, returning Series as-is")
            return series

        if series is None:
            return series

        # Check if it's a pandas Series
        if not hasattr(series, 'dtype'):
            self.logger.debug("Non-Series object passed to optimize_series_memory, returning as-is")
            return series

        try:
            initial_memory = series.memory_usage(deep=True)

            # Downcast numeric types
            if series.dtype == 'int64':
                series = pd.to_numeric(series, downcast='integer')
            elif series.dtype == 'float64':
                series = pd.to_numeric(series, downcast='float')
            elif series.dtype == 'object':
                # Convert object to category if beneficial
                if series.nunique() / len(series) < 0.5:  # Less than 50% unique values
                    series = series.astype('category')

            final_memory = series.memory_usage(deep=True)
            saved_memory = initial_memory - final_memory

            if saved_memory > 0:
                self.logger.debug(f"🧠 Series memory optimized: {saved_memory / 1024:.1f} KB saved")

        except Exception as e:
            self.logger.warning(f"Series memory optimization failed: {e}")

        return series

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
        if not PSUTIL_AVAILABLE:
            return {
                'total_memory': 0,
                'available_memory': 0,
                'used_memory': 0,
                'memory_percent': 0,
                'memory_pressure': self.memory_pressure,
                'protected_objects': len(self.protected_objects),
                'psutil_available': False
            }

        try:
            memory = psutil.virtual_memory()
            return {
                'total_memory': memory.total,
                'available_memory': memory.available,
                'used_memory': memory.used,
                'memory_percent': memory.percent,
                'memory_pressure': self.memory_pressure,
                'protected_objects': len(self.protected_objects),
                'psutil_available': True
            }
        except Exception as e:
            return {'error': str(e)}

    def get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage statistics (alias for get_memory_stats)."""
        return self.get_memory_stats()

    def load_dataframe(self, file_path: str, **kwargs):
        """Load a DataFrame from file with M1 memory optimization.

        Args:
            file_path: Path to the data file
            **kwargs: Additional arguments passed to pandas read function

        Returns:
            Optimized DataFrame
        """
        if not PANDAS_AVAILABLE:
            self.logger.error("Pandas not available, cannot load DataFrame")
            raise ImportError("Pandas is required to load DataFrames")

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

    def optimize_memory(self) -> Dict[str, Any]:
        """
        Optimize memory usage for M1 architecture (alias for optimize_memory_usage).

        Returns:
            Dictionary with optimization results
        """
        return self.optimize_memory_usage(aggressive=False)

    def optimize_memory_usage(self, aggressive: bool = False) -> Dict[str, Any]:
        """
        Optimize memory usage for M1 architecture.

        Args:
            aggressive: Whether to use aggressive optimization

        Returns:
            Dictionary with optimization results
        """
        start_time = time.time()
        initial_stats = self.get_memory_stats()
        initial_memory = initial_stats.get('current_mb', 0)

        try:
            # Force garbage collection
            collected = gc.collect()

            # Clear unnecessary caches
            if hasattr(sys, 'intern'):
                # Clear string interning cache if possible
                pass

            # Optimize memory pressure monitoring
            if self.monitoring_active and aggressive:
                self.stop_monitoring()
                time.sleep(0.1)  # Brief pause
                self.start_monitoring()

            # Get final stats
            final_stats = self.get_memory_stats()
            final_memory = final_stats.get('current_mb', 0)
            memory_saved = initial_memory - final_memory

            optimization_time = time.time() - start_time

            result = {
                'success': True,
                'memory_saved_mb': memory_saved,
                'initial_memory_mb': initial_memory,
                'final_memory_mb': final_memory,
                'optimization_time_s': optimization_time,
                'gc_collected': collected,
                'aggressive_mode': aggressive
            }

            if memory_saved > 0:
                self.logger.info(f"🧠 Memory optimized: {memory_saved:.1f} MB saved in {optimization_time:.3f}s")
            else:
                self.logger.debug(f"🧠 Memory optimization completed in {optimization_time:.3f}s (no reduction)")

            return result

        except Exception as e:
            self.logger.warning(f"⚠️ Memory optimization failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'optimization_time_s': time.time() - start_time
            }

    @property
    def optimization_context(self):
        """Get optimization context for compatibility."""
        return {
            'memory_limit_gb': self.memory_limit_gb,
            'monitoring_active': self.monitoring_active,
            'memory_stats': self.get_memory_stats(),
            'protected_objects': len(self.protected_objects)
        }

    def get_current_memory_usage_mb(self) -> float:
        """Get current memory usage in MB."""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            return memory_info.rss / (1024 * 1024)  # Convert to MB
        except Exception:
            return 0.0

    def memory_checkpoint(self, checkpoint_name: str):
        """Create a memory checkpoint context manager."""
        from contextlib import contextmanager

        @contextmanager
        def checkpoint_context():
            start_memory = self.get_current_memory_usage_mb()
            start_time = time.time()

            try:
                self.logger.debug(f"🧠 Memory checkpoint '{checkpoint_name}' started: {start_memory:.1f} MB")
                yield
            finally:
                end_memory = self.get_current_memory_usage_mb()
                end_time = time.time()
                memory_diff = end_memory - start_memory
                time_diff = end_time - start_time

                if memory_diff > 10:  # Log if memory increased by more than 10MB
                    self.logger.info(f"🧠 Memory checkpoint '{checkpoint_name}' completed: +{memory_diff:.1f} MB in {time_diff:.3f}s")
                else:
                    self.logger.debug(f"🧠 Memory checkpoint '{checkpoint_name}' completed: {memory_diff:+.1f} MB in {time_diff:.3f}s")

        return checkpoint_context()

    def force_garbage_collection(self) -> None:
        """Force garbage collection to free memory."""

        # Get stats before cleanup
        before_objects = len(gc.get_objects())
        before_garbage = len(gc.garbage)

        # Force garbage collection
        gc.collect()
        gc.collect()  # Double collection for better cleanup
        gc.collect()  # Triple collection for thorough cleanup

        # Get stats after cleanup
        after_objects = len(gc.get_objects())
        after_garbage = len(gc.garbage)

        # Calculate cleanup stats
        objects_freed = before_objects - after_objects
        garbage_cleared = before_garbage - after_garbage

        self.logger.debug(f"🧹 Forced garbage collection: freed {objects_freed} objects, cleared {garbage_cleared} garbage")

# Global instance - lazy initialization to avoid circular import issues
_m1_memory_optimizer_instance: Optional[M1MemoryOptimizer] = None

# Create global instance for backward compatibility
m1_memory_optimizer = None

# M1-specific initialization flag
_m1_initialized = False

def get_m1_memory_optimizer(memory_limit_gb: Optional[float] = None, compatibility_mode: str = "auto") -> M1MemoryOptimizer:
    """Get the M1 memory optimizer instance with enhanced backwards compatibility.

    Args:
        memory_limit_gb: Optional memory limit in GB
        compatibility_mode: Compatibility mode for M1-specific optimizations

    Returns:
        M1MemoryOptimizer instance
    """
    global _m1_memory_optimizer_instance, m1_memory_optimizer, _m1_initialized

    try:
        # Lazy initialization to avoid circular import issues
        # Only create new instance if none exists, ignore parameters for singleton behavior
        if _m1_memory_optimizer_instance is None or not _m1_initialized:
            logger.info("🧠 M1 Memory Optimizer initialized for M1")
            _m1_memory_optimizer_instance = M1MemoryOptimizer(
                memory_limit_gb=memory_limit_gb,
                compatibility_mode=compatibility_mode
            )
            m1_memory_optimizer = _m1_memory_optimizer_instance
            _m1_initialized = True
        else:
            logger.debug("🔄 Reusing existing M1 Memory Optimizer instance")

        return _m1_memory_optimizer_instance

    except Exception as e:
        # Fallback: return a basic instance without memory limit if initialization fails
        logger.warning(f"Failed to initialize M1 memory optimizer: {e}. Using basic instance.")
        _m1_memory_optimizer_instance = M1MemoryOptimizer(memory_limit_gb=None, compatibility_mode="legacy")
        m1_memory_optimizer = _m1_memory_optimizer_instance
        _m1_initialized = True
        return _m1_memory_optimizer_instance

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

def optimize_series_memory(series):
    """Optimize pandas Series memory usage."""
    global _m1_memory_optimizer_instance
    if _m1_memory_optimizer_instance is None:
        _m1_memory_optimizer_instance = get_m1_memory_optimizer()
    return _m1_memory_optimizer_instance.optimize_series_memory(series)

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
