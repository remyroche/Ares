"""
M1 Optimizer for Apple Silicon.

This module provides CPU optimization techniques specifically
designed for Apple Silicon's performance cores and efficiency cores.
"""

import logging
import threading
import time
from typing import Any, Dict, List, Optional, Callable, Union
import functools
import psutil
import multiprocessing as mp

# Import from hardware optimization modules
try:
    from src.utils.hardware.m1_cpu_optimizer import (
        get_m1_cpu_optimizer, M1CPUOptimizer,
        optimize_function_for_m1, parallel_map_m1,
        create_m1_optimized_thread_pool, run_cpu_intensive_task
    )
    M1_OPTIMIZATION_AVAILABLE = True
except ImportError:
    M1_OPTIMIZATION_AVAILABLE = False
    M1CPUOptimizer = None

try:
    from src.utils.matrix_operations.vectorized_core import get_vectorized_processing_core
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

class M1Optimizer:
    """Optimize NumPy/Numba ops for Apple Silicon with thread management and vectorization hints."""

    def __init__(self, enable_monitoring: bool = True, conservative_mode: bool = False):
        """Initialize M1 Optimizer.

        Args:
            enable_monitoring: Whether to enable performance monitoring
            conservative_mode: Use conservative CPU settings for reduced power usage
        """
        self.logger = logger.getChild('M1Optimizer')
        tprint(f"🚀 M1 Optimizer initialized (monitoring: {enable_monitoring}, conservative: {conservative_mode})", "INFO")

        if not M1_OPTIMIZATION_AVAILABLE:
            tprint("⚠️ M1 optimization tools not available, using fallback implementation", "WARNING")
            self.logger.warning("M1 optimization tools not available, using fallback implementation")
            self.m1_optimizer = None
        else:
            self.m1_optimizer = get_m1_cpu_optimizer()

        self.enable_monitoring = enable_monitoring
        self.conservative_mode = conservative_mode

        # Performance tracking
        self.performance_stats = {
            'optimizations_applied': 0,
            'total_execution_time': 0.0,
            'average_speedup': 1.0,
            'cpu_efficiency': 0.0
        }

        # Thread management
        self.optimized_thread_pools = {}
        self.active_optimizations = set()

        # Vectorization support
        if VECTORIZED_CORE_AVAILABLE:
            self.vectorized_core = get_vectorized_processing_core()
        else:
            self.vectorized_core = None

        # Start monitoring if enabled
        if self.enable_monitoring:
            self._start_monitoring()

        self.logger.info(f"M1 Optimizer initialized (monitoring: {enable_monitoring}, conservative: {conservative_mode})")

    def _start_monitoring(self):
        """Start performance monitoring."""
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True,
            name="M1Optimizer-Monitor"
        )
        self.monitoring_thread.start()
        self.logger.debug("M1 Optimizer monitoring started")

    def _monitoring_loop(self):
        """Main monitoring loop for performance tracking."""
        while self.enable_monitoring:
            try:
                self._update_performance_stats()
                time.sleep(5.0)  # Monitor every 5 seconds
            except Exception as e:
                self.logger.error(f"Monitoring error: {e}")
                time.sleep(10.0)

    def _update_performance_stats(self):
        """Update performance statistics."""
        if self.m1_optimizer:
            try:
                cpu_info = self.m1_optimizer.get_cpu_info()
                self.performance_stats['cpu_efficiency'] = (
                    cpu_info.get('performance_cores', 0) / max(1, cpu_info.get('total_cores', 1))
                )
            except Exception as e:
                self.logger.debug(f"Could not update CPU stats: {e}")

    def optimize_function(self, func: Callable, vectorize: bool = True) -> Callable:
        """Optimize a function for M1 execution.

        Args:
            func: Function to optimize
            vectorize: Whether to apply vectorization hints

        Returns:
            Optimized function wrapper
        """
        tprint(f"🔧 Optimizing function {func.__name__} for M1 execution", "DEBUG")
        if not M1_OPTIMIZATION_AVAILABLE:
            tprint("⚠️ M1 optimization not available, returning original function", "WARNING")
            return func

        try:
            # Apply M1-specific optimizations
            optimized_func = optimize_function_for_m1(func)

            # Add vectorization hints if enabled
            if vectorize and self.vectorized_core:
                @functools.wraps(func)
                def vectorized_wrapper(*args, **kwargs):
                    # Check if we can apply vectorization
                    if len(args) > 0 and hasattr(args[0], '__len__'):
                        try:
                            # Try to optimize data structures for vectorization
                            optimized_args = self._optimize_for_vectorization(args)
                            return optimized_func(*optimized_args, **kwargs)
                        except Exception:
                            pass

                    return optimized_func(*args, **kwargs)

                self.performance_stats['optimizations_applied'] += 1
                self.active_optimizations.add(id(optimized_func))
                return vectorized_wrapper
            else:
                self.performance_stats['optimizations_applied'] += 1
                self.active_optimizations.add(id(optimized_func))
                return optimized_func

        except Exception as e:
            self.logger.warning(f"Function optimization failed: {e}")
            return func

    def _optimize_for_vectorization(self, args) -> tuple:
        """Optimize arguments for vectorized processing."""
        optimized_args = []

        for arg in args:
            if hasattr(arg, 'shape') and hasattr(arg, 'dtype'):  # NumPy-like array
                # Apply memory optimization if available
                if self.vectorized_core:
                    try:
                        optimized_arg = self.vectorized_core.optimize_dataframe_for_processing(arg)
                        optimized_args.append(optimized_arg)
                    except Exception:
                        optimized_args.append(arg)
                else:
                    optimized_args.append(arg)
            else:
                optimized_args.append(arg)

        return tuple(optimized_args)

    def create_optimized_thread_pool(self, max_workers: Optional[int] = None,
                                   pool_name: str = "default") -> Any:
        """Create thread pool optimized for M1.

        Args:
            max_workers: Maximum number of worker threads
            pool_name: Name for the thread pool

        Returns:
            Optimized thread pool executor
        """
        if not M1_OPTIMIZATION_AVAILABLE:
            return threading.ThreadPoolExecutor(max_workers=max_workers)

        try:
            if pool_name in self.optimized_thread_pools:
                return self.optimized_thread_pools[pool_name]

            # Create M1-optimized thread pool
            if self.m1_optimizer:
                pool = self.m1_optimizer.create_optimized_thread_pool(max_workers)
            else:
                pool = create_m1_optimized_thread_pool(max_workers)

            self.optimized_thread_pools[pool_name] = pool
            self.logger.debug(f"Created optimized thread pool '{pool_name}'")
            return pool

        except Exception as e:
            self.logger.warning(f"Failed to create optimized thread pool: {e}")
            return threading.ThreadPoolExecutor(max_workers=max_workers)

    def parallel_map(self, func: Callable, items: List[Any],
                    max_workers: Optional[int] = None,
                    use_m1_optimization: bool = True) -> List[Any]:
        """Parallel map optimized for M1.

        Args:
            func: Function to apply
            items: Items to process
            max_workers: Maximum number of workers
            use_m1_optimization: Whether to use M1 optimizations

        Returns:
            List of results
        """
        tprint(f"⚡ Starting parallel map of {len(items)} items with function {func.__name__}", "DEBUG")
        start_time = time.time()

        try:
            if use_m1_optimization and M1_OPTIMIZATION_AVAILABLE:
                # Use M1-optimized parallel processing
                results = parallel_map_m1(func, items, max_workers)
            else:
                # Fallback to standard parallel processing
                with threading.ThreadPoolExecutor(max_workers=max_workers) as executor:
                    results = list(executor.map(func, items))

            execution_time = time.time() - start_time

            # Update performance stats
            if self.enable_monitoring:
                speedup = self._calculate_speedup(execution_time, len(items))
                self.performance_stats['total_execution_time'] += execution_time
                self.performance_stats['average_speedup'] = (
                    self.performance_stats['average_speedup'] * 0.9 + speedup * 0.1
                )

            return results

        except Exception as e:
            self.logger.error(f"Parallel map failed: {e}")
            # Fallback to sequential processing
            return [func(item) for item in items]

    def _calculate_speedup(self, execution_time: float, item_count: int) -> float:
        """Calculate speedup factor compared to sequential processing."""
        if item_count == 0:
            return 1.0

        # Estimate sequential time (assuming linear scaling)
        estimated_sequential = execution_time * item_count
        if estimated_sequential > 0:
            return execution_time / estimated_sequential
        return 1.0

    def run_cpu_intensive_task(self, func: Callable, *args, **kwargs) -> Any:
        """Run CPU-intensive task optimized for M1.

        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result
        """
        if not M1_OPTIMIZATION_AVAILABLE:
            return func(*args, **kwargs)

        try:
            return run_cpu_intensive_task(func, *args, **kwargs)
        except Exception as e:
            self.logger.warning(f"CPU-intensive task optimization failed: {e}")
            return func(*args, **kwargs)

    def optimize_matrix_operations(self, enable_acceleration: bool = True) -> Dict[str, Any]:
        """Optimize matrix operations for M1.

        Args:
            enable_acceleration: Whether to enable hardware acceleration

        Returns:
            Optimization report
        """
        start_time = time.time()

        try:
            if not M1_OPTIMIZATION_AVAILABLE or not self.m1_optimizer:
                return {
                    'success': False,
                    'error': 'M1 optimization not available',
                    'execution_time': time.time() - start_time
                }

            # Apply M1-specific optimizations
            self.m1_optimizer.optimize_numpy_operations()

            # Set conservative mode if requested
            if self.conservative_mode:
                self.m1_optimizer.set_conservative_mode()

            # Get optimization info
            cpu_info = self.m1_optimizer.get_cpu_info()

            return {
                'success': True,
                'optimization_applied': 'numpy_optimization',
                'cpu_info': cpu_info,
                'conservative_mode': self.conservative_mode,
                'execution_time': time.time() - start_time,
                'performance_cores_utilized': cpu_info.get('performance_cores', 0)
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'execution_time': time.time() - start_time
            }

    def get_optimization_context(self):
        """Get M1 optimization context manager."""
        if not M1_OPTIMIZATION_AVAILABLE or not self.m1_optimizer:
            # Return dummy context manager
            class DummyContext:
                def __enter__(self): return self
                def __exit__(self, *args): pass
            return DummyContext()

        return self.m1_optimizer.create_m1_optimized_context()

    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        report = {
            'm1_optimization_available': M1_OPTIMIZATION_AVAILABLE,
            'vectorized_core_available': VECTORIZED_CORE_AVAILABLE,
            'performance_stats': self.performance_stats.copy(),
            'monitoring_active': self.enable_monitoring,
            'conservative_mode': self.conservative_mode,
            'active_optimizations': len(self.active_optimizations),
            'thread_pools_created': len(self.optimized_thread_pools)
        }

        # Add CPU information if available
        if self.m1_optimizer:
            try:
                cpu_info = self.m1_optimizer.get_cpu_info()
                report['cpu_info'] = cpu_info
            except Exception as e:
                report['cpu_info_error'] = str(e)

        # Add vectorized core info if available
        if self.vectorized_core:
            try:
                vectorized_stats = self.vectorized_core.get_processing_stats()
                report['vectorized_core_stats'] = vectorized_stats
            except Exception as e:
                report['vectorized_core_error'] = str(e)

        return report

    def shutdown(self):
        """Shutdown the optimizer and cleanup resources."""
        self.enable_monitoring = False

        # Shutdown monitoring thread
        if hasattr(self, 'monitoring_thread') and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=2.0)

        # Shutdown thread pools
        for pool_name, pool in self.optimized_thread_pools.items():
            try:
                pool.shutdown(wait=True, timeout=2.0)
                self.logger.debug(f"Shutdown thread pool '{pool_name}'")
            except Exception as e:
                self.logger.warning(f"Error shutting down thread pool '{pool_name}': {e}")

        self.optimized_thread_pools.clear()
        self.active_optimizations.clear()

        self.logger.info("M1 Optimizer shutdown complete")

# Global instance for easy access
_m1_optimizer_instance = None

def get_m1_optimizer(enable_monitoring: bool = True, conservative_mode: bool = False) -> M1Optimizer:
    """Get global M1 optimizer instance."""
    global _m1_optimizer_instance

    if _m1_optimizer_instance is None:
        _m1_optimizer_instance = M1Optimizer(
            enable_monitoring=enable_monitoring,
            conservative_mode=conservative_mode
        )

    return _m1_optimizer_instance

# Convenience functions
def optimize_for_m1(func: Callable, vectorize: bool = True) -> Callable:
    """Optimize a function for M1 execution."""
    optimizer = get_m1_optimizer()
    return optimizer.optimize_function(func, vectorize)

def create_optimized_thread_pool(max_workers: Optional[int] = None, pool_name: str = "default"):
    """Create M1-optimized thread pool."""
    optimizer = get_m1_optimizer()
    return optimizer.create_optimized_thread_pool(max_workers, pool_name)

def parallel_map_optimized(func: Callable, items: List[Any],
                         max_workers: Optional[int] = None) -> List[Any]:
    """Parallel map with M1 optimization."""
    optimizer = get_m1_optimizer()
    return optimizer.parallel_map(func, items, max_workers)
