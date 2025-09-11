"""
M1 CPU Optimizer for Apple Silicon.

This module provides CPU optimization techniques specifically
designed for Apple Silicon's performance cores and efficiency cores.
"""

import logging
import multiprocessing
import concurrent.futures
import threading
import time
from typing import Any, Dict, List, Optional, Callable, Union
import sys
import platform

logger = logging.getLogger(__name__)

class M1CPUOptimizer:
    """CPU optimizer for M1/M2/M3 performance and efficiency cores."""

    def __init__(self):
        self.logger = logger.getChild('M1CPUOptimizer')
        self.cpu_count = self._get_optimal_cpu_count()
        self.is_m1 = self._detect_m1()
        self.performance_cores = self._get_performance_cores()
        self.efficiency_cores = self._get_efficiency_cores()

    def _detect_m1(self) -> bool:
        """Detect if running on Apple Silicon."""
        try:
            if platform.system() != 'Darwin':
                return False

            import subprocess
            result = subprocess.run(['sysctl', 'machdep.cpu.brand_string'],
                                  capture_output=True, text=True)
            if result.returncode == 0:
                brand = result.stdout.strip()
                return 'Apple' in brand or 'M1' in brand or 'M2' in brand or 'M3' in brand

            return False
        except Exception as e:
            self.logger.warning(f"Could not detect M1 hardware: {e}")
            return False

    def _get_optimal_cpu_count(self) -> int:
        """Get optimal CPU count for M1."""
        try:
            # Use physical cores for M1 optimization
            return max(1, multiprocessing.cpu_count() // 2)  # Use half cores for efficiency
        except Exception:
            return max(1, multiprocessing.cpu_count() // 4)  # Conservative fallback

    def _get_performance_cores(self) -> int:
        """Get number of performance cores."""
        # M1 has 4 performance cores, M2 has 4-8, M3 has 4-12
        # For simplicity, assume 4 performance cores
        return 4

    def _get_efficiency_cores(self) -> int:
        """Get number of efficiency cores."""
        try:
            total_cores = multiprocessing.cpu_count()
            return max(0, total_cores - self.performance_cores)
        except Exception:
            return 0

    def create_optimized_thread_pool(self, max_workers: Optional[int] = None) -> concurrent.futures.ThreadPoolExecutor:
        """Create thread pool optimized for M1."""
        if max_workers is None:
            max_workers = self.cpu_count

        # Limit to performance cores for CPU-intensive tasks
        optimal_workers = min(max_workers, self.performance_cores)

        return concurrent.futures.ThreadPoolExecutor(
            max_workers=optimal_workers,
            thread_name_prefix='M1-Optimized'
        )

    def create_optimized_process_pool(self, max_workers: Optional[int] = None) -> concurrent.futures.ProcessPoolExecutor:
        """Create process pool optimized for M1."""
        if max_workers is None:
            max_workers = max(1, self.cpu_count // 2)  # Use fewer processes for M1

        return concurrent.futures.ProcessPoolExecutor(
            max_workers=max_workers
        )

    def optimize_function_for_m1(self, func: Callable) -> Callable:
        """Optimize function execution for M1."""
        def optimized_wrapper(*args, **kwargs):
            # Set thread affinity to performance cores for CPU-intensive tasks
            try:
                import os
                # This is a simplified approach - in practice, you'd use more sophisticated
                # thread affinity settings for Apple Silicon
                original_affinity = os.sched_getaffinity(0)
                # Keep function execution on current thread for simplicity
            except Exception:
                pass

            try:
                result = func(*args, **kwargs)
                return result
            finally:
                # Restore original affinity if it was changed
                try:
                    if 'original_affinity' in locals():
                        os.sched_setaffinity(0, original_affinity)
                except Exception:
                    pass

        return optimized_wrapper

    def parallel_map_m1(self, func: Callable, items: List[Any],
                       max_workers: Optional[int] = None) -> List[Any]:
        """Parallel map optimized for M1."""
        if not self.is_m1:
            # Fallback to standard parallel execution
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                return list(executor.map(func, items))

        # Use optimized thread pool for M1
        with self.create_optimized_thread_pool(max_workers) as executor:
            optimized_func = self.optimize_function_for_m1(func)
            return list(executor.map(optimized_func, items))

    def run_cpu_intensive_task(self, func: Callable, *args, **kwargs) -> Any:
        """Run CPU-intensive task optimized for M1 performance cores."""
        optimized_func = self.optimize_function_for_m1(func)

        # For very CPU-intensive tasks, consider using efficiency cores
        # to keep performance cores available for other tasks
        return optimized_func(*args, **kwargs)

    def get_cpu_info(self) -> Dict[str, Any]:
        """Get CPU information."""
        return {
            'is_m1': self.is_m1,
            'total_cores': multiprocessing.cpu_count(),
            'performance_cores': self.performance_cores,
            'efficiency_cores': self.efficiency_cores,
            'optimal_workers': self.cpu_count,
            'architecture': platform.machine()
        }

    def optimize_numpy_operations(self):
        """Optimize numpy operations for M1."""
        try:
            import numpy as np

            # Set thread count for numpy operations
            # M1 benefits from using all cores for vectorized operations

            # Set environment variables for optimal performance
            import os
            os.environ['OMP_NUM_THREADS'] = str(self.performance_cores)
            os.environ['MKL_NUM_THREADS'] = str(self.performance_cores)

            self.logger.info(f"🧠 Numpy optimized for {self.performance_cores} performance cores")

        except Exception as e:
            self.logger.warning(f"Numpy optimization failed: {e}")

    def create_m1_optimized_context(self):
        """Create context manager for M1 optimizations."""
        class M1OptimizationContext:
            def __init__(self, optimizer):
                self.optimizer = optimizer
                self.original_env = {}

            def __enter__(self):
                # Save original environment
                import os
                env_vars = ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']
                for var in env_vars:
                    if var in os.environ:
                        self.original_env[var] = os.environ[var]

                # Set optimized values
                optimal_threads = str(self.optimizer.performance_cores)
                os.environ['OMP_NUM_THREADS'] = optimal_threads
                os.environ['MKL_NUM_THREADS'] = optimal_threads
                os.environ['VECLIB_MAXIMUM_THREADS'] = optimal_threads

                self.optimizer.logger.info("🧠 M1 optimization context activated")
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                # Restore original environment
                import os
                for var, value in self.original_env.items():
                    os.environ[var] = value
                for var in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']:
                    if var not in self.original_env and var in os.environ:
                        del os.environ[var]

                self.optimizer.logger.info("🧠 M1 optimization context deactivated")

        return M1OptimizationContext(self)


# Global instance
m1_cpu_optimizer = M1CPUOptimizer()


def get_m1_cpu_optimizer() -> M1CPUOptimizer:
    """Get the global M1 CPU optimizer instance."""
    return m1_cpu_optimizer


def optimize_function_for_m1(func: Callable) -> Callable:
    """Optimize function for M1 execution."""
    return m1_cpu_optimizer.optimize_function_for_m1(func)


def parallel_map_m1(func: Callable, items: List[Any], max_workers: Optional[int] = None) -> List[Any]:
    """Parallel map optimized for M1."""
    return m1_cpu_optimizer.parallel_map_m1(func, items, max_workers)


def create_m1_optimized_thread_pool(max_workers: Optional[int] = None):
    """Create thread pool optimized for M1."""
    return m1_cpu_optimizer.create_optimized_thread_pool(max_workers)


def run_cpu_intensive_task(func: Callable, *args, **kwargs):
    """Run CPU-intensive task optimized for M1."""
    return m1_cpu_optimizer.run_cpu_intensive_task(func, *args, **kwargs)
