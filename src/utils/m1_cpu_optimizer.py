"""
M1 CPU Optimizer for Training Pipeline.

This module provides CPU optimization utilities specifically designed for M1/M2/M3 Macs,
including intelligent parallel processing, thread optimization, and CPU-specific performance enhancements.
"""

import multiprocessing
import concurrent.futures

import logging

from typing import Any, Dict, List, Optional, Callable, TypeVar, Union
from functools import partial
import psutil
import os
import platform

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

T = TypeVar('T')

class M1CPUOptimizer:
    """CPU optimizer for M1 Macs with intelligent parallel processing."""

    def __init__(self, max_workers: Optional[int] = None, enable_hyperthreading: bool = True):
        """Initialize CPU optimizer.

        Args:
            max_workers: Maximum number of worker processes/threads
            enable_hyperthreading: Whether to use hyperthreading
        """
        self.enable_hyperthreading = enable_hyperthreading
        self.system_info = self._get_system_info()
        self.max_workers = max_workers or self._calculate_optimal_workers()
        self.logger = logger.getChild('M1CPUOptimizer')

        self.logger.info(f"⚡ M1 CPU Optimizer initialized ({self.system_info['cpu_count']} CPUs, {self.max_workers} max workers)")

    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information for optimization."""
        try:
            cpu_count = multiprocessing.cpu_count()
            physical_cores = psutil.cpu_count(logical=False)

            return {
                'cpu_count': cpu_count,
                'physical_cores': physical_cores,
                'logical_cores': cpu_count,
                'hyperthreading_available': cpu_count > physical_cores,
                'architecture': platform.machine(),
                'system': platform.system(),
                'processor': platform.processor()
            }
        except Exception as e:
            self.logger.warning(f"Failed to get system info: {e}")
            return {
                'cpu_count': 8,
                'physical_cores': 8,
                'logical_cores': 8,
                'hyperthreading_available': False,
                'architecture': 'arm64',
                'system': 'Darwin',
                'processor': 'Apple M1'
            }

    def _calculate_optimal_workers(self) -> int:
        """Calculate optimal number of workers for M1."""
        cpu_count = self.system_info['cpu_count']

        # M1 optimization: Use all cores but leave some for system
        if self.enable_hyperthreading and self.system_info['hyperthreading_available']:
            # Use logical cores for I/O bound tasks
            optimal = cpu_count
        else:
            # Use physical cores for CPU-bound tasks
            physical_cores = self.system_info['physical_cores']
            optimal = max(1, physical_cores - 1)  # Leave one core for system

        # Cap at reasonable limits for M1
        optimal = min(optimal, 8)  # M1 Max has 10 cores, but limit to 8 for stability

        return optimal

    def get_optimal_workers_for_task(self, task_type: str = "general") -> int:
        """Get optimal number of workers for specific task type."""
        base_workers = self.max_workers

        # Task-specific optimizations
        if task_type == "cpu_bound":
            # Use fewer workers for CPU-bound tasks to avoid overhead
            return max(1, base_workers // 2)
        elif task_type == "io_bound":
            # Use more workers for I/O-bound tasks
            return min(base_workers * 2, 16)
        elif task_type == "memory_bound":
            # Use fewer workers for memory-bound tasks
            return max(1, base_workers // 3)
        else:
            return base_workers

    def parallel_process(
        self,
        items: List[Any],
        processor_func: Callable[[Any], T],
        task_type: str = "general",
        chunk_size: Optional[int] = None,
        timeout: Optional[float] = None
    ) -> List[T]:
        """Process items in parallel with M1 optimization."""

        if not items:
            return []

        num_workers = self.get_optimal_workers_for_task(task_type)

        if len(items) == 1 or num_workers == 1:
            # No parallelism needed
            return [processor_func(item) for item in items]

        # Choose execution method based on task type
        if task_type == "cpu_bound":
            return self._parallel_process_cpu_bound(items, processor_func, num_workers, timeout)
        elif task_type == "io_bound":
            return self._parallel_process_io_bound(items, processor_func, num_workers, timeout)
        else:
            return self._parallel_process_general(items, processor_func, num_workers, chunk_size, timeout)

    def _parallel_process_cpu_bound(
        self,
        items: List[Any],
        processor_func: Callable[[Any], T],
        num_workers: int,
        timeout: Optional[float] = None
    ) -> List[T]:
        """Process CPU-bound tasks with multiprocessing."""
        try:
            with multiprocessing.Pool(processes=num_workers) as pool:
                if timeout:
                    results = pool.map_async(processor_func, items)
                    return results.get(timeout=timeout)
                else:
                    return pool.map(processor_func, items)
        except Exception as e:
            self.logger.warning(f"Multiprocessing failed, falling back to sequential: {e}")
            return [processor_func(item) for item in items]

    def _parallel_process_io_bound(
        self,
        items: List[Any],
        processor_func: Callable[[Any], T],
        num_workers: int,
        timeout: Optional[float] = None
    ) -> List[T]:
        """Process I/O-bound tasks with threading."""
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
                if timeout:
                    futures = [executor.submit(processor_func, item) for item in items]
                    return [future.result(timeout=timeout) for future in concurrent.futures.as_completed(futures)]
                else:
                    return list(executor.map(processor_func, items))
        except Exception as e:
            self.logger.warning(f"Threading failed, falling back to sequential: {e}")
            return [processor_func(item) for item in items]

    def _parallel_process_general(
        self,
        items: List[Any],
        processor_func: Callable[[Any], T],
        num_workers: int,
        chunk_size: Optional[int] = None,
        timeout: Optional[float] = None
    ) -> List[T]:
        """Process general tasks with adaptive method."""
        # Use threading by default for general tasks
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
                if timeout:
                    futures = [executor.submit(processor_func, item) for item in items]
                    return [future.result(timeout=timeout) for future in concurrent.futures.as_completed(futures)]
                else:
                    return list(executor.map(processor_func, items))
        except Exception as e:
            self.logger.warning(f"Parallel processing failed, falling back to sequential: {e}")
            return [processor_func(item) for item in items]

    def parallel_dataframe_processing(
        self,
        df: 'pd.DataFrame',
        processor_func: Callable[['pd.DataFrame'], T],
        chunk_size: Optional[int] = None,
        task_type: str = "general"
    ) -> List[T]:
        """Process DataFrame in parallel chunks."""

        if chunk_size is None:
            # Calculate optimal chunk size
            total_rows = len(df)
            num_workers = self.get_optimal_workers_for_task(task_type)
            chunk_size = max(1, total_rows // num_workers)

        # Split DataFrame into chunks
        chunks = []
        for i in range(0, len(df), chunk_size):
            chunks.append(df.iloc[i:i + chunk_size])

        # Process chunks in parallel
        partial_processor = partial(self._process_dataframe_chunk, processor_func)
        return self.parallel_process(chunks, partial_processor, task_type)

    def _process_dataframe_chunk(self, processor_func: Callable[['pd.DataFrame'], T], chunk: 'pd.DataFrame') -> T:
        """Process a single DataFrame chunk."""
        return processor_func(chunk)

    def optimize_numpy_operations(self):
        """Optimize NumPy operations for M1."""

        # Set thread count for NumPy
        optimal_threads = min(self.system_info['cpu_count'], 8)
        os.environ['OMP_NUM_THREADS'] = str(optimal_threads)
        os.environ['NUMEXPR_NUM_THREADS'] = str(optimal_threads)
        os.environ['MKL_NUM_THREADS'] = str(optimal_threads)

        self.logger.info(f"🔧 NumPy optimized for {optimal_threads} threads")

    def get_cpu_usage_report(self) -> Dict[str, Any]:
        """Generate CPU usage report."""
        try:
            cpu_percent = psutil.cpu_percent(interval=1, percpu=True)
            cpu_freq = psutil.cpu_freq()

            return {
                'cpu_percent_overall': sum(cpu_percent) / len(cpu_percent),
                'cpu_percent_per_core': cpu_percent,
                'cpu_freq_current': cpu_freq.current if cpu_freq else None,
                'cpu_freq_min': cpu_freq.min if cpu_freq else None,
                'cpu_freq_max': cpu_freq.max if cpu_freq else None,
                'optimal_workers': self.max_workers,
                'system_info': self.system_info
            }
        except Exception as e:
            self.logger.warning(f"Failed to get CPU usage report: {e}")
            return {'error': str(e)}

    def adaptive_worker_scaling(
        self,
        current_load: float,
        task_complexity: str = "medium"
    ) -> int:
        """Adaptively scale worker count based on system load."""

        # Complexity factors
        complexity_factors = {
            'low': 1.2,
            'medium': 1.0,
            'high': 0.8
        }

        factor = complexity_factors.get(task_complexity, 1.0)

        # Adjust based on CPU load
        if current_load > 80:
            # High load, reduce workers
            scaled_workers = max(1, int(self.max_workers * 0.5 * factor))
        elif current_load > 60:
            # Medium load, slight reduction
            scaled_workers = max(1, int(self.max_workers * 0.75 * factor))
        else:
            # Low load, can use more workers
            scaled_workers = min(self.max_workers * 2, int(self.max_workers * 1.5 * factor))

        return scaled_workers

class M1BatchProcessor:
    """Batch processor optimized for M1 architecture."""

    def __init__(self, cpu_optimizer: M1CPUOptimizer, batch_size: int = 1000):
        self.cpu_optimizer = cpu_optimizer
        self.batch_size = batch_size
        self.logger = logger.getChild('M1BatchProcessor')

    def calculate_optimal_batch_size(
        self,
        data_size: int,
        operation_type: str = "general",
        memory_limit_mb: int = 1024
    ) -> int:
        """Calculate optimal batch size based on data and system constraints."""

        # Base batch sizes for different operations
        base_sizes = {
            'matrix_mult': 512,
            'neural_net': 256,
            'general': 1000,
            'io_bound': 2000
        }

        base_size = base_sizes.get(operation_type, base_sizes['general'])

        # Adjust based on CPU count
        cpu_factor = self.cpu_optimizer.system_info['cpu_count'] / 8.0  # Normalize to 8 CPUs
        optimal_size = int(base_size * cpu_factor)

        # Adjust based on data size
        if data_size < optimal_size:
            optimal_size = data_size

        # Ensure reasonable bounds
        optimal_size = max(1, min(optimal_size, data_size))

        self.logger.debug(f"📏 Optimal batch size for {operation_type}: {optimal_size}")
        return optimal_size

    def process_in_batches(
        self,
        items: List[Any],
        processor_func: Callable[[List[Any]], T],
        operation_type: str = "general"
    ) -> List[T]:
        """Process items in optimized batches."""

        if not items:
            return []

        batch_size = self.calculate_optimal_batch_size(len(items), operation_type)

        if len(items) <= batch_size:
            return [processor_func(items)]

        results = []
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            result = processor_func(batch)
            results.append(result)

        return results

# Global instance
_m1_cpu_optimizer = None

def get_m1_cpu_optimizer() -> M1CPUOptimizer:
    """Get global M1 CPU optimizer instance."""
    global _m1_cpu_optimizer
    if _m1_cpu_optimizer is None:
        _m1_cpu_optimizer = initialize_m1_cpu_optimizer()
    return _m1_cpu_optimizer

def initialize_m1_cpu_optimizer() -> M1CPUOptimizer:
    """Initialize M1 CPU optimizer with optimal settings."""
    optimizer = M1CPUOptimizer()

    # Apply system-wide optimizations
    optimizer.optimize_numpy_operations()

    return optimizer

def parallel_map(
    func: Callable[[Any], T],
    items: List[Any],
    task_type: str = "general",
    num_workers: Optional[int] = None
) -> List[T]:
    """Convenience function for parallel mapping."""
    optimizer = get_m1_cpu_optimizer()

    if num_workers:
        optimizer.max_workers = num_workers

    return optimizer.parallel_process(items, func, task_type)

def parallel_dataframe_operation(
    df: 'pd.DataFrame',
    operation: Callable[['pd.DataFrame'], T],
    task_type: str = "general"
) -> List[T]:
    """Convenience function for parallel DataFrame operations."""
    optimizer = get_m1_cpu_optimizer()
    return optimizer.parallel_dataframe_processing(df, operation, task_type=task_type)

def parallel_monte_carlo_simulation(
    historical_data: np.ndarray,
    n_simulations: int,
    simulation_func: Callable[[np.ndarray, int], Dict[str, Any]],
    trading_days: int = 252,
    max_workers: Optional[int] = None
) -> Dict[str, Any]:
    """Parallel Monte Carlo simulation optimized for M1."""
    optimizer = get_m1_cpu_optimizer()

    # Split simulations across workers
    if max_workers is None:
        max_workers = optimizer.max_workers

    simulations_per_worker = n_simulations // max_workers
    remainder = n_simulations % max_workers

    # Create simulation tasks
    tasks = []
    for i in range(max_workers):
        worker_simulations = simulations_per_worker + (1 if i < remainder else 0)
        tasks.append((historical_data, worker_simulations, trading_days))

    def worker_simulation(args):
        data, n_sims, days = args
        return simulation_func(data, n_sims)

    # Execute in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(worker_simulation, tasks))

    # Combine results
    combined_results = {
        'returns': [],
        'sharpe_ratios': [],
        'max_drawdowns': [],
        'win_rates': [],
        'volatilities': [],
        'var_95': [],
        'cvar_95': [],
        'convergence_history': []
    }

    for result in results:
        for key in combined_results:
            if key in result:
                combined_results[key].extend(result[key])

    return combined_results

def optimized_monte_carlo_worker(
    historical_data: np.ndarray,
    n_simulations: int,
    trading_days: int = 252,
    random_seed: int = 42
) -> Dict[str, Any]:
    """Optimized worker function for Monte Carlo simulations."""
    np.random.seed(random_seed)

    results = {
        'returns': [],
        'sharpe_ratios': [],
        'max_drawdowns': [],
        'win_rates': [],
        'volatilities': [],
        'var_95': [],
        'cvar_95': [],
        'convergence_history': []
    }

    # Vectorized bootstrap sampling
    for sim in range(n_simulations):
        # Generate bootstrap sample
        bootstrap_returns = np.random.choice(
            historical_data, size=trading_days, replace=True
        )

        # Vectorized calculations
        cumulative_returns = np.cumprod(1 + bootstrap_returns)

        # Performance metrics
        total_return = cumulative_returns[-1] - 1
        annualized_return = (1 + total_return) ** (252 / trading_days) - 1
        annualized_volatility = np.std(bootstrap_returns) * np.sqrt(252)

        # Sharpe ratio (assuming 2% risk-free rate)
        risk_free_rate = 0.02
        sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility \
                      if annualized_volatility > 0 else 0

        # Maximum drawdown using vectorized operations
        peak = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - peak) / peak
        max_drawdown = np.min(drawdown)

        # Win rate
        win_rate = np.mean(bootstrap_returns > 0)

        # Value at Risk (95% confidence)
        var_95 = np.percentile(bootstrap_returns, 5)

        # Conditional Value at Risk (CVaR) - Expected Shortfall
        losses = bootstrap_returns[bootstrap_returns <= var_95]
        cvar_95 = np.mean(losses) if len(losses) > 0 else var_95

        # Store results
        results['returns'].append(float(total_return))
        results['sharpe_ratios'].append(float(sharpe_ratio))
        results['max_drawdowns'].append(float(max_drawdown))
        results['win_rates'].append(float(win_rate))
        results['volatilities'].append(float(annualized_volatility))
        results['var_95'].append(float(var_95))
        results['cvar_95'].append(float(cvar_95))

    return results
