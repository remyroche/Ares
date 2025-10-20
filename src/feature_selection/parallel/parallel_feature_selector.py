"""
Parallel Feature Selection

This module provides parallel processing capabilities for feature selection
operations using hardware optimization tools.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd

# Import hardware optimization tools
from src.utils.parallel_processing_optimizer import MacM1ParallelOptimizer
from src.utils.hardware import (
    get_integrated_hardware_manager,
    memory_efficient,
    performance_tracked,
    smart_cache,
    auto_optimize,
    WorkloadType,
    OptimizationLevel
)
from src.utils.tprint import tprint, tprint_success, tprint_warning, tprint_performance, tprint_debug

logger = logging.getLogger(__name__)

@dataclass
class ParallelConfig:
    """Configuration for parallel feature selection."""
    # Parallel processing settings
    max_workers: Optional[int] = None
    use_process_pool: bool = True
    chunk_size: int = 1000
    memory_limit_mb: int = 2048

    # Hardware optimization
    enable_hardware_optimization: bool = True
    enable_cpu_optimization: bool = True
    enable_memory_optimization: bool = True

    # Performance monitoring
    enable_performance_monitoring: bool = True
    log_timing: bool = True

    # Parallel selection methods
    enable_parallel_methods: bool = True
    parallel_methods: List[str] = None

    def __post_init__(self):
        if self.parallel_methods is None:
            self.parallel_methods = ['comprehensive', 'regularization', 'adaptive']

class ParallelFeatureSelector:
    """Parallel feature selector with hardware optimization."""

    def __init__(self, config: Optional[ParallelConfig] = None):
        """Initialize parallel feature selector."""
        self.config = config or ParallelConfig()
        self.logger = logger.getChild('ParallelFeatureSelector')

        # Initialize hardware tools
        if self.config.enable_hardware_optimization:
            # Get integrated hardware manager
            self.hardware_manager = get_integrated_hardware_manager()
            
            # Initialize parallel optimizer
            self.parallel_optimizer = MacM1ParallelOptimizer(
                max_workers=self.config.max_workers,
                chunk_size=self.config.chunk_size,
                use_process_pool=self.config.use_process_pool,
                memory_limit_mb=self.config.memory_limit_mb
            )
        else:
            self.parallel_optimizer = None
            self.hardware_manager = None

        # Performance tracking
        self.performance_stats = {
            'total_operations': 0,
            'parallel_operations': 0,
            'total_time': 0.0,
            'parallel_time': 0.0,
            'speedup': 0.0
        }

        tprint_success("⚡ ParallelFeatureSelector initialized")

    def _run_single_method(self, X: np.ndarray, y: np.ndarray,
                          method: str, **kwargs) -> Dict[str, Any]:
        """Run a single feature selection method."""
        try:
            from src.feature_selection import select_features

            start_time = time.time()
            result = select_features(X, y, method=method, **kwargs)
            end_time = time.time()

            if self.config.log_timing:
                tprint_performance(f"⏱️ {method}: {end_time - start_time:.2f}s")

            return {
                'method': method,
                'result': result,
                'execution_time': end_time - start_time,
                'success': result.get('success', False)
            }

        except Exception as e:
            self.logger.error(f"Method {method} failed: {e}")
            return {
                'method': method,
                'result': {'success': False, 'error': str(e)},
                'execution_time': 0.0,
                'success': False
            }

    @memory_efficient(memory_threshold_mb=1000.0, auto_cleanup=True)
    @performance_tracked(log_performance=True, track_memory=True)
    def parallel_selection(self, X: np.ndarray, y: np.ndarray,
                          methods: List[str], **kwargs) -> Dict[str, Any]:
        """Run multiple feature selection methods in parallel."""
        if not self.config.enable_parallel_methods:
            # Run sequentially
            return self._sequential_selection(X, y, methods, **kwargs)

        tprint_info(f"⚡ Starting parallel selection: {len(methods)} methods")

        start_time = time.time()

        try:
            if self.parallel_optimizer:
                # Use hardware-optimized parallel processing
                results = self._hardware_optimized_parallel_selection(X, y, methods, **kwargs)
            else:
                # Use standard parallel processing
                results = self._standard_parallel_selection(X, y, methods, **kwargs)

            end_time = time.time()
            execution_time = end_time - start_time

            # Update performance stats
            self.performance_stats['parallel_operations'] += 1
            self.performance_stats['parallel_time'] += execution_time

            # Calculate speedup
            sequential_time = sum(r.get('execution_time', 0) for r in results.values())
            speedup = sequential_time / execution_time if execution_time > 0 else 1.0
            self.performance_stats['speedup'] = speedup

            tprint_success(f"⚡ Parallel selection completed: {execution_time:.2f}s "
                         f"(speedup: {speedup:.1f}x)")

            return {
                'success': True,
                'results': results,
                'execution_time': execution_time,
                'speedup': speedup,
                'methods_used': methods
            }

        except Exception as e:
            self.logger.error(f"Parallel selection failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'execution_time': time.time() - start_time
            }

    def _hardware_optimized_parallel_selection(self, X: np.ndarray, y: np.ndarray,
                                             methods: List[str], **kwargs) -> Dict[str, Any]:
        """Use hardware-optimized parallel processing."""
        try:
            # Create parallel tasks
            tasks = []
            for method in methods:
                task = {
                    'func': self._run_single_method,
                    'args': (X, y, method),
                    'kwargs': kwargs
                }
                tasks.append(task)

            # Use parallel optimizer
            results = self.parallel_optimizer.parallel_apply(
                tasks,
                desc="Feature Selection"
            )

            # Organize results by method
            method_results = {}
            for i, result in enumerate(results):
                method = methods[i]
                method_results[method] = result

            return method_results

        except Exception as e:
            self.logger.error(f"Hardware-optimized parallel selection failed: {e}")
            raise

    def _standard_parallel_selection(self, X: np.ndarray, y: np.ndarray,
                                   methods: List[str], **kwargs) -> Dict[str, Any]:
        """Use standard parallel processing."""
        try:
            # Choose executor type
            if self.config.use_process_pool:
                executor_class = ProcessPoolExecutor
            else:
                executor_class = ThreadPoolExecutor

            # Create executor
            max_workers = self.config.max_workers or 4
            with executor_class(max_workers=max_workers) as executor:
                # Submit tasks
                future_to_method = {
                    executor.submit(self._run_single_method, X, y, method, **kwargs): method
                    for method in methods
                }

                # Collect results
                method_results = {}
                for future in as_completed(future_to_method):
                    method = future_to_method[future]
                    try:
                        result = future.result()
                        method_results[method] = result
                    except Exception as e:
                        self.logger.error(f"Method {method} failed: {e}")
                        method_results[method] = {
                            'method': method,
                            'result': {'success': False, 'error': str(e)},
                            'execution_time': 0.0,
                            'success': False
                        }

                return method_results

        except Exception as e:
            self.logger.error(f"Standard parallel selection failed: {e}")
            raise

    def _sequential_selection(self, X: np.ndarray, y: np.ndarray,
                            methods: List[str], **kwargs) -> Dict[str, Any]:
        """Run methods sequentially (fallback)."""
        tprint_warning("⚠️ Running sequential selection (parallel disabled)")

        method_results = {}
        for method in methods:
            result = self._run_single_method(X, y, method, **kwargs)
            method_results[method] = result

        return method_results

    @memory_efficient(memory_threshold_mb=800.0, auto_cleanup=True)
    @performance_tracked(log_performance=True, track_memory=True)
    def parallel_cross_validation(self, X: np.ndarray, y: np.ndarray,
                                 method: str, cv_folds: int = 5, **kwargs) -> Dict[str, Any]:
        """Run cross-validation in parallel."""
        tprint_info(f"⚡ Starting parallel CV: {method} with {cv_folds} folds")

        try:
            from sklearn.model_selection import TimeSeriesSplit

            # Create CV splits
            tscv = TimeSeriesSplit(n_splits=cv_folds)
            cv_tasks = []

            for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                task = {
                    'func': self._run_single_method,
                    'args': (X_train, y_train, method),
                    'kwargs': kwargs,
                    'fold': fold
                }
                cv_tasks.append(task)

            # Run CV in parallel
            if self.parallel_optimizer:
                cv_results = self.parallel_optimizer.parallel_apply(
                    cv_tasks,
                    desc="Cross-Validation"
                )
            else:
                # Sequential CV
                cv_results = []
                for task in cv_tasks:
                    result = self._run_single_method(*task['args'], **task['kwargs'])
                    result['fold'] = task['fold']
                    cv_results.append(result)

            # Analyze CV results
            successful_folds = [r for r in cv_results if r.get('success', False)]
            execution_times = [r.get('execution_time', 0) for r in successful_folds]

            return {
                'success': True,
                'cv_results': cv_results,
                'successful_folds': len(successful_folds),
                'total_folds': cv_folds,
                'avg_execution_time': np.mean(execution_times) if execution_times else 0.0,
                'method': method
            }

        except Exception as e:
            self.logger.error(f"Parallel CV failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'method': method
            }

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        stats = self.performance_stats.copy()

        if stats['parallel_operations'] > 0:
            stats['avg_speedup'] = stats['speedup'] / stats['parallel_operations']
        else:
            stats['avg_speedup'] = 0.0

        tprint_performance(f"📊 Performance Stats: {stats['avg_speedup']:.1f}x avg speedup, "
                         f"{stats['parallel_operations']} parallel ops")

        return stats

class ParallelSelectionManager:
    """Manager for parallel feature selection operations."""

    def __init__(self, config: Optional[ParallelConfig] = None):
        """Initialize parallel selection manager."""
        self.config = config or ParallelConfig()
        self.selector = ParallelFeatureSelector(self.config)

        tprint_success("⚡ ParallelSelectionManager initialized")

    def compare_methods(self, X: np.ndarray, y: np.ndarray,
                       methods: Optional[List[str]] = None, **kwargs) -> Dict[str, Any]:
        """Compare multiple feature selection methods in parallel."""
        if methods is None:
            methods = self.config.parallel_methods

        return self.selector.parallel_selection(X, y, methods, **kwargs)

    def run_cross_validation(self, X: np.ndarray, y: np.ndarray,
                           method: str, cv_folds: int = 5, **kwargs) -> Dict[str, Any]:
        """Run cross-validation in parallel."""
        return self.selector.parallel_cross_validation(X, y, method, cv_folds, **kwargs)

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        return self.selector.get_performance_stats()

def create_parallel_selector(config: Optional[ParallelConfig] = None) -> ParallelFeatureSelector:
    """Create a parallel feature selector."""
    return ParallelFeatureSelector(config)
