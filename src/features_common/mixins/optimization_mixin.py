"""
Optimization mixin for automatic optimization selection and tuning.

This mixin provides automatic optimization capabilities including
VectorBT optimization, adaptive parameter tuning, and performance monitoring.
"""

import time
import logging
from typing import Dict, Any, Optional, Callable, Union
import pandas as pd
import numpy as np

from ..config import get_unified_config

logger = logging.getLogger(__name__)

class OptimizationMixin:
    """
    Mixin class providing automatic optimization capabilities.

    This mixin can be added to any class to provide automatic optimization
    selection, parameter tuning, and performance monitoring.
    """

    def __init__(self, *args, **kwargs):
        """Initialize optimization mixin."""
        super().__init__(*args, **kwargs)

        # Get unified configuration
        self.config = get_unified_config()

        # Optimization state
        self._optimization_enabled = True
        self._optimization_stats = {
            'total_operations': 0,
            'optimized_operations': 0,
            'fallback_operations': 0,
            'performance_improvements': [],
            'auto_tuning_decisions': 0
        }

        # Performance tracking
        self._performance_history = []
        self._last_performance_check = 0
        self._performance_check_interval = 100  # Check every 100 operations

    def enable_optimization(self) -> None:
        """Enable optimization features."""
        self._optimization_enabled = True
        logger.debug("Optimization enabled")

    def disable_optimization(self) -> None:
        """Disable optimization features."""
        self._optimization_enabled = False
        logger.debug("Optimization disabled")

    def is_optimization_enabled(self) -> bool:
        """Check if optimization is enabled."""
        return self._optimization_enabled

    def should_use_vectorbt(self, data: Union[pd.Series, pd.DataFrame]) -> bool:
        """Determine if VectorBT should be used for the given data."""
        if not self._optimization_enabled:
            return False

        data_size = len(data) if hasattr(data, '__len__') else 0
        return self.config.should_use_vectorbt(data_size)

    def get_optimization_level(self) -> str:
        """Get the current optimization level."""
        return self.config.vectorbt.optimization_level

    def auto_optimize_operation(self,
                               operation_func: Callable,
                               data: Union[pd.Series, pd.DataFrame],
                               *args, **kwargs) -> Any:
        """
        Automatically optimize an operation based on data characteristics.

        Args:
            operation_func: The operation function to optimize
            data: Input data
            *args: Additional arguments for the operation
            **kwargs: Additional keyword arguments for the operation

        Returns:
            Result of the optimized operation
        """
        if not self._optimization_enabled:
            return operation_func(data, *args, **kwargs)

        start_time = time.time()
        self._optimization_stats['total_operations'] += 1

        try:
            # Determine optimization strategy
            strategy = self._determine_optimization_strategy(data)

            # Apply optimization
            if strategy == 'vectorbt':
                result = self._apply_vectorbt_optimization(operation_func, data, *args, **kwargs)
                self._optimization_stats['optimized_operations'] += 1
            elif strategy == 'batch':
                result = self._apply_batch_optimization(operation_func, data, *args, **kwargs)
                self._optimization_stats['optimized_operations'] += 1
            elif strategy == 'memory':
                result = self._apply_memory_optimization(operation_func, data, *args, **kwargs)
                self._optimization_stats['optimized_operations'] += 1
            else:
                result = operation_func(data, *args, **kwargs)
                self._optimization_stats['fallback_operations'] += 1

            # Track performance
            execution_time = time.time() - start_time
            self._track_performance(execution_time, strategy)

            return result

        except Exception as e:
            logger.warning(f"Optimization failed: {e}, using fallback")
            self._optimization_stats['fallback_operations'] += 1
            return operation_func(data, *args, **kwargs)

    def _determine_optimization_strategy(self, data: Union[pd.Series, pd.DataFrame]) -> str:
        """Determine the best optimization strategy for the given data."""
        data_size = len(data) if hasattr(data, '__len__') else 0

        # Prefer VectorBT by default if available and suitable
        if (self.config.vectorbt.enable_vectorbt and
            self.config.vectorbt.prefer_vectorbt and
            data_size >= self.config.vectorbt.data_size_threshold and
            self._is_vectorbt_suitable(data)):
            return 'vectorbt'

        # VectorBT optimization for large datasets (fallback)
        elif self.should_use_vectorbt(data):
            return 'vectorbt'

        # Batch processing for medium datasets
        elif (data_size >= self.config.optimization.batch_size and
              self.config.optimization.enable_batch_processing):
            return 'batch'

        # Memory optimization for any dataset
        elif self.config.optimization.memory_efficient:
            return 'memory'

        # No optimization
        else:
            return 'none'

    def _apply_vectorbt_optimization(self, operation_func: Callable,
                                   data: Union[pd.Series, pd.DataFrame],
                                   *args, **kwargs) -> Any:
        """Apply VectorBT optimization to the operation."""
        # This would integrate with VectorBT optimization components
        # For now, we'll just call the original function
        # In a full implementation, this would use VectorBT's optimized functions
        return operation_func(data, *args, **kwargs)

    def _apply_batch_optimization(self, operation_func: Callable,
                                data: Union[pd.Series, pd.DataFrame],
                                *args, **kwargs) -> Any:
        """Apply batch processing optimization to the operation."""
        # Process data in batches for memory efficiency
        if isinstance(data, pd.DataFrame):
            batch_size = self.config.optimization.batch_size
            results = []

            for i in range(0, len(data), batch_size):
                batch = data.iloc[i:i + batch_size]
                batch_result = operation_func(batch, *args, **kwargs)
                results.append(batch_result)

            # Combine results
            if results and isinstance(results[0], pd.DataFrame):
                return pd.concat(results, ignore_index=True)
            elif results and isinstance(results[0], pd.Series):
                return pd.concat(results, ignore_index=True)
            else:
                return results
        else:
            return operation_func(data, *args, **kwargs)

    def _apply_memory_optimization(self, operation_func: Callable,
                                 data: Union[pd.Series, pd.DataFrame],
                                 *args, **kwargs) -> Any:
        """Apply memory optimization to the operation."""
        # Optimize data types for memory efficiency
        if self.config.optimization.optimize_data_types:
            data = self._optimize_data_types(data)

        # Apply memory pooling if enabled
        if self.config.vectorbt.enable_memory_pooling:
            data = self._apply_memory_pooling(data)

        return operation_func(data, *args, **kwargs)

    def _optimize_data_types(self, data: Union[pd.Series, pd.DataFrame]) -> Union[pd.Series, pd.DataFrame]:
        """Optimize data types for memory efficiency."""
        if isinstance(data, pd.Series):
            if data.dtype == 'float64':
                # Check if float32 is sufficient
                if (data.min() >= np.finfo(np.float32).min and
                    data.max() <= np.finfo(np.float32).max):
                    return data.astype(np.float32)
        elif isinstance(data, pd.DataFrame):
            optimized_data = data.copy()
            for column in optimized_data.columns:
                if optimized_data[column].dtype == 'float64':
                    if (optimized_data[column].min() >= np.finfo(np.float32).min and
                        optimized_data[column].max() <= np.finfo(np.float32).max):
                        optimized_data[column] = optimized_data[column].astype(np.float32)
            return optimized_data

        return data

    def _apply_memory_pooling(self, data: Union[pd.Series, pd.DataFrame]) -> Union[pd.Series, pd.DataFrame]:
        """Apply memory pooling to the data."""
        # This would integrate with a memory pooling system
        # For now, we'll just return the data as-is
        return data

    def _track_performance(self, execution_time: float, strategy: str) -> None:
        """Track performance metrics for optimization decisions."""
        self._performance_history.append({
            'execution_time': execution_time,
            'strategy': strategy,
            'timestamp': time.time()
        })

        # Keep only recent history
        if len(self._performance_history) > 1000:
            self._performance_history = self._performance_history[-500:]

        # Check if we should auto-tune parameters
        if (self.config.optimization.auto_tune_parameters and
            len(self._performance_history) >= 10):
            self._auto_tune_parameters()

    def _auto_tune_parameters(self) -> None:
        """Automatically tune parameters based on performance history."""
        if not self._performance_history:
            return

        recent_history = self._performance_history[-10:]
        avg_time = np.mean([h['execution_time'] for h in recent_history])

        # If performance is below threshold, try more aggressive optimization
        if avg_time > self.config.optimization.performance_threshold:
            self._optimization_stats['auto_tuning_decisions'] += 1
            logger.debug(f"Auto-tuning parameters due to slow performance: {avg_time:.3f}s")

            # This would implement actual parameter tuning
            # For now, we'll just log the decision

    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization statistics."""
        stats = self._optimization_stats.copy()

        # Calculate optimization rate
        if stats['total_operations'] > 0:
            stats['optimization_rate'] = stats['optimized_operations'] / stats['total_operations']
            stats['fallback_rate'] = stats['fallback_operations'] / stats['total_operations']
        else:
            stats['optimization_rate'] = 0
            stats['fallback_rate'] = 0

        # Add performance metrics
        if self._performance_history:
            recent_times = [h['execution_time'] for h in self._performance_history[-10:]]
            stats['avg_execution_time'] = np.mean(recent_times)
            stats['std_execution_time'] = np.std(recent_times)
            stats['min_execution_time'] = np.min(recent_times)
            stats['max_execution_time'] = np.max(recent_times)

        return stats

    def reset_optimization_stats(self) -> None:
        """Reset optimization statistics."""
        self._optimization_stats = {
            'total_operations': 0,
            'optimized_operations': 0,
            'fallback_operations': 0,
            'performance_improvements': [],
            'auto_tuning_decisions': 0
        }
        self._performance_history = []

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get a summary of performance metrics."""
        if not self._performance_history:
            return {'message': 'No performance data available'}

        recent_history = self._performance_history[-50:]  # Last 50 operations

        # Group by strategy
        strategy_stats = {}
        for record in recent_history:
            strategy = record['strategy']
            if strategy not in strategy_stats:
                strategy_stats[strategy] = []
            strategy_stats[strategy].append(record['execution_time'])

        # Calculate statistics for each strategy
        summary = {}
        for strategy, times in strategy_stats.items():
            summary[strategy] = {
                'count': len(times),
                'avg_time': np.mean(times),
                'std_time': np.std(times),
                'min_time': np.min(times),
                'max_time': np.max(times)
            }

        return summary
