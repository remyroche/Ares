"""
Unified Vectorization Manager for Backtesting

This module provides a centralized manager for all VectorBT operations across
the backtesting system, ensuring consistent optimization and performance.

Key Features:
- Centralized VectorBT configuration
- Unified rolling operations
- Performance monitoring and optimization
- Memory management
- GPU acceleration support
- Cross-component analytics
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import time
import gc
from pathlib import Path
import json

# VectorBT imports
from src.feature_generation.utils.vectorbt_rolling_optimizer import VectorBTRollingOptimizer
from src.feature_selection.vectorbt.vectorbt_unified_framework import VectorBTUnifiedFramework

# Common utilities
from src.utils.common_operations import safe_json_dump, safe_json_load, ensure_directory
from src.utils.math_validation import validate_finite, validate_positive
from src.core.decorators import handles_errors, traced, log_execution_time

logger = logging.getLogger(__name__)

class VectorBTOperationType(Enum):
    """Types of VectorBT operations."""
    ROLLING_STATISTICS = "rolling_statistics"
    CORRELATION_ANALYSIS = "correlation_analysis"
    RISK_METRICS = "risk_metrics"
    PERFORMANCE_ANALYSIS = "performance_analysis"
    REGIME_ANALYSIS = "regime_analysis"
    BOOTSTRAP_SAMPLING = "bootstrap_sampling"
    FEATURE_SELECTION = "feature_selection"
    PARAMETER_OPTIMIZATION = "parameter_optimization"

@dataclass
class VectorBTConfig:
    """Configuration for VectorBT unified manager."""
    # Basic settings
    enable_parallel: bool = True
    enable_memory_optimization: bool = True
    enable_gpu_acceleration: bool = False
    chunk_size: int = 1000
    fast_fail: bool = False
    enable_logging: bool = True

    # Performance settings
    max_workers: int = 4
    memory_limit_mb: int = 1024
    cache_size: int = 100

    # Analytics settings
    enable_rolling_analytics: bool = True
    enable_cross_metric_analysis: bool = True
    enable_predictive_analytics: bool = True
    analytics_window_size: int = 20

    # Custom parameters
    custom_params: Dict[str, Any] = field(default_factory=dict)

@dataclass
class VectorBTOperationResult:
    """Result of a VectorBT operation."""
    operation_type: VectorBTOperationType
    success: bool
    result: Any
    execution_time: float
    memory_used_mb: float
    vectorbt_optimized: bool
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class VectorBTUnifiedManager:
    """
    Unified Vectorization Manager for centralized VectorBT operations.

    This manager provides:
    - Centralized VectorBT configuration
    - Unified rolling operations
    - Performance monitoring
    - Memory management
    - Cross-component analytics
    """

    def __init__(self, config: VectorBTConfig):
        """Initialize VectorBT unified manager."""
        self.config = config
        self.logger = logger.getChild('VectorBTUnifiedManager')

        # Initialize VectorBT components
        self.rolling_optimizer = VectorBTRollingOptimizer(
            enable_gpu=config.enable_gpu_acceleration,
            enable_parallel=config.enable_parallel,
            memory_efficient=config.enable_memory_optimization,
            chunk_size=config.chunk_size,
            fast_fail=config.fast_fail,
            enable_logging=config.enable_logging
        )

        self.unified_framework = VectorBTUnifiedFramework()

        # Performance tracking
        self.performance_stats = {
            'total_operations': 0,
            'successful_operations': 0,
            'failed_operations': 0,
            'total_execution_time': 0.0,
            'total_memory_used': 0.0,
            'operations_by_type': {},
            'average_execution_time': 0.0,
            'memory_efficiency': 0.0
        }

        # Operation cache
        self.operation_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0

        # Analytics storage
        self.analytics_data = {}
        self.cross_metric_correlations = {}

        self.logger.info("🚀 VectorBTUnifiedManager initialized")
        self.logger.info(f"📊 Parallel processing: {config.enable_parallel}")
        self.logger.info(f"📊 Memory optimization: {config.enable_memory_optimization}")
        self.logger.info(f"📊 GPU acceleration: {config.enable_gpu_acceleration}")
        self.logger.info(f"📊 Analytics enabled: {config.enable_rolling_analytics}")

    @traced(span_name='vectorbt_operation')
    async def execute_operation(self, operation_type: VectorBTOperationType,
                              operation_func: Callable, *args, **kwargs) -> VectorBTOperationResult:
        """Execute a VectorBT operation with monitoring and optimization."""
        start_time = time.time()
        start_memory = self._get_memory_usage()

        try:
            self.logger.debug(f"🔄 Executing {operation_type.value} operation")

            # Check cache first
            cache_key = self._generate_cache_key(operation_type, operation_func, args, kwargs)
            if cache_key in self.operation_cache:
                self.cache_hits += 1
                cached_result = self.operation_cache[cache_key]
                self.logger.debug(f"📋 Cache hit for {operation_type.value}")
                return cached_result

            self.cache_misses += 1

            # Execute operation
            result = await operation_func(*args, **kwargs)

            # Calculate performance metrics
            execution_time = time.time() - start_time
            end_memory = self._get_memory_usage()
            memory_used = end_memory - start_memory

            # Create result
            operation_result = VectorBTOperationResult(
                operation_type=operation_type,
                success=True,
                result=result,
                execution_time=execution_time,
                memory_used_mb=memory_used,
                vectorbt_optimized=True,
                metadata={
                    'cache_key': cache_key,
                    'timestamp': datetime.now().isoformat()
                }
            )

            # Cache result if appropriate
            if self._should_cache_result(operation_type, execution_time):
                self.operation_cache[cache_key] = operation_result
                if len(self.operation_cache) > self.config.cache_size:
                    self._evict_oldest_cache_entry()

            # Update performance stats
            self._update_performance_stats(operation_result)

            self.logger.debug(f"✅ {operation_type.value} completed in {execution_time:.3f}s")
            return operation_result

        except Exception as e:
            execution_time = time.time() - start_time
            end_memory = self._get_memory_usage()
            memory_used = end_memory - start_memory

            self.logger.error(f"❌ {operation_type.value} operation failed: {e}")

            operation_result = VectorBTOperationResult(
                operation_type=operation_type,
                success=False,
                result=None,
                execution_time=execution_time,
                memory_used_mb=memory_used,
                vectorbt_optimized=False,
                error=str(e),
                metadata={
                    'timestamp': datetime.now().isoformat(),
                    'error_type': type(e).__name__
                }
            )

            self._update_performance_stats(operation_result)
            return operation_result

    async def rolling_statistics(self, data: Union[pd.Series, pd.DataFrame],
                               window: int, operations: List[str] = None) -> Dict[str, Any]:
        """Calculate rolling statistics using VectorBT with enhanced optimization."""
        if operations is None:
            operations = ['mean', 'std', 'min', 'max', 'skew', 'kurt']

        async def _calculate_rolling_stats():
            results = {}

            # Use VectorBTRollingOptimizer for enhanced performance
            for op in operations:
                if hasattr(self.rolling_optimizer, f'rolling_{op}'):
                    func = getattr(self.rolling_optimizer, f'rolling_{op}')
                    results[op] = func(data, window=window)
                else:
                    self.logger.warning(f"⚠️ Rolling operation {op} not available")

            return results

        result = await self.execute_operation(
            VectorBTOperationType.ROLLING_STATISTICS,
            _calculate_rolling_stats
        )

        return result.result if result.success else {}

    async def calculate_rolling_metrics_enhanced(self, data: Union[pd.Series, pd.DataFrame],
                                               windows: List[int] = None) -> Dict[str, Any]:
        """
        Calculate enhanced rolling metrics using VectorBTRollingOptimizer.

        Args:
            data: Input data
            windows: List of window sizes

        Returns:
            Dictionary of rolling metrics
        """
        if windows is None:
            windows = [5, 10, 20, 50, 100]

        async def _calculate_enhanced_rolling_metrics():
            results = {}

            for window in windows:
                window_results = {}

                # Use VectorBTRollingOptimizer for each operation
                if hasattr(data, 'close'):
                    close_prices = data['close']

                    window_results['mean'] = self.rolling_optimizer.rolling_mean(close_prices, window=window)
                    window_results['std'] = self.rolling_optimizer.rolling_std(close_prices, window=window)
                    window_results['min'] = self.rolling_optimizer.rolling_min(close_prices, window=window)
                    window_results['max'] = self.rolling_optimizer.rolling_max(close_prices, window=window)
                    window_results['skew'] = self.rolling_optimizer.rolling_skew(close_prices, window=window)
                    window_results['kurt'] = self.rolling_optimizer.rolling_kurt(close_prices, window=window)

                results[f'window_{window}'] = window_results

            return results

        result = await self.execute_operation(
            VectorBTOperationType.ROLLING_STATISTICS,
            _calculate_enhanced_rolling_metrics
        )

        return result.result if result.success else {}

    async def calculate_technical_indicators_enhanced(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate enhanced technical indicators using VectorBTRollingOptimizer.

        Args:
            data: Input OHLCV data

        Returns:
            Dictionary of technical indicators
        """
        async def _calculate_enhanced_technical_indicators():
            results = {}

            if 'close' in data.columns:
                close_prices = data['close']

                # Moving averages
                results['sma_20'] = self.rolling_optimizer.rolling_mean(close_prices, window=20)
                results['sma_50'] = self.rolling_optimizer.rolling_mean(close_prices, window=50)
                results['sma_200'] = self.rolling_optimizer.rolling_mean(close_prices, window=200)

                # Volatility
                results['volatility_20'] = self.rolling_optimizer.rolling_std(close_prices, window=20)
                results['volatility_50'] = self.rolling_optimizer.rolling_std(close_prices, window=50)

                # Price ranges
                if 'high' in data.columns and 'low' in data.columns:
                    high_prices = data['high']
                    low_prices = data['low']

                    # ATR calculation
                    if hasattr(self.rolling_optimizer, 'rolling_atr'):
                        results['atr_20'] = self.rolling_optimizer.rolling_atr(
                            high_prices, low_prices, close_prices, window=20
                        )
                        results['atr_50'] = self.rolling_optimizer.rolling_atr(
                            high_prices, low_prices, close_prices, window=50
                        )

            return results

        result = await self.execute_operation(
            VectorBTOperationType.TECHNICAL_INDICATORS,
            _calculate_enhanced_technical_indicators
        )

        return result.result if result.success else {}

    async def optimize_parameter_evaluation(self, objective_function: Callable,
                                          parameters: Dict[str, Any],
                                          data: Optional[pd.DataFrame] = None) -> Any:
        """
        Optimize parameter evaluation using VectorBT enhancements.

        Args:
            objective_function: Function to evaluate parameters
            parameters: Parameters to evaluate
            data: Optional data for optimization context

        Returns:
            Evaluation result
        """
        async def _optimized_evaluation():
            # Use VectorBTRollingOptimizer for data preprocessing if data is available
            if data is not None and len(data) > 1000:
                # Calculate rolling metrics using VectorBTRollingOptimizer
                rolling_metrics = await self.calculate_rolling_metrics_enhanced(data)
                technical_indicators = await self.calculate_technical_indicators_enhanced(data)

                # Add to parameters for the objective function
                enhanced_parameters = parameters.copy()
                enhanced_parameters['rolling_metrics'] = rolling_metrics
                enhanced_parameters['technical_indicators'] = technical_indicators
                enhanced_parameters['vectorbt_optimized'] = True

                return await objective_function(enhanced_parameters)
            else:
                return await objective_function(parameters)

        result = await self.execute_operation(
            VectorBTOperationType.PARAMETER_OPTIMIZATION,
            _optimized_evaluation
        )

        return result.result if result.success else None

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        stats = self.performance_stats.copy()

        if stats['total_operations'] > 0:
            stats['success_rate'] = stats['successful_operations'] / stats['total_operations']
            stats['average_execution_time'] = stats['total_execution_time'] / stats['total_operations']
            stats['memory_efficiency'] = stats['total_memory_used'] / stats['total_operations']

        stats['cache_hit_rate'] = self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0
        stats['cache_size'] = len(self.operation_cache)

        return stats

    def _generate_cache_key(self, operation_type: VectorBTOperationType,
                          operation_func: Callable, args: tuple, kwargs: dict) -> str:
        """Generate a cache key for the operation."""
        # Simple cache key based on operation type and function name
        key_parts = [
            operation_type.value,
            operation_func.__name__,
            str(hash(str(args))),
            str(hash(str(sorted(kwargs.items()))))
        ]
        return "_".join(key_parts)

    def _should_cache_result(self, operation_type: VectorBTOperationType,
                           execution_time: float) -> bool:
        """Determine if a result should be cached."""
        # Cache expensive operations
        expensive_operations = {
            VectorBTOperationType.BOOTSTRAP_SAMPLING,
            VectorBTOperationType.REGIME_ANALYSIS,
            VectorBTOperationType.CORRELATION_ANALYSIS
        }

        return (operation_type in expensive_operations or
                execution_time > 1.0)  # Cache operations taking more than 1 second

    def _evict_oldest_cache_entry(self):
        """Evict the oldest cache entry."""
        if self.operation_cache:
            oldest_key = next(iter(self.operation_cache))
            del self.operation_cache[oldest_key]

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0

    def _update_performance_stats(self, result: VectorBTOperationResult):
        """Update performance statistics."""
        self.performance_stats['total_operations'] += 1
        self.performance_stats['total_execution_time'] += result.execution_time
        self.performance_stats['total_memory_used'] += result.memory_used_mb

        if result.success:
            self.performance_stats['successful_operations'] += 1
        else:
            self.performance_stats['failed_operations'] += 1

        # Update operation type stats
        op_type = result.operation_type.value
        if op_type not in self.performance_stats['operations_by_type']:
            self.performance_stats['operations_by_type'][op_type] = 0
        self.performance_stats['operations_by_type'][op_type] += 1

# Global manager instance
_global_manager = None

def get_vectorbt_unified_manager(config: VectorBTConfig = None) -> VectorBTUnifiedManager:
    """Get the global VectorBT unified manager instance."""
    global _global_manager
    if _global_manager is None:
        if config is None:
            config = VectorBTConfig()
        _global_manager = VectorBTUnifiedManager(config)
    return _global_manager

def reset_vectorbt_unified_manager():
    """Reset the global VectorBT unified manager."""
    global _global_manager
    _global_manager = None
