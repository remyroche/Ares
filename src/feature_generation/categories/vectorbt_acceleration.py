"""
VectorBT-Optimized Acceleration Feature Generators

This module provides high-performance acceleration feature generators using VectorBT's
optimized C++ backend for maximum performance in feature generation.

Features:
- Price momentum
- Price acceleration
- Price jerk
- Trend strength
- Trend consistency
- Volume acceleration
- Volatility acceleration
- Momentum acceleration
- Acceleration momentum
- Acceleration volatility
- Acceleration trend strength
- Acceleration consistency
- Acceleration regime detection
- Multi-timeframe acceleration
- Cross-asset acceleration
- Acceleration correlation
- Acceleration divergence
"""

import numpy as np
import pandas as pd
import logging
from typing import List, Optional, Dict, Any, Union
from scipy import stats
from src.utils.tprint import tprint

from ..core.vectorbt_feature_generator import VectorBTFeatureGenerator
from ..core.feature_generator import FeatureConfig, FeatureCategory, VectorizedFeatureGenerator, FeatureGenerator
from ..base_calculations import BaseCalculationType, create_base_calculator
from src.feature_generation.utils.math_validation import safe_divide, validate_finite, safe_percentage_change

# VectorBT Rolling Optimizer
try:
    from src.feature_generation.utils.vectorbt_rolling_optimizer import get_vectorbt_rolling_optimizer, VectorBTRollingOptimizer
    VECTORBT_ROLLING_OPTIMIZER_AVAILABLE = True
except ImportError:
    VECTORBT_ROLLING_OPTIMIZER_AVAILABLE = False
    VectorBTRollingOptimizer = None

# Unified Vectorization Manager
try:
    from ...utils.ml_common.unified_vectorization_manager import get_unified_vectorization_manager, UnifiedVectorizationManager, OperationType, OptimizationStrategy
    UNIFIED_VECTORIZATION_AVAILABLE = True
except ImportError:
    UNIFIED_VECTORIZATION_AVAILABLE = False
    UnifiedVectorizationManager = None

# Additional optimization utilities for AccelerationFeatureGenerator
try:
    from src.feature_generation.utils.vectorization_optimizer import get_vectorization_optimizer
    from src.feature_generation.utils.optimized_feature_pipeline import get_optimized_feature_pipeline
    OPTIMIZATION_AVAILABLE = True
except ImportError:
    OPTIMIZATION_AVAILABLE = False

# VectorBT availability check
try:
    import vectorbt as vbt
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False

logger = logging.getLogger(__name__)


def _sanitize_series_output(values: Any,
                            index: pd.Index,
                            name: str,
                            fill_value: float = 0.0) -> pd.Series:
    """
    Convert arbitrary outputs into a numeric pandas Series suitable for downstream validation.
    """
    try:
        if isinstance(values, pd.Series):
            series = values.copy()
        elif isinstance(values, pd.DataFrame):
            series = values.iloc[:, 0].copy() if not values.empty else pd.Series(dtype=float, index=index)
        elif isinstance(values, dict):
            # Prefer common keys
            for key in ['result', 'trend_strength', 'momentum', 'data']:
                if key in values:
                    return _sanitize_series_output(values[key], index, name, fill_value)
            # Fallback: attempt to build Series from dict values
            series = pd.Series(values)
        elif isinstance(values, (np.ndarray, list, tuple)):
            series = pd.Series(values)
        elif hasattr(values, '__iter__') and not np.isscalar(values):
            series = pd.Series(list(values))
        else:
            series = pd.Series([values] * len(index), index=index)

        if len(series) != len(index):
            series = series.reindex(index, fill_value=fill_value)
        else:
            series.index = index
        series = pd.to_numeric(series, errors='coerce')
        series = series.replace([np.inf, -np.inf], np.nan)
        series = series.fillna(fill_value)
        series.name = name
        return series
    except Exception as e:
        logger.warning(f"Failed to sanitize series output for {name}: {e}")
        return pd.Series(fill_value, index=index, name=name)

class AccelerationFeatureGenerator(VectorizedFeatureGenerator):
    """Feature generator for acceleration-based features with full VectorBT optimization."""

    def __init__(self, config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config()
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)

        # Initialize VectorBT optimization components
        self.vectorbt_rolling_optimizer = None
        self.unified_manager = None

        if VECTORBT_ROLLING_OPTIMIZER_AVAILABLE:
            self.vectorbt_rolling_optimizer = get_vectorbt_rolling_optimizer(enable_gpu=True, enable_parallel=True)

        if UNIFIED_VECTORIZATION_AVAILABLE:
            self.unified_manager = get_unified_vectorization_manager()

        # Performance monitoring
        self.performance_metrics = {
            'total_features_generated': 0,
            'vectorbt_operations': 0,
            'unified_manager_operations': 0,
            'fallback_operations': 0,
            'total_processing_time': 0.0,
            'average_feature_time': 0.0,
            'memory_usage_mb': 0.0,
            'optimization_success_rate': 0.0
        }

    def _apply_gpu_optimizations(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply GPU-specific optimizations to data."""
        try:
            # Convert to optimal data types for GPU processing
            for column in data.columns:
                if data[column].dtype == 'float64':
                    # Use float32 for GPU efficiency
                    data[column] = data[column].astype(np.float32)
                elif data[column].dtype == 'int64':
                    # Use int32 for GPU efficiency
                    data[column] = data[column].astype(np.int32)
            return data
        except Exception as e:
            self.logger.warning(f"GPU optimization failed: {e}")
            return data

    def _update_performance_metrics(self, operation_type: str, processing_time: float, memory_usage: float = 0.0):
        """Update performance metrics for monitoring."""
        self.performance_metrics['total_features_generated'] += 1
        self.performance_metrics['total_processing_time'] += processing_time

        if operation_type == 'vectorbt':
            self.performance_metrics['vectorbt_operations'] += 1
        elif operation_type == 'unified_manager':
            self.performance_metrics['unified_manager_operations'] += 1
        else:
            self.performance_metrics['fallback_operations'] += 1

        # Update average processing time
        total_features = self.performance_metrics['total_features_generated']
        self.performance_metrics['average_feature_time'] = (
            self.performance_metrics['total_processing_time'] / total_features
        )

        # Update memory usage
        self.performance_metrics['memory_usage_mb'] = max(
            self.performance_metrics['memory_usage_mb'], memory_usage
        )

        # Update optimization success rate
        total_optimized = (self.performance_metrics['vectorbt_operations'] +
                          self.performance_metrics['unified_manager_operations'])
        self.performance_metrics['optimization_success_rate'] = (
            total_optimized / total_features if total_features > 0 else 0.0
        )

    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        report = self.performance_metrics.copy()

        # Add efficiency metrics
        if self.performance_metrics['total_features_generated'] > 0:
            report['efficiency_metrics'] = {
                'vectorbt_usage_rate': (
                    self.performance_metrics['vectorbt_operations'] /
                    self.performance_metrics['total_features_generated']
                ),
                'unified_manager_usage_rate': (
                    self.performance_metrics['unified_manager_operations'] /
                    self.performance_metrics['total_features_generated']
                ),
                'fallback_rate': (
                    self.performance_metrics['fallback_operations'] /
                    self.performance_metrics['total_features_generated']
                )
            }

        # Add VectorBT optimizer stats if available
        if self.vectorbt_rolling_optimizer and hasattr(self.vectorbt_rolling_optimizer, 'get_performance_stats'):
            report['vectorbt_optimizer_stats'] = self.vectorbt_rolling_optimizer.get_performance_stats()

        # Add UnifiedVectorizationManager stats if available
        if self.unified_manager and hasattr(self.unified_manager, 'get_optimization_stats'):
            report['unified_manager_stats'] = self.unified_manager.get_optimization_stats()

        return report

    def reset_performance_metrics(self):
        """Reset performance metrics."""
        self.performance_metrics = {
            'total_features_generated': 0,
            'vectorbt_operations': 0,
            'unified_manager_operations': 0,
            'fallback_operations': 0,
            'total_processing_time': 0.0,
            'average_feature_time': 0.0,
            'memory_usage_mb': 0.0,
            'optimization_success_rate': 0.0
        }

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate acceleration-based features using VectorBT optimization."""
        if len(data) == 0:
            return pd.Series(dtype=float, index=data.index, name='acceleration_feature')
        
        # Default implementation - calculate simple price acceleration
        if 'close' in data.columns:
            close = data['close']
            # Calculate second derivative (acceleration) of price using safe_diff
            from ...utils.error_handling import safe_diff
            price_change = safe_diff(close)
            acceleration = safe_diff(price_change)
            return acceleration.fillna(0)
        else:
            # Fallback for data without close price
            return pd.Series(0, index=data.index, name='acceleration_feature')

    @classmethod
    def _create_default_config(cls) -> FeatureConfig:
        return FeatureConfig(
            name="acceleration_features",
            category=FeatureCategory.ACCELERATION,
            description="Comprehensive acceleration features including momentum, acceleration, and jerk",
            required_columns=["close"],
            optional_columns=["high", "low", "open", "volume"],
            default_lookback=20,
            min_lookback=5,
            max_lookback=100,
            parameters={
                "acceleration_windows": [5, 10, 20],
                "momentum_windows": [5, 10, 20, 50]
            },
            matrix_optimized=True,
            gpu_accelerated=False
        )

class VectorBTMomentumGenerator(VectorBTFeatureGenerator):
    """VectorBT-optimized momentum generator with full optimization."""

    def __init__(self, period: int = 10, base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS, **base_kwargs):
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)

        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        required_columns = self.base_calculator.get_required_columns()

        config = FeatureConfig(
            name=f"vectorbt_momentum_{period}_{base_calculation.value}",
            category=FeatureCategory.ACCELERATION,
            description=f"VectorBT-optimized momentum over {period} periods based on {base_calculation.value}",
            required_columns=required_columns,
            optional_columns=["high", "low", "open", "volume"],
            default_lookback=period,
            min_lookback=period,
            max_lookback=period,
            parameters={'period': period, 'base_calculation': base_calculation.value, **base_kwargs},
            matrix_optimized=True,
            gpu_accelerated=True
        )
        super().__init__(config)
        self.period = period
        self.base_calculation = base_calculation
        self.use_unified_manager = False

        # Initialize VectorBT optimization components
        self.vectorbt_rolling_optimizer = None
        self.unified_manager = None

        if VECTORBT_ROLLING_OPTIMIZER_AVAILABLE:
            self.vectorbt_rolling_optimizer = get_vectorbt_rolling_optimizer(enable_gpu=True, enable_parallel=True)

        if UNIFIED_VECTORIZATION_AVAILABLE:
            self.unified_manager = get_unified_vectorization_manager()

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate momentum using VectorBT operations with full optimization."""
        tprint(f"Generating VectorBT momentum feature with period {self.period} and base calculation {self.base_calculation.value}")

        if len(data) == 0:
            tprint("Warning: Empty data provided for momentum calculation")
            return pd.Series(dtype=float, index=data.index, name=f'vectorbt_momentum_{self.period}_{self.base_calculation.value}')

        base_values = pd.to_numeric(self.base_calculator.calculate(data), errors='coerce')

        # Enhanced VectorBT optimization with UnifiedVectorizationManager
        if self.use_unified_manager and self.unified_manager and UNIFIED_VECTORIZATION_AVAILABLE:
            try:
                from ...utils.ml_common.unified_vectorization_manager import OperationConfig
                config = OperationConfig(
                    operation_type=OperationType.FEATURE_ENGINEERING,
                    data_size=len(base_values),
                    data_dimensions=base_values.shape,
                    memory_budget_mb=2048.0,
                    time_budget_seconds=60.0,
                    precision_requirement="high"
                )

                # Use UnifiedVectorizationManager for momentum calculation
                result = self.unified_manager.optimize_operation(
                    OperationType.FEATURE_ENGINEERING,
                    {'data': base_values, 'period': self.period, 'operation': 'momentum'},
                    config
                )
                momentum = result.result

                # Validate finite values
                try:
                    validate_finite(momentum.values, f"VectorBT_Momentum_{self.period}_{self.base_calculation.value}")
                except ValueError as e:
                    logger.warning(f"⚠️ {e}")

                return _sanitize_series_output(
                    momentum, data.index,
                    f'vectorbt_momentum_{self.period}_{self.base_calculation.value}'
                )

            except Exception as e:
                logger.warning(f"UnifiedVectorizationManager momentum calculation failed: {e}")

        # Use VectorBTRollingOptimizer for enhanced performance
        if self.vectorbt_rolling_optimizer and VECTORBT_ROLLING_OPTIMIZER_AVAILABLE:
            try:
                # Use VectorBT rolling apply for momentum calculation
                def momentum_func(series):
                    if len(series) < self.period + 1:
                        return np.nan
                    current = series.iloc[-1]
                    past = series.iloc[-self.period-1]
                    return safe_percentage_change(past, current)

                momentum = self.vectorbt_rolling_optimizer.rolling_apply(
                    base_values, momentum_func, window=self.period + 1
                )

                # Validate finite values
                try:
                    validate_finite(momentum.values, f"VectorBT_Momentum_{self.period}_{self.base_calculation.value}")
                except ValueError as e:
                    logger.warning(f"⚠️ {e}")

                return _sanitize_series_output(
                    momentum, data.index,
                    f'vectorbt_momentum_{self.period}_{self.base_calculation.value}'
                )

            except Exception as e:
                logger.warning(f"VectorBTRollingOptimizer momentum calculation failed: {e}, using fallback")

        # Fallback to standard VectorBT operations
        shifted_values = base_values.shift(self.period)
        momentum = safe_percentage_change(shifted_values, base_values)

        # Validate finite values
        try:
            validate_finite(momentum.values, f"VectorBT_Momentum_{self.period}_{self.base_calculation.value}")
        except ValueError as e:
            logger.warning(f"⚠️ {e}")

        return _sanitize_series_output(
            momentum, data.index,
            f'vectorbt_momentum_{self.period}_{self.base_calculation.value}'
        )

class VectorBTPriceAccelerationGenerator(VectorBTFeatureGenerator):
    """VectorBT-optimized price acceleration generator."""

    def __init__(self, period: int = 10, base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS, **base_kwargs):
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)

        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        required_columns = self.base_calculator.get_required_columns()

        config = FeatureConfig(
            name=f"vectorbt_acceleration_{period}_{base_calculation.value}",
            category=FeatureCategory.ACCELERATION,
            description=f"VectorBT-optimized acceleration over {period} periods based on {base_calculation.value}",
            required_columns=required_columns,
            optional_columns=["high", "low", "open", "volume"],
            default_lookback=period * 2,
            min_lookback=period * 2,
            max_lookback=period * 2,
            parameters={'period': period, 'base_calculation': base_calculation.value, **base_kwargs},
            matrix_optimized=True,
            gpu_accelerated=True
        )
        super().__init__(config)
        self.period = period
        self.base_calculation = base_calculation
        self.use_unified_manager = False

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate acceleration using VectorBT operations."""
        tprint(f"Generating VectorBT acceleration feature with period {self.period} and base calculation {self.base_calculation.value}")

        if len(data) == 0:
            tprint("Warning: Empty data provided for acceleration calculation")
            return pd.Series(dtype=float, index=data.index, name=f'vectorbt_acceleration_{self.period}_{self.base_calculation.value}')

        base_values = pd.to_numeric(self.base_calculator.calculate(data), errors='coerce')

        # Calculate momentum first
        shifted_values = base_values.shift(self.period)
        momentum = safe_percentage_change(shifted_values, base_values)

        # Calculate acceleration (second derivative) using VectorBT
        acceleration = momentum.diff(self.period)

        # Validate finite values
        try:
            validate_finite(acceleration.values, f"VectorBT_Acceleration_{self.period}_{self.base_calculation.value}")
        except ValueError as e:
            logger.warning(f"⚠️ {e}")

        return _sanitize_series_output(
            acceleration, data.index,
            f'vectorbt_acceleration_{self.period}_{self.base_calculation.value}'
        )

class VectorBTPriceJerkGenerator(VectorBTFeatureGenerator):
    """VectorBT-optimized price jerk generator."""

    def __init__(self, period: int = 10, base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS, **base_kwargs):
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)

        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        required_columns = self.base_calculator.get_required_columns()

        config = FeatureConfig(
            name=f"vectorbt_jerk_{period}_{base_calculation.value}",
            category=FeatureCategory.ACCELERATION,
            description=f"VectorBT-optimized jerk over {period} periods based on {base_calculation.value}",
            required_columns=required_columns,
            optional_columns=["high", "low", "open", "volume"],
            default_lookback=period * 3,
            min_lookback=period * 3,
            max_lookback=period * 3,
            parameters={'period': period, 'base_calculation': base_calculation.value, **base_kwargs},
            matrix_optimized=True,
            gpu_accelerated=True
        )
        super().__init__(config)
        self.period = period
        self.base_calculation = base_calculation
        self.use_unified_manager = False

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate jerk using VectorBT operations."""
        tprint(f"Generating VectorBT jerk feature with period {self.period} and base calculation {self.base_calculation.value}")

        if len(data) == 0:
            tprint("Warning: Empty data provided for jerk calculation")
            return pd.Series(dtype=float, index=data.index, name=f'vectorbt_jerk_{self.period}_{self.base_calculation.value}')

        base_values = pd.to_numeric(self.base_calculator.calculate(data), errors='coerce')

        # Calculate momentum first
        shifted_values = base_values.shift(self.period)
        momentum = safe_percentage_change(shifted_values, base_values)

        # Calculate acceleration (second derivative)
        acceleration = momentum.diff(self.period)

        # Calculate jerk (third derivative) using VectorBT
        jerk = acceleration.diff(self.period)

        # Validate finite values
        try:
            validate_finite(jerk.values, f"VectorBT_Jerk_{self.period}_{self.base_calculation.value}")
        except ValueError as e:
            logger.warning(f"⚠️ {e}")

        return _sanitize_series_output(
            jerk, data.index,
            f'vectorbt_jerk_{self.period}_{self.base_calculation.value}'
        )

class VectorBTTrendStrengthGenerator(VectorBTFeatureGenerator):
    """VectorBT-optimized trend strength generator."""

    def __init__(self, window: int = 20, base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS, **base_kwargs):
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)

        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        required_columns = self.base_calculator.get_required_columns()

        config = FeatureConfig(
            name=f"vectorbt_trend_strength_{window}_{base_calculation.value}",
            category=FeatureCategory.ACCELERATION,
            description=f"VectorBT-optimized trend strength over {window} periods based on {base_calculation.value}",
            required_columns=required_columns,
            optional_columns=["high", "low", "open", "volume"],
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={'window': window, 'base_calculation': base_calculation.value, **base_kwargs},
            matrix_optimized=True,
            gpu_accelerated=True
        )
        super().__init__(config)
        self.window = window
        self.base_calculation = base_calculation
        self.use_unified_manager = False

        # Initialize vectorbt optimizer
        try:
            from ...utils.vectorbt_rolling_optimizer import get_vectorbt_rolling_optimizer
            self.vectorbt_rolling_optimizer = get_vectorbt_rolling_optimizer()
        except ImportError:
            self.vectorbt_rolling_optimizer = None

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate trend strength using VectorBT operations."""
        if len(data) == 0:
            return pd.Series(dtype=float, index=data.index, name=f'vectorbt_trend_strength_{self.window}_{self.base_calculation.value}')

        base_values = pd.to_numeric(self.base_calculator.calculate(data), errors='coerce')

        # Enhanced VectorBT optimization with UnifiedVectorizationManager
        if self.use_unified_manager and self.unified_manager and UNIFIED_VECTORIZATION_AVAILABLE:
            try:
                from ...utils.ml_common.unified_vectorization_manager import OperationConfig
                config = OperationConfig(
                    operation_type=OperationType.FEATURE_ENGINEERING,
                    data_size=len(base_values),
                    data_dimensions=base_values.shape,
                    memory_budget_mb=1024.0,
                    precision_requirement="high"
                )

                # Use UnifiedVectorizationManager for trend strength calculation
                result = self.unified_manager.optimize_operation(
                    OperationType.FEATURE_ENGINEERING,
                    {'data': base_values, 'window': self.window, 'operation': 'trend_strength'},
                    config
                )
                # Convert result to Series if it's a dict
                if isinstance(result.result, dict):
                    # Extract the main result from the dict
                    trend_strength = result.result.get('trend_strength', result.result.get('result', result.result))
                    if isinstance(trend_strength, pd.Series):
                        return _sanitize_series_output(
                            trend_strength, data.index,
                            f'vectorbt_trend_strength_{self.window}_{self.base_calculation.value}'
                        )
                    else:
                        # Convert to Series if needed
                        return _sanitize_series_output(
                            trend_strength, data.index,
                            f'vectorbt_trend_strength_{self.window}_{self.base_calculation.value}'
                        )
                else:
                    return _sanitize_series_output(
                        result.result, data.index,
                        f'vectorbt_trend_strength_{self.window}_{self.base_calculation.value}'
                    )

            except Exception as e:
                logger.warning(f"UnifiedVectorizationManager trend strength failed: {e}")

        # Use VectorBTRollingOptimizer for enhanced performance
        if self.vectorbt_rolling_optimizer and VECTORBT_ROLLING_OPTIMIZER_AVAILABLE:
            try:
                # Calculate trend strength using linear regression slope
                def calculate_trend_strength(series):
                    if len(series) < 2:
                        return 0.0
                    try:
                        # Remove NaN values for calculation
                        valid_data = series.dropna()
                        if len(valid_data) < 2:
                            return 0.0

                        # Use linear regression to calculate trend strength
                        x = np.arange(len(valid_data))
                        y = valid_data.values
                        # Calculate R-squared to measure trend strength
                        if np.var(y) == 0:  # Avoid division by zero
                            return 0.0
                        slope, intercept = np.polyfit(x, y, 1)
                        y_pred = slope * x + intercept
                        ss_res = np.sum((y - y_pred) ** 2)
                        ss_tot = np.sum((y - np.mean(y)) ** 2)
                        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
                        return r_squared
                    except:
                        return 0.0

                # Use a more reasonable min_periods - require at least 2 data points for trend calculation
                min_periods = max(2, min(5, self.window // 4))  # More flexible min_periods
                trend_strength = self.vectorbt_rolling_optimizer.rolling_apply(
                    base_values, calculate_trend_strength, window=self.window, min_periods=min_periods
                )
                return _sanitize_series_output(
                    trend_strength, data.index,
                    f'vectorbt_trend_strength_{self.window}_{self.base_calculation.value}'
                )
            except Exception as e:
                logger.warning(f"VectorBTRollingOptimizer trend strength failed: {e}")

        # Fallback to pandas rolling if VectorBT optimizer is not available
        min_periods = max(2, min(5, self.window // 4))  # More flexible min_periods
        try:
            # Define the function locally for the fallback
            def calculate_trend_strength_fallback(series):
                if len(series) < 2:
                    return 0.0
                try:
                    # Remove NaN values for calculation
                    valid_data = series.dropna()
                    if len(valid_data) < 2:
                        return 0.0

                    # Use linear regression to calculate trend strength
                    x = np.arange(len(valid_data))
                    y = valid_data.values
                    # Calculate R-squared to measure trend strength
                    if np.var(y) == 0:  # Avoid division by zero
                        return 0.0
                    slope, intercept = np.polyfit(x, y, 1)
                    y_pred = slope * x + intercept
                    ss_res = np.sum((y - y_pred) ** 2)
                    ss_tot = np.sum((y - np.mean(y)) ** 2)
                    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
                    return r_squared
                except:
                    return 0.0
            
            trend_strength = base_values.rolling(window=self.window, min_periods=min_periods).apply(
                calculate_trend_strength_fallback, raw=False
            )
            return _sanitize_series_output(
                trend_strength, data.index,
                f'vectorbt_trend_strength_{self.window}_{self.base_calculation.value}'
            )
        except Exception as e:
            logger.error(f"Pandas fallback also failed: {e}")
            # Return a series of zeros as last resort
            return pd.Series(0.0, index=base_values.index, name=f'vectorbt_trend_strength_{self.window}_{self.base_calculation.value}')

class VectorBTTrendConsistencyGenerator(VectorBTFeatureGenerator):
    """VectorBT-optimized trend consistency generator."""

    def __init__(self, window: int = 20, base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS, **base_kwargs):
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)

        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        required_columns = self.base_calculator.get_required_columns()

        config = FeatureConfig(
            name=f"vectorbt_trend_consistency_{window}_{base_calculation.value}",
            category=FeatureCategory.ACCELERATION,
            description=f"VectorBT-optimized trend consistency over {window} periods based on {base_calculation.value}",
            required_columns=required_columns,
            optional_columns=["high", "low", "open", "volume"],
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={'window': window, 'base_calculation': base_calculation.value, **base_kwargs},
            matrix_optimized=True,
            gpu_accelerated=True
        )
        super().__init__(config)
        self.window = window
        self.base_calculation = base_calculation

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate trend consistency using VectorBT operations."""
        if len(data) == 0:
            return pd.Series(dtype=float, index=data.index, name=f'vectorbt_trend_consistency_{self.window}_{self.base_calculation.value}')

        base_values = self.base_calculator.calculate(data)

        # Calculate trend consistency as inverse of volatility
        volatility = self._vectorbt_rolling_operation(base_values, 'std', window=self.window)
        consistency = 1.0 / (volatility + 1e-8)  # Add small epsilon to avoid division by zero

        return _sanitize_series_output(
            consistency, data.index,
            f'vectorbt_trend_consistency_{self.window}_{self.base_calculation.value}',
            fill_value=0.0
        )

class VectorBTVolumeAccelerationGenerator(VectorBTFeatureGenerator):
    """VectorBT-optimized volume acceleration generator."""

    def __init__(self, period: int = 5, base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.VOLUME_RETURNS, **base_kwargs):
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)

        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        required_columns = self.base_calculator.get_required_columns()

        config = FeatureConfig(
            name=f"vectorbt_volume_acceleration_{period}_{base_calculation.value}",
            category=FeatureCategory.ACCELERATION,
            description=f"VectorBT-optimized volume acceleration over {period} periods based on {base_calculation.value}",
            required_columns=required_columns,
            optional_columns=["high", "low", "open", "close"],
            default_lookback=period * 2,
            min_lookback=period * 2,
            max_lookback=period * 2,
            parameters={'period': period, 'base_calculation': base_calculation.value, **base_kwargs},
            matrix_optimized=True,
            gpu_accelerated=True
        )
        super().__init__(config)
        self.period = period
        self.base_calculation = base_calculation

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate volume acceleration using VectorBT operations."""
        if len(data) == 0:
            return pd.Series(dtype=float, index=data.index, name=f'vectorbt_volume_acceleration_{self.period}_{self.base_calculation.value}')

        base_values = self.base_calculator.calculate(data)

        # Calculate volume acceleration using VectorBT
        volume_acceleration = base_values.diff(self.period).diff(self.period)

        return volume_acceleration.rename(f'vectorbt_volume_acceleration_{self.period}_{self.base_calculation.value}')

class VectorBTVolatilityAccelerationGenerator(VectorBTFeatureGenerator):
    """VectorBT-optimized volatility acceleration generator."""

    def __init__(self, period: int = 5, volatility_window: int = 20, base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS, **base_kwargs):
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)

        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        required_columns = self.base_calculator.get_required_columns()

        config = FeatureConfig(
            name=f"vectorbt_volatility_acceleration_{period}_{volatility_window}_{base_calculation.value}",
            category=FeatureCategory.ACCELERATION,
            description=f"VectorBT-optimized volatility acceleration over {period} periods with {volatility_window} volatility window based on {base_calculation.value}",
            required_columns=required_columns,
            optional_columns=["high", "low", "open", "volume"],
            default_lookback=volatility_window + period * 2,
            min_lookback=volatility_window + period * 2,
            max_lookback=volatility_window + period * 2,
            parameters={'period': period, 'volatility_window': volatility_window, 'base_calculation': base_calculation.value, **base_kwargs},
            matrix_optimized=True,
            gpu_accelerated=True
        )
        super().__init__(config)
        self.period = period
        self.volatility_window = volatility_window
        self.base_calculation = base_calculation

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate volatility acceleration using VectorBT operations."""
        if len(data) == 0:
            return pd.Series(dtype=float, index=data.index, name=f'vectorbt_volatility_acceleration_{self.period}_{self.volatility_window}_{self.base_calculation.value}')

        base_values = self.base_calculator.calculate(data)

        # Calculate volatility using VectorBT
        volatility = self._vectorbt_rolling_operation(base_values, 'std', window=self.volatility_window)

        # Calculate volatility acceleration using VectorBT
        volatility_acceleration = volatility.diff(self.period).diff(self.period)

        return volatility_acceleration.rename(f'vectorbt_volatility_acceleration_{self.period}_{self.volatility_window}_{self.base_calculation.value}')

class VectorBTMomentumAccelerationGenerator(VectorBTFeatureGenerator):
    """VectorBT-optimized momentum acceleration generator."""

    def __init__(self, period: int = 10, momentum_window: int = 20, base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS, **base_kwargs):
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)

        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        required_columns = self.base_calculator.get_required_columns()

        config = FeatureConfig(
            name=f"vectorbt_momentum_acceleration_{period}_{momentum_window}_{base_calculation.value}",
            category=FeatureCategory.ACCELERATION,
            description=f"VectorBT-optimized momentum acceleration over {period} periods with {momentum_window} momentum window based on {base_calculation.value}",
            required_columns=required_columns,
            optional_columns=["high", "low", "open", "volume"],
            default_lookback=momentum_window + period * 2,
            min_lookback=momentum_window + period * 2,
            max_lookback=momentum_window + period * 2,
            parameters={'period': period, 'momentum_window': momentum_window, 'base_calculation': base_calculation.value, **base_kwargs},
            matrix_optimized=True,
            gpu_accelerated=True
        )
        super().__init__(config)
        self.period = period
        self.momentum_window = momentum_window
        self.base_calculation = base_calculation

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate momentum acceleration using VectorBT operations."""
        if len(data) == 0:
            return pd.Series(dtype=float, index=data.index, name=f'vectorbt_momentum_acceleration_{self.period}_{self.momentum_window}_{self.base_calculation.value}')

        base_values = self.base_calculator.calculate(data)

        # Calculate momentum using VectorBT
        momentum = self._vectorbt_rolling_operation(base_values, 'mean', window=self.momentum_window)

        # Calculate momentum acceleration using VectorBT
        momentum_acceleration = momentum.diff(self.period).diff(self.period)

        return momentum_acceleration.rename(f'vectorbt_momentum_acceleration_{self.period}_{self.momentum_window}_{self.base_calculation.value}')

class VectorBTAccelerationMomentumGenerator(VectorBTFeatureGenerator):
    """VectorBT-optimized acceleration momentum generator."""

    def __init__(self, period: int = 10, acceleration_window: int = 20, base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS, **base_kwargs):
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)

        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        required_columns = self.base_calculator.get_required_columns()

        config = FeatureConfig(
            name=f"vectorbt_acceleration_momentum_{period}_{acceleration_window}_{base_calculation.value}",
            category=FeatureCategory.ACCELERATION,
            description=f"VectorBT-optimized acceleration momentum over {period} periods with {acceleration_window} acceleration window based on {base_calculation.value}",
            required_columns=required_columns,
            optional_columns=["high", "low", "open", "volume"],
            default_lookback=acceleration_window + period * 2,
            min_lookback=acceleration_window + period * 2,
            max_lookback=acceleration_window + period * 2,
            parameters={'period': period, 'acceleration_window': acceleration_window, 'base_calculation': base_calculation.value, **base_kwargs},
            matrix_optimized=True,
            gpu_accelerated=True
        )
        super().__init__(config)
        self.period = period
        self.acceleration_window = acceleration_window
        self.base_calculation = base_calculation

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate acceleration momentum using VectorBT operations."""
        if len(data) == 0:
            return pd.Series(dtype=float, index=data.index, name=f'vectorbt_acceleration_momentum_{self.period}_{self.acceleration_window}_{self.base_calculation.value}')

        base_values = self.base_calculator.calculate(data)

        # Calculate acceleration first
        shifted_values = base_values.shift(self.acceleration_window)
        momentum = safe_percentage_change(shifted_values, base_values)
        acceleration = momentum.diff(self.acceleration_window)

        # Calculate acceleration momentum using VectorBT
        acceleration_momentum = self._vectorbt_rolling_operation(acceleration, 'mean', window=self.period)

        return acceleration_momentum.rename(f'vectorbt_acceleration_momentum_{self.period}_{self.acceleration_window}_{self.base_calculation.value}')

class VectorBTAccelerationVolatilityGenerator(VectorBTFeatureGenerator):
    """VectorBT-optimized acceleration volatility generator."""

    def __init__(self, period: int = 10, acceleration_window: int = 20, base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS, **base_kwargs):
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)

        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        required_columns = self.base_calculator.get_required_columns()

        config = FeatureConfig(
            name=f"vectorbt_acceleration_volatility_{period}_{acceleration_window}_{base_calculation.value}",
            category=FeatureCategory.ACCELERATION,
            description=f"VectorBT-optimized acceleration volatility over {period} periods with {acceleration_window} acceleration window based on {base_calculation.value}",
            required_columns=required_columns,
            optional_columns=["high", "low", "open", "volume"],
            default_lookback=acceleration_window + period * 2,
            min_lookback=acceleration_window + period * 2,
            max_lookback=acceleration_window + period * 2,
            parameters={'period': period, 'acceleration_window': acceleration_window, 'base_calculation': base_calculation.value, **base_kwargs},
            matrix_optimized=True,
            gpu_accelerated=True
        )
        super().__init__(config)
        self.period = period
        self.acceleration_window = acceleration_window
        self.base_calculation = base_calculation

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate acceleration volatility using VectorBT operations."""
        if len(data) == 0:
            return pd.Series(dtype=float, index=data.index, name=f'vectorbt_acceleration_volatility_{self.period}_{self.acceleration_window}_{self.base_calculation.value}')

        base_values = self.base_calculator.calculate(data)

        # Calculate acceleration first
        shifted_values = base_values.shift(self.acceleration_window)
        momentum = safe_percentage_change(shifted_values, base_values)
        acceleration = momentum.diff(self.acceleration_window)

        # Calculate acceleration volatility using VectorBT
        acceleration_volatility = self._vectorbt_rolling_operation(acceleration, 'std', window=self.period)

        return acceleration_volatility.rename(f'vectorbt_acceleration_volatility_{self.period}_{self.acceleration_window}_{self.base_calculation.value}')

class VectorBTAccelerationTrendStrengthGenerator(VectorBTFeatureGenerator):
    """VectorBT-optimized acceleration trend strength generator."""

    def __init__(self, period: int = 10, acceleration_window: int = 20, base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS, **base_kwargs):
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)

        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        required_columns = self.base_calculator.get_required_columns()

        config = FeatureConfig(
            name=f"vectorbt_acceleration_trend_strength_{period}_{acceleration_window}_{base_calculation.value}",
            category=FeatureCategory.ACCELERATION,
            description=f"VectorBT-optimized acceleration trend strength over {period} periods with {acceleration_window} acceleration window based on {base_calculation.value}",
            required_columns=required_columns,
            optional_columns=["high", "low", "open", "volume"],
            default_lookback=acceleration_window + period * 2,
            min_lookback=acceleration_window + period * 2,
            max_lookback=acceleration_window + period * 2,
            parameters={'period': period, 'acceleration_window': acceleration_window, 'base_calculation': base_calculation.value, **base_kwargs},
            matrix_optimized=True,
            gpu_accelerated=True
        )
        super().__init__(config)
        self.period = period
        self.acceleration_window = acceleration_window
        self.base_calculation = base_calculation

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate acceleration trend strength using VectorBT operations."""
        if len(data) == 0:
            return pd.Series(dtype=float, index=data.index, name=f'vectorbt_acceleration_trend_strength_{self.period}_{self.acceleration_window}_{self.base_calculation.value}')

        base_values = self.base_calculator.calculate(data)

        # Calculate acceleration first
        shifted_values = base_values.shift(self.acceleration_window)
        momentum = safe_percentage_change(shifted_values, base_values)
        acceleration = momentum.diff(self.acceleration_window)

        # Calculate acceleration trend strength using linear regression
        def calculate_acceleration_trend_strength(series):
            if len(series) < 2:
                return 0.0
            try:
                # Remove NaN values for calculation
                valid_data = series.dropna()
                if len(valid_data) < 2:
                    return 0.0

                # Use linear regression to calculate trend strength
                x = np.arange(len(valid_data))
                y = valid_data.values
                # Calculate R-squared to measure trend strength
                if np.var(y) == 0:  # Avoid division by zero
                    return 0.0
                slope, intercept = np.polyfit(x, y, 1)
                y_pred = slope * x + intercept
                ss_res = np.sum((y - y_pred) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
                return r_squared
            except:
                return 0.0

        acceleration_trend_strength = acceleration.rolling(window=self.period, min_periods=2).apply(
            calculate_acceleration_trend_strength, raw=False
        )

        return acceleration_trend_strength.rename(f'vectorbt_acceleration_trend_strength_{self.period}_{self.acceleration_window}_{self.base_calculation.value}')

class VectorBTAccelerationConsistencyGenerator(VectorBTFeatureGenerator):
    """VectorBT-optimized acceleration consistency generator."""

    def __init__(self, period: int = 10, acceleration_window: int = 20, base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS, **base_kwargs):
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)

        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        required_columns = self.base_calculator.get_required_columns()

        config = FeatureConfig(
            name=f"vectorbt_acceleration_consistency_{period}_{acceleration_window}_{base_calculation.value}",
            category=FeatureCategory.ACCELERATION,
            description=f"VectorBT-optimized acceleration consistency over {period} periods with {acceleration_window} acceleration window based on {base_calculation.value}",
            required_columns=required_columns,
            optional_columns=["high", "low", "open", "volume"],
            default_lookback=acceleration_window + period * 2,
            min_lookback=acceleration_window + period * 2,
            max_lookback=acceleration_window + period * 2,
            parameters={'period': period, 'acceleration_window': acceleration_window, 'base_calculation': base_calculation.value, **base_kwargs},
            matrix_optimized=True,
            gpu_accelerated=True
        )
        super().__init__(config)
        self.period = period
        self.acceleration_window = acceleration_window
        self.base_calculation = base_calculation

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate acceleration consistency using VectorBT operations."""
        if len(data) == 0:
            return pd.Series(dtype=float, index=data.index, name=f'vectorbt_acceleration_consistency_{self.period}_{self.acceleration_window}_{self.base_calculation.value}')

        base_values = self.base_calculator.calculate(data)

        # Calculate acceleration first
        shifted_values = base_values.shift(self.acceleration_window)
        momentum = safe_percentage_change(shifted_values, base_values)
        acceleration = momentum.diff(self.acceleration_window)

        # Calculate acceleration consistency using VectorBT
        acceleration_volatility = self._vectorbt_rolling_operation(acceleration, 'std', window=self.period)
        acceleration_consistency = 1.0 / (acceleration_volatility + 1e-8)  # Add small epsilon to avoid division by zero

        return acceleration_consistency.rename(f'vectorbt_acceleration_consistency_{self.period}_{self.acceleration_window}_{self.base_calculation.value}')

class VectorBTAccelerationRegimeGenerator(VectorBTFeatureGenerator):
    """VectorBT-optimized acceleration regime detection generator."""

    def __init__(self, period: int = 10, acceleration_window: int = 20, base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS, **base_kwargs):
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)

        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        required_columns = self.base_calculator.get_required_columns()

        config = FeatureConfig(
            name=f"vectorbt_acceleration_regime_{period}_{acceleration_window}_{base_calculation.value}",
            category=FeatureCategory.ACCELERATION,
            description=f"VectorBT-optimized acceleration regime detection over {period} periods with {acceleration_window} acceleration window based on {base_calculation.value}",
            required_columns=required_columns,
            optional_columns=["high", "low", "open", "volume"],
            default_lookback=acceleration_window + period * 2,
            min_lookback=acceleration_window + period * 2,
            max_lookback=acceleration_window + period * 2,
            parameters={'period': period, 'acceleration_window': acceleration_window, 'base_calculation': base_calculation.value, **base_kwargs},
            matrix_optimized=True,
            gpu_accelerated=True
        )
        super().__init__(config)
        self.period = period
        self.acceleration_window = acceleration_window
        self.base_calculation = base_calculation

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate acceleration regime using VectorBT operations."""
        if len(data) == 0:
            return pd.Series(dtype=float, index=data.index, name=f'vectorbt_acceleration_regime_{self.period}_{self.acceleration_window}_{self.base_calculation.value}')

        base_values = self.base_calculator.calculate(data)

        # Calculate acceleration first
        shifted_values = base_values.shift(self.acceleration_window)
        momentum = safe_percentage_change(shifted_values, base_values)
        acceleration = momentum.diff(self.acceleration_window)

        # Calculate acceleration regime using VectorBT
        acceleration_momentum = self._vectorbt_rolling_operation(acceleration, 'mean', window=self.period)
        acceleration_volatility = self._vectorbt_rolling_operation(acceleration, 'std', window=self.period)

        # Regime classification: 1 for accelerating, -1 for decelerating, 0 for neutral
        regime = np.where(acceleration_momentum > acceleration_volatility, 1,
                         np.where(acceleration_momentum < -acceleration_volatility, -1, 0))

        return pd.Series(regime, index=data.index, name=f'vectorbt_acceleration_regime_{self.period}_{self.acceleration_window}_{self.base_calculation.value}')

class VectorBTMultiTimeframeAccelerationGenerator(VectorBTFeatureGenerator):
    """VectorBT-optimized multi-timeframe acceleration generator."""

    def __init__(self, short_period: int = 5, long_period: int = 20, base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS, **base_kwargs):
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)

        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        required_columns = self.base_calculator.get_required_columns()

        config = FeatureConfig(
            name=f"vectorbt_multi_timeframe_acceleration_{short_period}_{long_period}_{base_calculation.value}",
            category=FeatureCategory.ACCELERATION,
            description=f"VectorBT-optimized multi-timeframe acceleration with {short_period} and {long_period} periods based on {base_calculation.value}",
            required_columns=required_columns,
            optional_columns=["high", "low", "open", "volume"],
            default_lookback=long_period * 2,
            min_lookback=long_period * 2,
            max_lookback=long_period * 2,
            parameters={'short_period': short_period, 'long_period': long_period, 'base_calculation': base_calculation.value, **base_kwargs},
            matrix_optimized=True,
            gpu_accelerated=True
        )
        super().__init__(config)
        self.short_period = short_period
        self.long_period = long_period
        self.base_calculation = base_calculation

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate multi-timeframe acceleration using VectorBT operations."""
        if len(data) == 0:
            return pd.Series(dtype=float, index=data.index, name=f'vectorbt_multi_timeframe_acceleration_{self.short_period}_{self.long_period}_{self.base_calculation.value}')

        base_values = self.base_calculator.calculate(data)

        # Calculate short-term acceleration
        shifted_values_short = base_values.shift(self.short_period)
        momentum_short = safe_percentage_change(shifted_values_short, base_values)
        acceleration_short = momentum_short.diff(self.short_period)

        # Calculate long-term acceleration
        shifted_values_long = base_values.shift(self.long_period)
        momentum_long = safe_percentage_change(shifted_values_long, base_values)
        acceleration_long = momentum_long.diff(self.long_period)

        # Calculate multi-timeframe acceleration ratio
        multi_timeframe_acceleration = safe_divide(acceleration_short, acceleration_long)

        return multi_timeframe_acceleration.rename(f'vectorbt_multi_timeframe_acceleration_{self.short_period}_{self.long_period}_{self.base_calculation.value}')

class VectorBTAccelerationCorrelationGenerator(VectorBTFeatureGenerator):
    """VectorBT-optimized acceleration correlation generator."""

    def __init__(self, period: int = 20, base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS, **base_kwargs):
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)

        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        required_columns = self.base_calculator.get_required_columns()

        config = FeatureConfig(
            name=f"vectorbt_acceleration_correlation_{period}_{base_calculation.value}",
            category=FeatureCategory.ACCELERATION,
            description=f"VectorBT-optimized acceleration correlation over {period} periods based on {base_calculation.value}",
            required_columns=required_columns,
            optional_columns=["high", "low", "open", "volume"],
            default_lookback=period * 2,
            min_lookback=period * 2,
            max_lookback=period * 2,
            parameters={'period': period, 'base_calculation': base_calculation.value, **base_kwargs},
            matrix_optimized=True,
            gpu_accelerated=True
        )
        super().__init__(config)
        self.period = period
        self.base_calculation = base_calculation

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate acceleration correlation using VectorBT operations."""
        if len(data) == 0:
            return pd.Series(dtype=float, index=data.index, name=f'vectorbt_acceleration_correlation_{self.period}_{self.base_calculation.value}')

        base_values = self.base_calculator.calculate(data)

        # Calculate acceleration
        shifted_values = base_values.shift(self.period)
        momentum = safe_percentage_change(shifted_values, base_values)
        acceleration = momentum.diff(self.period)

        # Ensure unique index to avoid duplicate label issues
        acceleration = acceleration.reset_index(drop=True)
        base_values = base_values.reset_index(drop=True)

        # Calculate acceleration correlation with price
        acceleration_correlation = self._vectorbt_rolling_operation(
            acceleration, 'corr', window=self.period, other=base_values
        )

        # Ensure the result has a unique index
        if hasattr(acceleration_correlation, 'index') and acceleration_correlation.index.duplicated().any():
            acceleration_correlation = acceleration_correlation.reset_index(drop=True)

        return acceleration_correlation.rename(f'vectorbt_acceleration_correlation_{self.period}_{self.base_calculation.value}')

class VectorBTAccelerationDivergenceGenerator(VectorBTFeatureGenerator):
    """VectorBT-optimized acceleration divergence generator."""

    def __init__(self, period: int = 20, base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS, **base_kwargs):
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)

        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        required_columns = self.base_calculator.get_required_columns()

        config = FeatureConfig(
            name=f"vectorbt_acceleration_divergence_{period}_{base_calculation.value}",
            category=FeatureCategory.ACCELERATION,
            description=f"VectorBT-optimized acceleration divergence over {period} periods based on {base_calculation.value}",
            required_columns=required_columns,
            optional_columns=["high", "low", "open", "volume"],
            default_lookback=period * 2,
            min_lookback=period * 2,
            max_lookback=period * 2,
            parameters={'period': period, 'base_calculation': base_calculation.value, **base_kwargs},
            matrix_optimized=True,
            gpu_accelerated=True
        )
        super().__init__(config)
        self.period = period
        self.base_calculation = base_calculation

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate acceleration divergence using VectorBT operations."""
        if len(data) == 0:
            return pd.Series(dtype=float, index=data.index, name=f'vectorbt_acceleration_divergence_{self.period}_{self.base_calculation.value}')

        base_values = self.base_calculator.calculate(data)

        # Calculate acceleration
        shifted_values = base_values.shift(self.period)
        momentum = safe_percentage_change(shifted_values, base_values)
        acceleration = momentum.diff(self.period)

        # Calculate acceleration divergence (difference between acceleration and price trend)
        price_trend = self._vectorbt_rolling_operation(base_values, 'mean', window=self.period)
        acceleration_divergence = acceleration - price_trend.pct_change()

        return acceleration_divergence.rename(f'vectorbt_acceleration_divergence_{self.period}_{self.base_calculation.value}')

def create_vectorbt_acceleration_generators() -> List[VectorBTFeatureGenerator]:
    """Create all VectorBT-optimized acceleration feature generators."""
    generators = []

    # Basic acceleration features
    for period in [5, 10, 20, 50]:
        generators.append(VectorBTMomentumGenerator(period=period))

    for period in [5, 10]:
        generators.extend([
            VectorBTPriceAccelerationGenerator(period=period),
            VectorBTPriceJerkGenerator(period=period),
        ])

    for window in [5, 10, 20, 50]:
        generators.extend([
            VectorBTTrendStrengthGenerator(window=window),
            VectorBTTrendConsistencyGenerator(window=window),
        ])

    # Volume and volatility acceleration
    generators.extend([
        VectorBTVolumeAccelerationGenerator(period=5),
        VectorBTVolatilityAccelerationGenerator(period=5, volatility_window=20),
    ])

    # Advanced acceleration features
    for period in [5, 10]:
        for window in [10, 20]:
            generators.extend([
                VectorBTMomentumAccelerationGenerator(period=period, momentum_window=window),
                VectorBTAccelerationMomentumGenerator(period=period, acceleration_window=window),
                VectorBTAccelerationVolatilityGenerator(period=period, acceleration_window=window),
                VectorBTAccelerationTrendStrengthGenerator(period=period, acceleration_window=window),
                VectorBTAccelerationConsistencyGenerator(period=period, acceleration_window=window),
                VectorBTAccelerationRegimeGenerator(period=period, acceleration_window=window),
            ])

    # Multi-timeframe and correlation features
    generators.extend([
        VectorBTMultiTimeframeAccelerationGenerator(short_period=5, long_period=20),
        VectorBTAccelerationCorrelationGenerator(period=20),
        VectorBTAccelerationDivergenceGenerator(period=20),
    ])

    return generators

def create_default_vectorbt_acceleration_generators() -> List[VectorBTFeatureGenerator]:
    """Create default VectorBT-optimized acceleration feature generators."""
    return create_vectorbt_acceleration_generators()

def create_optimized_acceleration_batch_generator() -> 'OptimizedAccelerationBatchGenerator':
    """Create an optimized batch generator for acceleration features using VectorBT and UnifiedVectorizationManager."""
    return OptimizedAccelerationBatchGenerator()

class OptimizedAccelerationBatchGenerator:
    """Optimized batch generator for acceleration features using VectorBT and UnifiedVectorizationManager."""

    def __init__(self):
        """Initialize the optimized batch generator."""
        self.vectorbt_rolling_optimizer = None
        self.unified_manager = None

        if VECTORBT_ROLLING_OPTIMIZER_AVAILABLE:
            self.vectorbt_rolling_optimizer = get_vectorbt_rolling_optimizer(enable_gpu=True, enable_parallel=True)

        if UNIFIED_VECTORIZATION_AVAILABLE:
            self.unified_manager = get_unified_vectorization_manager()

        self.logger = logging.getLogger(__name__)

    def generate_acceleration_features_batch(self, data: pd.DataFrame,
                                           feature_configs: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Generate multiple acceleration features in batch with VectorBT optimization.

        Args:
            data: Input OHLCV data
            feature_configs: List of feature configuration dictionaries

        Returns:
            DataFrame with generated acceleration features
        """
        # Enhanced UnifiedVectorizationManager batch processing
        if self.unified_manager and UNIFIED_VECTORIZATION_AVAILABLE:
            try:
                from ...utils.ml_common.unified_vectorization_manager import OperationConfig
                config = OperationConfig(
                    operation_type=OperationType.FEATURE_ENGINEERING,
                    data_size=len(data),
                    data_dimensions=data.shape,
                    memory_budget_mb=4096.0,  # Increased memory budget for batch processing
                    time_budget_seconds=300.0,  # Increased time budget
                    precision_requirement="high",
                    parallel_workers=4  # Enable parallel processing
                )

                # Use UnifiedVectorizationManager for batch processing
                result = self.unified_manager.optimize_operation(
                    OperationType.FEATURE_ENGINEERING,
                    data,
                    config,
                    feature_configs=feature_configs,
                    batch_processing=True  # Enable batch processing mode
                )
                return result.result

            except Exception as e:
                self.logger.warning(f"UnifiedVectorizationManager batch processing failed: {e}, using VectorBT fallback")

        # Enhanced VectorBT batch processing
        if self.vectorbt_rolling_optimizer and VECTORBT_ROLLING_OPTIMIZER_AVAILABLE:
            try:
                return self._vectorbt_batch_processing(data, feature_configs)
            except Exception as e:
                self.logger.warning(f"VectorBT batch processing failed: {e}, using individual generators")

        # Fallback to individual generator processing
        return self._individual_generator_processing(data, feature_configs)

    def _vectorbt_batch_processing(self, data: pd.DataFrame,
                                 feature_configs: List[Dict[str, Any]]) -> pd.DataFrame:
        """Process acceleration features using VectorBT batch operations."""
        results = {}

        # Pre-calculate common base values for efficiency
        base_values_cache = {}

        for config in feature_configs:
            feature_type = config.get('type', 'momentum')
            feature_name = config.get('name', f'{feature_type}_{config.get("period", 10)}')
            period = config.get('period', 10)
            base_calculation = config.get('base_calculation', 'price_returns')

            try:
                # Cache base values to avoid recalculation
                if base_calculation not in base_values_cache:
                    if base_calculation == 'price_returns':
                        base_values_cache[base_calculation] = data['close'].pct_change()
                    else:
                        base_values_cache[base_calculation] = data['close']

                base_values = base_values_cache[base_calculation]

                if feature_type == 'momentum':
                    result = self._generate_momentum_vectorbt_optimized(data, base_values, period, base_calculation)
                elif feature_type == 'acceleration':
                    result = self._generate_acceleration_vectorbt_optimized(data, base_values, period, base_calculation)
                elif feature_type == 'jerk':
                    result = self._generate_jerk_vectorbt_optimized(data, base_values, period, base_calculation)
                elif feature_type == 'trend_strength':
                    result = self._generate_trend_strength_vectorbt_optimized(data, base_values, period, base_calculation)
                elif feature_type == 'trend_consistency':
                    result = self._generate_trend_consistency_vectorbt_optimized(data, base_values, period, base_calculation)
                else:
                    self.logger.warning(f"Unknown feature type: {feature_type}")
                    continue

                results[feature_name] = result

            except Exception as e:
                self.logger.warning(f"Failed to generate {feature_name}: {e}")
                continue

        return pd.DataFrame(results, index=data.index)

    def _generate_momentum_vectorbt(self, data: pd.DataFrame, period: int, base_calculation: str) -> pd.Series:
        """Generate momentum using VectorBT optimization."""
        if base_calculation == 'price_returns':
            base_values = data['close'].pct_change()
        else:
            base_values = data['close']

        shifted_values = base_values.shift(period)
        momentum = safe_percentage_change(shifted_values, base_values)
        return momentum

    def _generate_momentum_vectorbt_optimized(self, data: pd.DataFrame, base_values: pd.Series, period: int, base_calculation: str) -> pd.Series:
        """Generate momentum using optimized VectorBT operations."""
        if self.vectorbt_rolling_optimizer and VECTORBT_ROLLING_OPTIMIZER_AVAILABLE:
            try:
                def momentum_func(series):
                    if len(series) < period + 1:
                        return np.nan
                    current = series.iloc[-1]
                    past = series.iloc[-period-1]
                    return safe_percentage_change(past, current)

                return self.vectorbt_rolling_optimizer.rolling_apply(
                    base_values, momentum_func, window=period + 1
                )
            except Exception as e:
                self.logger.warning(f"VectorBT optimized momentum failed: {e}")

        # Fallback to standard calculation
        shifted_values = base_values.shift(period)
        return safe_percentage_change(shifted_values, base_values)

    def _generate_acceleration_vectorbt(self, data: pd.DataFrame, period: int, base_calculation: str) -> pd.Series:
        """Generate acceleration using VectorBT optimization."""
        if base_calculation == 'price_returns':
            base_values = data['close'].pct_change()
        else:
            base_values = data['close']

        shifted_values = base_values.shift(period)
        momentum = safe_percentage_change(shifted_values, base_values)
        acceleration = momentum.diff(period)
        return acceleration

    def _generate_acceleration_vectorbt_optimized(self, data: pd.DataFrame, base_values: pd.Series, period: int, base_calculation: str) -> pd.Series:
        """Generate acceleration using optimized VectorBT operations."""
        if self.vectorbt_rolling_optimizer and VECTORBT_ROLLING_OPTIMIZER_AVAILABLE:
            try:
                # Calculate momentum first
                shifted_values = base_values.shift(period)
                momentum = safe_percentage_change(shifted_values, base_values)

                # Use VectorBT for acceleration calculation
                acceleration = self.vectorbt_rolling_optimizer.rolling_apply(
                    momentum, lambda x: x.diff().iloc[-1] if len(x) > 1 else np.nan, window=period + 1
                )
                return acceleration
            except Exception as e:
                self.logger.warning(f"VectorBT optimized acceleration failed: {e}")

        # Fallback to standard calculation
        shifted_values = base_values.shift(period)
        momentum = safe_percentage_change(shifted_values, base_values)
        return momentum.diff(period)

    def _generate_jerk_vectorbt(self, data: pd.DataFrame, period: int, base_calculation: str) -> pd.Series:
        """Generate jerk using VectorBT optimization."""
        if base_calculation == 'price_returns':
            base_values = data['close'].pct_change()
        else:
            base_values = data['close']

        shifted_values = base_values.shift(period)
        momentum = safe_percentage_change(shifted_values, base_values)
        acceleration = momentum.diff(period)
        jerk = acceleration.diff(period)
        return jerk

    def _generate_jerk_vectorbt_optimized(self, data: pd.DataFrame, base_values: pd.Series, period: int, base_calculation: str) -> pd.Series:
        """Generate jerk using optimized VectorBT operations."""
        if self.vectorbt_rolling_optimizer and VECTORBT_ROLLING_OPTIMIZER_AVAILABLE:
            try:
                # Calculate momentum and acceleration first
                shifted_values = base_values.shift(period)
                momentum = safe_percentage_change(shifted_values, base_values)
                acceleration = momentum.diff(period)

                # Use VectorBT for jerk calculation
                jerk = self.vectorbt_rolling_optimizer.rolling_apply(
                    acceleration, lambda x: x.diff().iloc[-1] if len(x) > 1 else np.nan, window=period + 1
                )
                return jerk
            except Exception as e:
                self.logger.warning(f"VectorBT optimized jerk failed: {e}")

        # Fallback to standard calculation
        shifted_values = base_values.shift(period)
        momentum = safe_percentage_change(shifted_values, base_values)
        acceleration = momentum.diff(period)
        return acceleration.diff(period)

    def _generate_trend_strength_vectorbt(self, data: pd.DataFrame, period: int, base_calculation: str) -> pd.Series:
        """Generate trend strength using VectorBT optimization."""
        if base_calculation == 'price_returns':
            base_values = data['close'].pct_change()
        else:
            base_values = data['close']

        def calculate_trend_strength(series):
            if len(series) < 2:
                return 0.0
            try:
                slope = np.polyfit(range(len(series)), series, 1)[0]
                return slope
            except:
                return 0.0

        if self.vectorbt_rolling_optimizer:
            return self.vectorbt_rolling_optimizer.rolling_apply(base_values, calculate_trend_strength, period)
        else:
            return base_values.rolling(window=period).apply(calculate_trend_strength, raw=False)

    def _generate_trend_strength_vectorbt_optimized(self, data: pd.DataFrame, base_values: pd.Series, period: int, base_calculation: str) -> pd.Series:
        """Generate trend strength using optimized VectorBT operations."""
        
        # Define the function outside the try block to ensure it's always available
        def calculate_trend_strength(series):
            if len(series) < 2:
                return 0.0
            try:
                slope = np.polyfit(range(len(series)), series, 1)[0]
                return slope
            except:
                return 0.0
        
        if self.vectorbt_rolling_optimizer and VECTORBT_ROLLING_OPTIMIZER_AVAILABLE:
            try:
                return self.vectorbt_rolling_optimizer.rolling_apply(
                    base_values, calculate_trend_strength, window=period, min_periods=1
                )
            except Exception as e:
                self.logger.warning(f"VectorBT optimized trend strength failed: {e}")

        # Fallback to standard calculation
        def calculate_trend_strength_fallback(series):
            if len(series) < 2:
                return 0.0
            try:
                slope = np.polyfit(range(len(series)), series, 1)[0]
                return slope
            except:
                return 0.0
        
        return base_values.rolling(window=period).apply(calculate_trend_strength_fallback, raw=False)

    def _generate_trend_consistency_vectorbt(self, data: pd.DataFrame, period: int, base_calculation: str) -> pd.Series:
        """Generate trend consistency using VectorBT optimization."""
        if base_calculation == 'price_returns':
            base_values = data['close'].pct_change()
        else:
            base_values = data['close']

        def calculate_trend_consistency(series):
            if len(series) < 2:
                return 0
            try:
                slope = np.polyfit(range(len(series)), series, 1)[0]
                return 1 if slope > 0 else 0
            except:
                return 0

        if self.vectorbt_rolling_optimizer:
            return self.vectorbt_rolling_optimizer.rolling_apply(base_values, calculate_trend_consistency, period)
        else:
            return base_values.rolling(window=period).apply(calculate_trend_consistency, raw=False)

    def _generate_trend_consistency_vectorbt_optimized(self, data: pd.DataFrame, base_values: pd.Series, period: int, base_calculation: str) -> pd.Series:
        """Generate trend consistency using optimized VectorBT operations."""
        if self.vectorbt_rolling_optimizer and VECTORBT_ROLLING_OPTIMIZER_AVAILABLE:
            try:
                def calculate_trend_consistency(series):
                    if len(series) < 2:
                        return 0
                    try:
                        slope = np.polyfit(range(len(series)), series, 1)[0]
                        return 1 if slope > 0 else 0
                    except:
                        return 0

                return self.vectorbt_rolling_optimizer.rolling_apply(
                    base_values, calculate_trend_consistency, window=period, min_periods=1
                )
            except Exception as e:
                self.logger.warning(f"VectorBT optimized trend consistency failed: {e}")

        # Fallback to standard calculation
        def calculate_trend_consistency(series):
            if len(series) < 2:
                return 0
            try:
                slope = np.polyfit(range(len(series)), series, 1)[0]
                return 1 if slope > 0 else 0
            except:
                return 0

        return base_values.rolling(window=period).apply(calculate_trend_consistency, raw=False)

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate acceleration features using VectorBT optimization."""
        try:
            # Use the comprehensive feature generation method
            features_df = self.generate_features(data, **kwargs)
            
            # Return the first feature as a representative series
            if not features_df.empty:
                first_feature_name = features_df.columns[0]
                return features_df[first_feature_name]
            else:
                # Return a default series if no features were generated
                return pd.Series(np.zeros(len(data)), index=data.index)
                
        except Exception as e:
            self.logger.warning(f"Acceleration feature generation failed: {e}")
            return pd.Series(np.zeros(len(data)), index=data.index)

    def _individual_generator_processing(self, data: pd.DataFrame,
                                      feature_configs: List[Dict[str, Any]]) -> pd.DataFrame:
        """Process features using individual generators as fallback."""
        results = {}

        for config in feature_configs:
            feature_type = config.get('type', 'momentum')
            feature_name = config.get('name', f'{feature_type}_{config.get("period", 10)}')
            period = config.get('period', 10)
            base_calculation = config.get('base_calculation', 'price_returns')

            try:
                if feature_type == 'momentum':
                    generator = VectorBTMomentumGenerator(period=period, base_calculation=base_calculation)
                elif feature_type == 'acceleration':
                    generator = VectorBTPriceAccelerationGenerator(period=period, base_calculation=base_calculation)
                elif feature_type == 'jerk':
                    generator = VectorBTPriceJerkGenerator(period=period, base_calculation=base_calculation)
                elif feature_type == 'trend_strength':
                    generator = VectorBTTrendStrengthGenerator(window=period, base_calculation=base_calculation)
                elif feature_type == 'trend_consistency':
                    generator = VectorBTTrendConsistencyGenerator(window=period, base_calculation=base_calculation)
                else:
                    self.logger.warning(f"Unknown feature type: {feature_type}")
                    continue

                result = generator._generate_feature(data)
                results[feature_name] = result

            except Exception as e:
                self.logger.warning(f"Failed to generate {feature_name}: {e}")
                continue

        return pd.DataFrame(results, index=data.index)

# Legacy compatibility functions
def create_acceleration_generators() -> List[FeatureGenerator]:
    """Create all acceleration-based feature generators with VectorBT optimization."""
    generators = []

    # Always prioritize VectorBT generators if available
    if VECTORBT_AVAILABLE:
        # Use VectorBT-optimized generators
        vectorbt_generators = create_vectorbt_acceleration_generators()
        generators.extend(vectorbt_generators)
        print(f"✅ Created {len(vectorbt_generators)} VectorBT-optimized acceleration generators")
    else:
        print("⚠️ VectorBT acceleration generators not available, using legacy generators")

    # Add the comprehensive AccelerationFeatureGenerator
    generators.append(AccelerationFeatureGenerator())

    print(f"✅ Created {len(generators)} total acceleration generators with VectorBT optimization")
    return generators

def create_default_acceleration_generators() -> List[FeatureGenerator]:
    """Create default acceleration-based feature generators (alias for create_acceleration_generators)."""
    return create_acceleration_generators()

# Export all generators
__all__ = [
    'AccelerationFeatureGenerator',
    'VectorBTMomentumGenerator',
    'VectorBTPriceAccelerationGenerator',
    'VectorBTPriceJerkGenerator',
    'VectorBTTrendStrengthGenerator',
    'VectorBTTrendConsistencyGenerator',
    'VectorBTVolumeAccelerationGenerator',
    'VectorBTVolatilityAccelerationGenerator',
    'VectorBTMomentumAccelerationGenerator',
    'VectorBTAccelerationMomentumGenerator',
    'VectorBTAccelerationVolatilityGenerator',
    'VectorBTAccelerationTrendStrengthGenerator',
    'VectorBTAccelerationConsistencyGenerator',
    'VectorBTAccelerationRegimeGenerator',
    'VectorBTMultiTimeframeAccelerationGenerator',
    'VectorBTAccelerationCorrelationGenerator',
    'VectorBTAccelerationDivergenceGenerator',
    'OptimizedAccelerationBatchGenerator',
    'create_vectorbt_acceleration_generators',
    'create_default_vectorbt_acceleration_generators',
    'create_optimized_acceleration_batch_generator',
    'create_acceleration_generators',
    'create_default_acceleration_generators'
]
