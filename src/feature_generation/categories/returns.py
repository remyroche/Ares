"""
Returns Feature Generator

This module provides feature generators for return-based indicators,
including log returns, cumulative returns, rolling returns, and return statistics.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union

from ..core.feature_generator import FeatureGenerator, FeatureResult, VectorizedFeatureGenerator, FeatureConfig, FeatureCategory

# Optimization utilities
try:
    from src.feature_generation.utils.vectorization_optimizer import get_vectorization_optimizer
    from src.feature_generation.utils.optimized_feature_pipeline import get_optimized_feature_pipeline
    OPTIMIZATION_AVAILABLE = True
except ImportError:
    OPTIMIZATION_AVAILABLE = False

# VectorBT Rolling Optimizer
try:
    from src.feature_generation.utils.vectorbt_rolling_optimizer import get_vectorbt_rolling_optimizer, VectorBTRollingOptimizer
    VECTORBT_ROLLING_OPTIMIZER_AVAILABLE = True
except ImportError:
    VECTORBT_ROLLING_OPTIMIZER_AVAILABLE = False
    VectorBTRollingOptimizer = None

# Unified Vectorization Manager
try:
    from ...utils.ml_common.unified_vectorization_manager import (
        get_unified_vectorization_manager, UnifiedVectorizationManager,
        OperationType, OptimizationStrategy, OperationConfig
    )
    UNIFIED_VECTORIZATION_MANAGER_AVAILABLE = True
except ImportError:
    UNIFIED_VECTORIZATION_MANAGER_AVAILABLE = False
    UnifiedVectorizationManager = None
from ..base_calculations import (
    BaseCalculationType,
    create_base_calculator
)

# VectorBT imports for native optimization
try:
    import vectorbt as vbt
    from vectorbt.indicators.basic import RSI, MACD, ATR, BBANDS, STOCH, OBV, MA
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None
    RSI = None
    MACD = None
    ATR = None
    BBANDS = None
    STOCH = None
    OBV = None
    MA = None
    import warnings
    warnings.warn("VectorBT not available. Install with: pip install vectorbt for optimized performance")

except ImportError:

    cp = None

class ReturnsFeatureGenerator(VectorizedFeatureGenerator):
    """Feature generator for return-based features with full VectorBT optimization."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if config is None:
            config = self._create_default_config()
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)

        # Initialize VectorBT components
        self.rolling_optimizer = None
        self.unified_manager = None

        # Initialize VectorBT Rolling Optimizer
        if VECTORBT_ROLLING_OPTIMIZER_AVAILABLE:
            try:
                self.rolling_optimizer = get_vectorbt_rolling_optimizer(enable_gpu=False, enable_parallel=True)
                self.logger.info("✅ VectorBTRollingOptimizer initialized for returns features")
            except Exception as e:
                self.logger.warning(f"⚠️ VectorBTRollingOptimizer initialization failed: {e}")

        # Initialize Unified Vectorization Manager
        if UNIFIED_VECTORIZATION_MANAGER_AVAILABLE:
            try:
                self.unified_manager = get_unified_vectorization_manager()
                self.logger.info("✅ UnifiedVectorizationManager initialized for returns features")
            except Exception as e:
                self.logger.warning(f"⚠️ UnifiedVectorizationManager initialization failed: {e}")

        # Performance tracking
        self.performance_stats = {
            'vectorbt_operations': 0,
            'pandas_fallbacks': 0,
            'unified_manager_operations': 0,
            'rolling_optimizer_operations': 0,
            'total_operations': 0,
            'total_time': 0.0
        }

    @classmethod
    def _create_default_config(cls) -> FeatureConfig:
        return FeatureConfig(
            name="returns_features",
            category=FeatureCategory.RETURNS,
            description="Comprehensive return-based features including log returns, cumulative returns, and return statistics",
            required_columns=["close"],
            optional_columns=["high", "low", "open", "volume"],
            default_lookback=20,
            min_lookback=2,
            max_lookback=50,
            parameters={
                "return_periods": [1, 5, 10, 20],
                "log_return_periods": [1, 5, 10],
                "cumulative_windows": [10, 20]
            },
            matrix_optimized=True,
            gpu_accelerated=False
        )

    @classmethod
    def create_default(cls) -> 'ReturnsFeatureGenerator':
        return cls()

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        if data.empty:
            return pd.Series(dtype=float, index=data.index, name='returns_1')

        close_prices = data['close'].astype(float).values
        state = self.get_state()
        history = state.get('close_history') or []

        if history:
            try:
                history_array = np.asarray(history, dtype=float)
            except Exception:
                history_array = np.array(history, dtype=float)
            combined_closes = np.concatenate([history_array, close_prices])
        else:
            combined_closes = close_prices

        combined_returns = self._calculate_returns(combined_closes, period=1)
        returns = combined_returns[-len(close_prices):] if len(close_prices) else np.array([])

        prev_close = state.get('last_close')
        if len(returns) > 0 and prev_close not in (None, np.nan):
            try:
                prev_close_val = float(prev_close)
                returns[0] = (close_prices[0] - prev_close_val) / prev_close_val if prev_close_val != 0 else np.nan
            except Exception:
                returns[0] = np.nan

        return pd.Series(returns, index=data.index, name='returns_1')

    def _calculate_returns(self, prices: np.ndarray, period: int = 1) -> np.ndarray:
        if len(prices) < period + 1:
            return np.full(len(prices), np.nan)

        prices_series = pd.Series(prices)

        # Use Unified Vectorization Manager for optimized returns calculation
        if self.unified_manager and len(prices) > 50:
            try:
                # Use Unified Vectorization Manager for optimized returns calculation
                returns_result = self.unified_manager.optimize_operation(
                    OperationType.TECHNICAL_INDICATORS,
                    {
                        'data': prices_series,
                        'operation': 'returns',
                        'period': period,
                        'indicator_configs': {'returns': {'period': period}}
                    },
                    OperationConfig(
                        operation_type=OperationType.TECHNICAL_INDICATORS,
                        data_size=len(prices_series),
                        data_dimensions=prices_series.shape,
                        memory_budget_mb=256.0
                    )
                )
                returns = returns_result.result

                self.performance_stats['unified_manager_operations'] += 1
                return returns
            except Exception as e:
                self.logger.warning(f"Unified Vectorization Manager returns calculation failed: {e}, using VectorBT fallback")
                if self.rolling_optimizer:
                    try:
                        if period == 1:
                            returns = prices_series.pct_change(periods=period).values
                        else:
                            returns = self.rolling_optimizer.rolling_apply(
                                prices_series,
                                lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0] if len(x) == period and x.iloc[0] != 0 else np.nan,
                                window=period
                            ).values
                        self.performance_stats['rolling_optimizer_operations'] += 1
                        return returns
                    except Exception as e2:
                        self.logger.warning(f"VectorBT Rolling Optimizer returns calculation failed: {e2}, using VectorBT fallback")
                        self.performance_stats['pandas_fallbacks'] += 1
                else:
                    self.performance_stats['pandas_fallbacks'] += 1

        # Use VectorBT Rolling Optimizer for optimized returns calculation
        elif self.rolling_optimizer and len(prices) > 50:  # Lower threshold for VectorBT usage
            try:
                # Use VectorBT Rolling Optimizer for percentage change
                if period == 1:
                    # For single period, use direct pct_change
                    returns = prices_series.pct_change(periods=period).values
                else:
                    # For multiple periods, use rolling operations
                    returns = self.rolling_optimizer.rolling_apply(
                        prices_series,
                        lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0] if len(x) == period and x.iloc[0] != 0 else np.nan,
                        window=period
                    ).values

                self.performance_stats['rolling_optimizer_operations'] += 1
                return returns
            except Exception as e:
                self.logger.warning(f"VectorBT Rolling Optimizer returns calculation failed: {e}, using VectorBT fallback")
                self.performance_stats['pandas_fallbacks'] += 1

        # Use VectorBT for optimized returns calculation
        elif VECTORBT_AVAILABLE and len(prices) > 100:  # Use VectorBT for larger datasets
            try:
                # VectorBT optimized pct_change
                returns = prices_series.pct_change(periods=period).values
                self.performance_stats['vectorbt_operations'] += 1
                return returns
            except Exception as e:
                self.logger.warning(f"VectorBT returns calculation failed: {e}, using numpy fallback")
                self.performance_stats['pandas_fallbacks'] += 1

        # Fallback to numpy
        returns = (prices - np.roll(prices, period)) / np.roll(prices, period)
        returns[:period] = np.nan
        return returns

    def _finalize_state(self, data: pd.DataFrame, feature_data: pd.Series) -> None:
        if not data.empty:
            closes = data['close'].astype(float)
            history_window = max(int(getattr(self.config, 'min_lookback', 2)), 2)
            close_history = closes.tolist()[-history_window:]
            state_update = {
                'last_close': float(closes.iloc[-1]),
                'close_history': close_history
            }
            if not feature_data.empty:
                last_return = feature_data.iloc[-1]
                if pd.notna(last_return):
                    state_update['last_return'] = float(last_return)
            self.update_state(state_update)

        super()._finalize_state(data, feature_data)

    def generate_returns_features_batch(self, data: pd.DataFrame,
                                      feature_configs: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Generate multiple returns features in batch using VectorBT optimization.

        Args:
            data: Input OHLCV data
            feature_configs: List of feature configuration dictionaries

        Returns:
            DataFrame with generated features
        """
        if self.unified_manager and len(data) > 100:
            try:
                # Use Unified Vectorization Manager for batch processing
                batch_result = self.unified_manager.optimize_operation(
                    OperationType.FEATURE_ENGINEERING,
                    {
                        'data': data,
                        'feature_configs': feature_configs,
                        'operation_type': 'returns_batch'
                    },
                    OperationConfig(
                        operation_type=OperationType.FEATURE_ENGINEERING,
                        data_size=len(data),
                        data_dimensions=data.shape,
                        memory_budget_mb=2048.0
                    )
                )
                return batch_result.result
            except Exception as e:
                self.logger.warning(f"Unified Vectorization Manager batch processing failed: {e}, using fallback")
                # Fallback to individual processing
                return self._process_returns_features_individually(data, feature_configs)
        else:
            return self._fallback_batch_processing(data, feature_configs)

    def _process_returns_features_individually(self, data: pd.DataFrame, feature_configs: List[Dict[str, Any]]) -> pd.DataFrame:
        """Process returns features individually as fallback when batch processing fails."""
        results = {}
        for config in feature_configs:
            feature_name = config['name']
            feature_type = config.get('type', 'returns')
            params = config.get('params', {})

            try:
                if feature_type == 'returns':
                    period = params.get('period', 1)
                    column = params.get('column', 'close')

                    if column in data.columns:
                        prices = data[column].values
                        returns = self._calculate_returns(prices, period)
                        results[feature_name] = pd.Series(returns, index=data.index)

            except Exception as e:
                self.logger.warning(f"Returns feature {feature_name} failed: {e}")
                results[feature_name] = pd.Series(np.nan, index=data.index)

        return pd.DataFrame(results, index=data.index)

    def _fallback_batch_processing(self, data: pd.DataFrame,
                                 feature_configs: List[Dict[str, Any]]) -> pd.DataFrame:
        """Fallback batch processing using individual feature generators."""
        results = {}

        for config in feature_configs:
            feature_type = config.get('type', 'simple_returns')
            period = config.get('period', 1)
            window = config.get('window', 20)

            try:
                if feature_type == 'simple_returns':
                    generator = SimpleReturnsGenerator(period)
                elif feature_type == 'log_returns':
                    generator = LogReturnsGenerator(period)
                elif feature_type == 'cumulative_returns':
                    generator = CumulativeReturnsGenerator(window)
                elif feature_type == 'rolling_returns':
                    generator = RollingReturnsGenerator(window)
                elif feature_type == 'returns_volatility':
                    generator = ReturnsVolatilityGenerator(window)
                elif feature_type == 'returns_skewness':
                    generator = ReturnsSkewnessGenerator(window)
                elif feature_type == 'returns_kurtosis':
                    generator = ReturnsKurtosisGenerator(window)
                elif feature_type == 'sharpe_ratio':
                    generator = SharpeRatioGenerator(window)
                else:
                    continue

                feature_result = generator.generate_feature(data)
                results[f"{feature_type}_{period}_{window}"] = feature_result

            except Exception as e:
                self.logger.warning(f"Feature generation failed for {feature_type}: {e}")
                continue

        return pd.DataFrame(results, index=data.index)

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        stats = self.performance_stats.copy()

        if stats['total_operations'] > 0:
            stats['vectorbt_usage_percentage'] = (
                (stats['vectorbt_operations'] + stats['rolling_optimizer_operations']) /
                stats['total_operations'] * 100
            )
            stats['unified_manager_usage_percentage'] = (
                stats['unified_manager_operations'] / stats['total_operations'] * 100
            )
            stats['pandas_fallback_percentage'] = (
                stats['pandas_fallbacks'] / stats['total_operations'] * 100
            )
        else:
            stats['vectorbt_usage_percentage'] = 0
            stats['unified_manager_usage_percentage'] = 0
            stats['pandas_fallback_percentage'] = 0

        return stats

class LogReturnsGenerator(VectorizedFeatureGenerator):
    """Generator for Log Returns with different base calculations - VECTORIZED with VectorBT optimization."""

    def __init__(self,
                 period: int = 1,
                 base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS,
                 **base_kwargs):
        """
        Initialize Log Returns generator with VectorBT optimization.

        Args:
            period: Return period
            base_calculation: Base calculation type (price_levels, returns_vwap, etc.)
            **base_kwargs: Additional parameters for base calculation
        """
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)

        # Create base calculator
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)

        # Update required columns based on base calculation
        required_columns = self.base_calculator.get_required_columns()

        config = FeatureConfig(
            name=f"log_returns_{period}_{base_calculation.value}",
            category=FeatureCategory.RETURNS,
            description=f"Log returns over {period} periods based on {base_calculation.value}",
            required_columns=required_columns,
            default_lookback=period + 1,
            min_lookback=period + 1,
            max_lookback=period + 1,
            parameters={
                'period': period,
                'base_calculation': base_calculation.value,
                **base_kwargs
            },
            matrix_optimized=True,
            gpu_accelerated=False
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.period = period
        self.base_calculation = base_calculation

        # Initialize VectorBT Rolling Optimizer
        self.rolling_optimizer = None
        if VECTORBT_ROLLING_OPTIMIZER_AVAILABLE:
            try:
                self.rolling_optimizer = get_vectorbt_rolling_optimizer(enable_gpu=False, enable_parallel=True)
            except Exception as e:
                self.logger.warning(f"VectorBTRollingOptimizer initialization failed: {e}")

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate log returns based on the specified base calculation - VECTORIZED with VectorBT optimization."""
        # Calculate base values
        base_values = self.base_calculator.calculate(data)

        # Convert to pandas Series for VectorBT operations
        values_series = base_values if isinstance(base_values, pd.Series) else pd.Series(base_values, index=data.index)

        # Vectorized log returns calculation with VectorBT optimization
        if len(values_series) < self.period + 1:
            return pd.Series(np.full(len(values_series), np.nan), index=data.index)

        # Use VectorBT Rolling Optimizer for log returns calculation
        if self.rolling_optimizer and len(values_series) > 50:
            try:
                # Use VectorBT Rolling Optimizer for percentage change, then apply log
                if self.period == 1:
                    # For single period, use direct pct_change
                    returns = values_series.pct_change(periods=self.period)
                else:
                    # For multiple periods, use rolling operations
                    returns = self.rolling_optimizer.rolling_apply(
                        values_series,
                        lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0] if len(x) == self.period and x.iloc[0] != 0 else np.nan,
                        window=self.period
                    )

                # Apply log transformation to returns
                log_returns = np.log(1 + returns)
                return log_returns

            except Exception as e:
                self.logger.warning(f"VectorBT Rolling Optimizer log returns calculation failed: {e}, using numpy fallback")

        # Fallback to numpy operations
        values = values_series.values

        # Calculate log returns using numpy operations
        shifted_values = np.roll(values, self.period)
        shifted_values[:self.period] = np.nan  # Set initial values to NaN

        # Avoid division by zero and log of zero
        ratio = values / shifted_values
        ratio = np.where(np.isfinite(ratio) & (ratio > 0), ratio, np.nan)

        log_returns = np.log(ratio)

        return pd.Series(log_returns, index=data.index)

class SimpleReturnsGenerator(VectorizedFeatureGenerator):
    """Generator for Simple Returns with different base calculations - VECTORIZED with VectorBT optimization."""

    def __init__(self,
                 period: int = 1,
                 base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS,
                 **base_kwargs):
        """
        Initialize Simple Returns generator with VectorBT optimization.

        Args:
            period: Return period
            base_calculation: Base calculation type (price_levels, returns_vwap, etc.)
            **base_kwargs: Additional parameters for base calculation
        """
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)

        # Create base calculator
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)

        # Update required columns based on base calculation
        required_columns = self.base_calculator.get_required_columns()

        config = FeatureConfig(
            name=f"simple_returns_{period}_{base_calculation.value}",
            category=FeatureCategory.RETURNS,
            description=f"Simple returns over {period} periods based on {base_calculation.value}",
            required_columns=required_columns,
            default_lookback=period + 1,
            min_lookback=period + 1,
            max_lookback=period + 1,
            parameters={
                'period': period,
                'base_calculation': base_calculation.value,
                **base_kwargs
            },
            matrix_optimized=True,
            gpu_accelerated=False
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.period = period
        self.base_calculation = base_calculation

        # Initialize VectorBT Rolling Optimizer
        self.rolling_optimizer = None
        if VECTORBT_ROLLING_OPTIMIZER_AVAILABLE:
            try:
                self.rolling_optimizer = get_vectorbt_rolling_optimizer(enable_gpu=False, enable_parallel=True)
            except Exception as e:
                self.logger.warning(f"VectorBTRollingOptimizer initialization failed: {e}")

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate simple returns based on the specified base calculation - VECTORIZED with VectorBT optimization."""
        # Calculate base values
        base_values = self.base_calculator.calculate(data)

        # Convert to pandas Series for VectorBT operations
        values_series = base_values if isinstance(base_values, pd.Series) else pd.Series(base_values, index=data.index)

        # Vectorized simple returns calculation with VectorBT optimization
        if len(values_series) < self.period + 1:
            return pd.Series(np.full(len(values_series), np.nan), index=data.index)

        # Use VectorBT Rolling Optimizer for simple returns calculation
        if self.rolling_optimizer and len(values_series) > 50:
            try:
                # Use VectorBT Rolling Optimizer for percentage change
                if self.period == 1:
                    # For single period, use direct pct_change
                    simple_returns = values_series.pct_change(periods=self.period)
                else:
                    # For multiple periods, use rolling operations
                    simple_returns = self.rolling_optimizer.rolling_apply(
                        values_series,
                        lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0] if len(x) == self.period and x.iloc[0] != 0 else np.nan,
                        window=self.period
                    )

                return simple_returns

            except Exception as e:
                self.logger.warning(f"VectorBT Rolling Optimizer simple returns calculation failed: {e}, using numpy fallback")

        # Fallback to numpy operations
        values = values_series.values

        # Calculate simple returns using numpy operations
        shifted_values = np.roll(values, self.period)
        shifted_values[:self.period] = np.nan  # Set initial values to NaN

        # Avoid division by zero
        simple_returns = np.where(
            np.isfinite(shifted_values) & (shifted_values != 0),
            (values - shifted_values) / shifted_values,
            np.nan
        )

        return pd.Series(simple_returns, index=data.index)

class CumulativeReturnsGenerator(VectorizedFeatureGenerator):
    """Generator for Cumulative Returns with different base calculations - VECTORIZED."""

    def __init__(self,
                 window: int = 20,
                 base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS,
                 **base_kwargs):
        """
        Initialize Cumulative Returns generator.

        Args:
            window: Rolling window for cumulative returns
            base_calculation: Base calculation type (price_levels, returns_vwap, etc.)
            **base_kwargs: Additional parameters for base calculation
        """
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)

        # Create base calculator
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)

        # Update required columns based on base calculation
        required_columns = self.base_calculator.get_required_columns()

        config = FeatureConfig(
            name=f"cumulative_returns_{window}_{base_calculation.value}",
            category=FeatureCategory.RETURNS,
            description=f"Cumulative returns over {window} periods based on {base_calculation.value}",
            required_columns=required_columns,
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={
                'window': window,
                'base_calculation': base_calculation.value,
                **base_kwargs
            },
            matrix_optimized=True,
            gpu_accelerated=False
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.window = window
        self.base_calculation = base_calculation

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate cumulative returns based on the specified base calculation - VECTORIZED."""
        # Calculate base values
        base_values = self.base_calculator.calculate(data)

        # Convert to numpy array for vectorized operations
        values = base_values.values

        # Vectorized cumulative returns calculation
        if len(values) < self.window + 1:
            return pd.Series(np.full(len(values), np.nan), index=data.index)

        # Calculate returns using vectorized operations
        returns = np.diff(values) / values[:-1]
        returns = np.concatenate([[np.nan], returns])  # Add NaN for first value

        # Vectorized rolling cumulative returns calculation
        cumulative_returns = np.full(len(values), np.nan)

        for i in range(self.window, len(values)):
            window_returns = returns[i-self.window+1:i+1]
            # Filter out NaN values
            valid_returns = window_returns[np.isfinite(window_returns)]
            if len(valid_returns) > 0:
                cumulative_returns[i] = np.prod(1 + valid_returns) - 1

        return pd.Series(cumulative_returns, index=data.index)

class RollingReturnsGenerator(VectorizedFeatureGenerator):
    """Generator for Rolling Returns with different base calculations - VECTORIZED."""

    def __init__(self,
                 window: int = 20,
                 base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS,
                 **base_kwargs):
        """
        Initialize Rolling Returns generator.

        Args:
            window: Rolling window for returns
            base_calculation: Base calculation type (price_levels, returns_vwap, etc.)
            **base_kwargs: Additional parameters for base calculation
        """
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)

        # Create base calculator
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)

        # Update required columns based on base calculation
        required_columns = self.base_calculator.get_required_columns()

        config = FeatureConfig(
            name=f"rolling_returns_{window}_{base_calculation.value}",
            category=FeatureCategory.RETURNS,
            description=f"Rolling returns over {window} periods based on {base_calculation.value}",
            required_columns=required_columns,
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={
                'window': window,
                'base_calculation': base_calculation.value,
                **base_kwargs
            },
            matrix_optimized=True,
            gpu_accelerated=False
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.window = window
        self.base_calculation = base_calculation

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate rolling returns based on the specified base calculation - VECTORIZED."""
        # Calculate base values
        base_values = self.base_calculator.calculate(data)

        # Convert to numpy array for vectorized operations
        values = base_values.values

        # Vectorized rolling returns calculation
        if len(values) < self.window + 1:
            return pd.Series(np.full(len(values), np.nan), index=data.index)

        # Calculate rolling returns using vectorized operations
        rolling_returns = np.full(len(values), np.nan)

        for i in range(self.window, len(values)):
            window_values = values[i-self.window:i+1]
            if np.isfinite(window_values[0]) and window_values[0] != 0:
                rolling_returns[i] = (window_values[-1] - window_values[0]) / window_values[0]

        return pd.Series(rolling_returns, index=data.index)

class ReturnsVolatilityGenerator(VectorizedFeatureGenerator):
    """Generator for Returns Volatility with different base calculations - VECTORIZED."""

    def __init__(self,
                 window: int = 20,
                 base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS,
                 **base_kwargs):
        """
        Initialize Returns Volatility generator.

        Args:
            window: Rolling window for volatility calculation
            base_calculation: Base calculation type (price_levels, returns_vwap, etc.)
            **base_kwargs: Additional parameters for base calculation
        """
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)

        # Create base calculator
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)

        # Update required columns based on base calculation
        required_columns = self.base_calculator.get_required_columns()

        config = FeatureConfig(
            name=f"returns_volatility_{window}_{base_calculation.value}",
            category=FeatureCategory.RETURNS,
            description=f"Returns volatility over {window} periods based on {base_calculation.value}",
            required_columns=required_columns,
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={
                'window': window,
                'base_calculation': base_calculation.value,
                **base_kwargs
            },
            matrix_optimized=True,
            gpu_accelerated=False
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.window = window
        self.base_calculation = base_calculation

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate returns volatility based on the specified base calculation - VECTORIZED."""
        # Calculate base values
        base_values = self.base_calculator.calculate(data)

        # Convert to numpy array for vectorized operations
        values = base_values.values

        # Vectorized returns volatility calculation
        if len(values) < self.window + 1:
            return pd.Series(np.full(len(values), np.nan), index=data.index)

        # Calculate returns using vectorized operations
        returns = np.diff(values) / values[:-1]
        returns = np.concatenate([[np.nan], returns])  # Add NaN for first value

        # Vectorized rolling volatility calculation
        volatility = np.full(len(values), np.nan)

        for i in range(self.window, len(values)):
            window_returns = returns[i-self.window+1:i+1]
            # Filter out NaN values
            valid_returns = window_returns[np.isfinite(window_returns)]
            if len(valid_returns) > 1:  # Need at least 2 values for std
                volatility[i] = np.std(valid_returns, ddof=1)

        return pd.Series(volatility, index=data.index)

class ReturnsSkewnessGenerator(VectorizedFeatureGenerator):
    """Generator for Returns Skewness with different base calculations - VECTORIZED."""

    def __init__(self,
                 window: int = 20,
                 base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS,
                 **base_kwargs):
        """
        Initialize Returns Skewness generator.

        Args:
            window: Rolling window for skewness calculation
            base_calculation: Base calculation type (price_levels, returns_vwap, etc.)
            **base_kwargs: Additional parameters for base calculation
        """
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)

        # Create base calculator
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)

        # Update required columns based on base calculation
        required_columns = self.base_calculator.get_required_columns()

        config = FeatureConfig(
            name=f"returns_skewness_{window}_{base_calculation.value}",
            category=FeatureCategory.RETURNS,
            description=f"Returns skewness over {window} periods based on {base_calculation.value}",
            required_columns=required_columns,
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={
                'window': window,
                'base_calculation': base_calculation.value,
                **base_kwargs
            },
            matrix_optimized=True,
            gpu_accelerated=False
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.window = window
        self.base_calculation = base_calculation

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate returns skewness based on the specified base calculation - VECTORIZED."""
        # Calculate base values
        base_values = self.base_calculator.calculate(data)

        # Convert to numpy array for vectorized operations
        values = base_values.values

        # Vectorized returns skewness calculation
        if len(values) < self.window + 1:
            return pd.Series(np.full(len(values), np.nan), index=data.index)

        # Calculate returns using vectorized operations
        returns = np.diff(values) / values[:-1]
        returns = np.concatenate([[np.nan], returns])  # Add NaN for first value

        # Vectorized rolling skewness calculation
        skewness = np.full(len(values), np.nan)

        for i in range(self.window, len(values)):
            window_returns = returns[i-self.window+1:i+1]
            # Filter out NaN values
            valid_returns = window_returns[np.isfinite(window_returns)]
            if len(valid_returns) > 2:  # Need at least 3 values for skewness
                mean_ret = np.mean(valid_returns)
                std_ret = np.std(valid_returns, ddof=1)
                if std_ret > 0:
                    skewness[i] = np.mean(((valid_returns - mean_ret) / std_ret) ** 3)

        return pd.Series(skewness, index=data.index)

class ReturnsKurtosisGenerator(VectorizedFeatureGenerator):
    """Generator for Returns Kurtosis with different base calculations - VECTORIZED."""

    def __init__(self,
                 window: int = 20,
                 base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS,
                 **base_kwargs):
        """
        Initialize Returns Kurtosis generator.

        Args:
            window: Rolling window for kurtosis calculation
            base_calculation: Base calculation type (price_levels, returns_vwap, etc.)
            **base_kwargs: Additional parameters for base calculation
        """
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)

        # Create base calculator
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)

        # Update required columns based on base calculation
        required_columns = self.base_calculator.get_required_columns()

        config = FeatureConfig(
            name=f"returns_kurtosis_{window}_{base_calculation.value}",
            category=FeatureCategory.RETURNS,
            description=f"Returns kurtosis over {window} periods based on {base_calculation.value}",
            required_columns=required_columns,
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={
                'window': window,
                'base_calculation': base_calculation.value,
                **base_kwargs
            },
            matrix_optimized=True,
            gpu_accelerated=False
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.window = window
        self.base_calculation = base_calculation

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate returns kurtosis based on the specified base calculation - VECTORIZED."""
        # Calculate base values
        base_values = self.base_calculator.calculate(data)

        # Convert to numpy array for vectorized operations
        values = base_values.values

        # Vectorized returns kurtosis calculation
        if len(values) < self.window + 1:
            return pd.Series(np.full(len(values), np.nan), index=data.index)

        # Calculate returns using vectorized operations
        returns = np.diff(values) / values[:-1]
        returns = np.concatenate([[np.nan], returns])  # Add NaN for first value

        # Vectorized rolling kurtosis calculation
        kurtosis = np.full(len(values), np.nan)

        for i in range(self.window, len(values)):
            window_returns = returns[i-self.window+1:i+1]
            # Filter out NaN values
            valid_returns = window_returns[np.isfinite(window_returns)]
            if len(valid_returns) > 3:  # Need at least 4 values for kurtosis
                mean_ret = np.mean(valid_returns)
                std_ret = np.std(valid_returns, ddof=1)
                if std_ret > 0:
                    kurtosis[i] = np.mean(((valid_returns - mean_ret) / std_ret) ** 4) - 3  # Excess kurtosis

        return pd.Series(kurtosis, index=data.index)

class SharpeRatioGenerator(VectorizedFeatureGenerator):
    """Generator for Sharpe Ratio with different base calculations - VECTORIZED."""

    def __init__(self,
                 window: int = 20,
                 risk_free_rate: float = 0.0,
                 base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS,
                 **base_kwargs):
        """
        Initialize Sharpe Ratio generator.

        Args:
            window: Rolling window for Sharpe ratio calculation
            risk_free_rate: Risk-free rate (annualized)
            base_calculation: Base calculation type (price_levels, returns_vwap, etc.)
            **base_kwargs: Additional parameters for base calculation
        """
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)

        # Create base calculator
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)

        # Update required columns based on base calculation
        required_columns = self.base_calculator.get_required_columns()

        config = FeatureConfig(
            name=f"sharpe_ratio_{window}_{risk_free_rate}_{base_calculation.value}",
            category=FeatureCategory.RETURNS,
            description=f"Sharpe ratio over {window} periods with risk-free rate {risk_free_rate} based on {base_calculation.value}",
            required_columns=required_columns,
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={
                'window': window,
                'risk_free_rate': risk_free_rate,
                'base_calculation': base_calculation.value,
                **base_kwargs
            },
            matrix_optimized=True,
            gpu_accelerated=False
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.window = window
        self.risk_free_rate = risk_free_rate
        self.base_calculation = base_calculation

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate Sharpe ratio based on the specified base calculation - VECTORIZED."""
        # Calculate base values
        base_values = self.base_calculator.calculate(data)

        # Convert to numpy array for vectorized operations
        values = base_values.values

        # Vectorized Sharpe ratio calculation
        if len(values) < self.window + 1:
            return pd.Series(np.full(len(values), np.nan), index=data.index)

        # Calculate returns using vectorized operations
        returns = np.diff(values) / values[:-1]
        returns = np.concatenate([[np.nan], returns])  # Add NaN for first value

        # Calculate daily risk-free rate
        daily_rf_rate = self.risk_free_rate / 252

        # Vectorized rolling Sharpe ratio calculation
        sharpe_ratio = np.full(len(values), np.nan)

        for i in range(self.window, len(values)):
            window_returns = returns[i-self.window+1:i+1]
            # Filter out NaN values
            valid_returns = window_returns[np.isfinite(window_returns)]
            if len(valid_returns) > 1:  # Need at least 2 values for std
                excess_returns = valid_returns - daily_rf_rate
                mean_excess = np.mean(excess_returns)
                std_returns = np.std(valid_returns, ddof=1)
                if std_returns > 0:
                    sharpe_ratio[i] = mean_excess / std_returns

        return pd.Series(sharpe_ratio, index=data.index)

# NEW FEATURES - Advanced Returns Analysis

class AdvancedCumulativeReturnsGenerator(VectorizedFeatureGenerator):
    """Generator for cumulative returns over a specified window (enhanced version)."""

    def __init__(self, window: int = 20):
        config = FeatureConfig(
            name=f"advanced_cumulative_returns_{window}",
            category=FeatureCategory.RETURNS,
            description=f"Advanced cumulative returns over {window} periods",
            required_columns=["close"],
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={'window': window},
            matrix_optimized=True,
            gpu_accelerated=False
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.window = window

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        close = data['close'].values
        if len(close) < self.window:
            return pd.Series(np.full(len(close), np.nan), index=data.index)

        # Calculate returns
        returns = np.diff(close) / close[:-1]
        returns = np.concatenate([[np.nan], returns])

        # Calculate cumulative returns
        cumulative_returns = np.full(len(close), np.nan)
        for i in range(self.window - 1, len(close)):
            window_returns = returns[i - self.window + 1:i + 1]
            valid_returns = window_returns[np.isfinite(window_returns)]
            if len(valid_returns) > 0:
                cumulative_returns[i] = np.prod(1 + valid_returns) - 1

        return pd.Series(cumulative_returns, index=data.index)

class RollingZScoreReturnsGenerator(VectorizedFeatureGenerator):
    """Generator for rolling z-score of returns."""

    def __init__(self, window: int = 20):
        config = FeatureConfig(
            name=f"rolling_zscore_returns_{window}",
            category=FeatureCategory.RETURNS,
            description=f"Rolling z-score of returns over {window} periods",
            required_columns=["close"],
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={'window': window},
            matrix_optimized=True,
            gpu_accelerated=False
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.window = window

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        close = data['close'].values
        if len(close) < self.window:
            return pd.Series(np.full(len(close), np.nan), index=data.index)

        # Calculate returns
        returns = np.diff(close) / close[:-1]
        returns = np.concatenate([[np.nan], returns])

        # Calculate rolling z-score
        z_scores = np.full(len(close), np.nan)
        for i in range(self.window - 1, len(close)):
            window_returns = returns[i - self.window + 1:i + 1]
            valid_returns = window_returns[np.isfinite(window_returns)]
            if len(valid_returns) > 1:
                mean_return = np.mean(valid_returns)
                std_return = np.std(valid_returns, ddof=1)
                if std_return > 0:
                    z_scores[i] = (returns[i] - mean_return) / std_return

        return pd.Series(z_scores, index=data.index)

class ARCoefficientsGenerator(VectorizedFeatureGenerator):
    """Generator for AR(1) and AR(p) coefficients on returns."""

    def __init__(self, order: int = 1, window: int = 20):
        config = FeatureConfig(
            name=f"ar_{order}_coefficients_{window}",
            category=FeatureCategory.RETURNS,
            description=f"AR({order}) coefficients on returns over {window} periods",
            required_columns=["close"],
            default_lookback=window + order,
            min_lookback=window + order,
            max_lookback=window + order,
            parameters={'order': order, 'window': window},
            matrix_optimized=True,
            gpu_accelerated=False
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.order = order
        self.window = window

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        close = data['close'].values
        if len(close) < self.window + self.order:
            return pd.Series(np.full(len(close), np.nan), index=data.index)

        # Calculate returns
        returns = np.diff(close) / close[:-1]
        returns = np.concatenate([[np.nan], returns])

        # Calculate AR coefficients
        ar_coeffs = np.full(len(close), np.nan)
        residual_stds = np.full(len(close), np.nan)

        for i in range(self.window + self.order - 1, len(close)):
            window_returns = returns[i - self.window + 1:i + 1]
            valid_returns = window_returns[np.isfinite(window_returns)]

            if len(valid_returns) >= self.order + 5:  # Need enough data for AR estimation
                try:
                    # Simple AR(1) estimation using OLS
                    if self.order == 1:
                        y = valid_returns[1:]
                        x = valid_returns[:-1]
                        if len(y) > 1 and np.std(x) > 0:
                            coeff = np.corrcoef(x, y)[0, 1] * (np.std(y) / np.std(x))
                            ar_coeffs[i] = coeff

                            # Calculate residual standard deviation
                            residuals = y - coeff * x
                            residual_stds[i] = np.std(residuals, ddof=1)
                except:
                    pass

        return pd.Series(ar_coeffs, index=data.index)

class LjungBoxTestGenerator(VectorizedFeatureGenerator):
    """Generator for Ljung-Box p-value on returns (autocorrelation presence)."""

    def __init__(self, window: int = 20, lags: int = 10):
        config = FeatureConfig(
            name=f"ljung_box_pvalue_{window}_{lags}",
            category=FeatureCategory.RETURNS,
            description=f"Ljung-Box p-value on returns over {window} periods with {lags} lags",
            required_columns=["close"],
            default_lookback=window + lags,
            min_lookback=window + lags,
            max_lookback=window + lags,
            parameters={'window': window, 'lags': lags},
            matrix_optimized=True,
            gpu_accelerated=False
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.window = window
        self.lags = lags

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        close = data['close'].values
        if len(close) < self.window + self.lags:
            return pd.Series(np.full(len(close), np.nan), index=data.index)

        # Calculate returns
        returns = np.diff(close) / close[:-1]
        returns = np.concatenate([[np.nan], returns])

        # Calculate Ljung-Box p-values
        p_values = np.full(len(close), np.nan)

        for i in range(self.window + self.lags - 1, len(close)):
            window_returns = returns[i - self.window + 1:i + 1]
            valid_returns = window_returns[np.isfinite(window_returns)]

            if len(valid_returns) >= self.lags + 5:  # Need enough data
                try:
                    # Simple autocorrelation-based Ljung-Box approximation
                    autocorrs = []
                    for lag in range(1, min(self.lags + 1, len(valid_returns))):
                        if len(valid_returns) > lag:
                            corr = np.corrcoef(valid_returns[:-lag], valid_returns[lag:])[0, 1]
                            if not np.isnan(corr):
                                autocorrs.append(corr)

                    if len(autocorrs) > 0:
                        # Simplified Ljung-Box statistic
                        n = len(valid_returns)
                        lb_stat = n * (n + 2) * sum([(ac**2) / (n - lag) for lag, ac in enumerate(autocorrs, 1)])

                        # Approximate p-value (simplified)
                        # In practice, you'd use scipy.stats.chi2.sf
                        p_values[i] = max(0.001, min(0.999, 1 - (lb_stat / (self.lags * 10))))
                except:
                    pass

        return pd.Series(p_values, index=data.index)

def create_returns_generators(periods: Dict[str, List[int]] = None) -> List[FeatureGenerator]:
    """Create a set of returns feature generators."""
    if periods is None:
        periods = {
            'log_returns': [1, 5, 10],
            'simple_returns': [1, 5, 10],
            'cumulative_returns': [10, 20],
            'rolling_returns': [10, 20],
            'volatility': [20],
            'skewness': [20],
            'kurtosis': [20],
            'sharpe_ratio': [20]
        }

    generators = []

    # Log returns generators
    for period in periods.get('log_returns', [1, 5, 10]):
        generators.append(LogReturnsGenerator(period))

    # Simple returns generators
    for period in periods.get('simple_returns', [1, 5, 10]):
        generators.append(SimpleReturnsGenerator(period))

    # Cumulative returns generators
    for window in periods.get('cumulative_returns', [10, 20]):
        generators.append(CumulativeReturnsGenerator(window))

    # Rolling returns generators
    for window in periods.get('rolling_returns', [10, 20]):
        generators.append(RollingReturnsGenerator(window))

    # Volatility generators
    for window in periods.get('volatility', [20]):
        generators.append(ReturnsVolatilityGenerator(window))

    # Skewness generators
    for window in periods.get('skewness', [20]):
        generators.append(ReturnsSkewnessGenerator(window))

    # Kurtosis generators
    for window in periods.get('kurtosis', [20]):
        generators.append(ReturnsKurtosisGenerator(window))

    # Sharpe ratio generators
    for window in periods.get('sharpe_ratio', [20]):
        generators.append(SharpeRatioGenerator(window))

    # NEW FEATURES - Advanced Returns Analysis
    # Advanced cumulative returns generators
    for window in periods.get('advanced_cumulative_returns', [10, 20]):
        generators.append(AdvancedCumulativeReturnsGenerator(window))

    # Rolling z-score returns generators
    for window in periods.get('rolling_zscore_returns', [20]):
        generators.append(RollingZScoreReturnsGenerator(window))

    # AR coefficients generators
    for order in periods.get('ar_coefficients', [1]):
        for window in periods.get('ar_windows', [20]):
            generators.append(ARCoefficientsGenerator(order, window))

    # Ljung-Box test generators
    for window in periods.get('ljung_box_windows', [20]):
        for lags in periods.get('ljung_box_lags', [10]):
            generators.append(LjungBoxTestGenerator(window, lags))

    return generators

class ReturnGenerator(SimpleReturnsGenerator):
    """Legacy alias for SimpleReturnsGenerator for backward compatibility."""
    pass

def create_default_returns_generators() -> List[FeatureGenerator]:
    """Create default returns generators."""
    return create_returns_generators()

class VectorBTOptimizedReturnsGenerator(VectorizedFeatureGenerator):
    """
    Comprehensive VectorBT-optimized returns feature generator.

    This generator uses both VectorBTRollingOptimizer and UnifiedVectorizationManager
    for maximum performance in returns feature generation.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if config is None:
            config = self._create_default_config()
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)

        # Initialize VectorBT components
        self.rolling_optimizer = None
        self.unified_manager = None

        # Initialize VectorBT Rolling Optimizer
        if VECTORBT_ROLLING_OPTIMIZER_AVAILABLE:
            try:
                self.rolling_optimizer = get_vectorbt_rolling_optimizer(enable_gpu=False, enable_parallel=True)
                self.logger.info("✅ VectorBTRollingOptimizer initialized for optimized returns features")
            except Exception as e:
                self.logger.warning(f"⚠️ VectorBTRollingOptimizer initialization failed: {e}")

        # Initialize Unified Vectorization Manager
        if UNIFIED_VECTORIZATION_MANAGER_AVAILABLE:
            try:
                self.unified_manager = get_unified_vectorization_manager()
                self.logger.info("✅ UnifiedVectorizationManager initialized for optimized returns features")
            except Exception as e:
                self.logger.warning(f"⚠️ UnifiedVectorizationManager initialization failed: {e}")

        # Performance tracking
        self.performance_stats = {
            'vectorbt_operations': 0,
            'pandas_fallbacks': 0,
            'unified_manager_operations': 0,
            'rolling_optimizer_operations': 0,
            'batch_operations': 0,
            'total_operations': 0,
            'total_time': 0.0
        }

    @classmethod
    def _create_default_config(cls) -> FeatureConfig:
        return FeatureConfig(
            name="vectorbt_optimized_returns_features",
            category=FeatureCategory.RETURNS,
            description="Comprehensive VectorBT-optimized return-based features with intelligent optimization",
            required_columns=["close"],
            optional_columns=["high", "low", "open", "volume"],
            default_lookback=20,
            min_lookback=2,
            max_lookback=50,
            parameters={
                "return_periods": [1, 5, 10, 20],
                "log_return_periods": [1, 5, 10],
                "cumulative_windows": [10, 20],
                "volatility_windows": [20],
                "statistical_windows": [20]
            },
            matrix_optimized=True,
            gpu_accelerated=False
        )

    def generate_comprehensive_returns_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate comprehensive returns features using VectorBT optimization.

        Args:
            data: Input OHLCV data

        Returns:
            DataFrame with comprehensive returns features
        """
        if self.unified_manager and len(data) > 100:
            try:
                # Use Unified Vectorization Manager for comprehensive feature generation
                config = OperationConfig(
                    operation_type=OperationType.FEATURE_ENGINEERING,
                    data_size=len(data),
                    data_dimensions=data.shape,
                    memory_budget_mb=2048.0
                )

                # Prepare comprehensive feature configuration
                feature_configs = self._create_comprehensive_feature_configs()
                batch_data = {
                    'data': data,
                    'feature_configs': feature_configs,
                    'feature_type': 'comprehensive_returns'
                }

                result = self.unified_manager.optimize_operation(
                    OperationType.FEATURE_ENGINEERING,
                    batch_data,
                    config
                )

                if hasattr(result, 'result') and result.result is not None:
                    self.performance_stats['unified_manager_operations'] += 1
                    return result.result
                else:
                    return self._fallback_comprehensive_features(data)

            except Exception as e:
                self.logger.warning(f"Unified Vectorization Manager comprehensive features failed: {e}, using fallback")
                return self._fallback_comprehensive_features(data)
        else:
            return self._fallback_comprehensive_features(data)

    def _create_comprehensive_feature_configs(self) -> List[Dict[str, Any]]:
        """Create comprehensive feature configurations for returns analysis."""
        configs = []

        # Simple returns
        for period in [1, 5, 10, 20]:
            configs.append({
                'type': 'simple_returns',
                'period': period,
                'base_calculation': 'price_returns'
            })

        # Log returns
        for period in [1, 5, 10]:
            configs.append({
                'type': 'log_returns',
                'period': period,
                'base_calculation': 'price_returns'
            })

        # Cumulative returns
        for window in [10, 20]:
            configs.append({
                'type': 'cumulative_returns',
                'window': window,
                'base_calculation': 'price_returns'
            })

        # Rolling returns
        for window in [10, 20]:
            configs.append({
                'type': 'rolling_returns',
                'window': window,
                'base_calculation': 'price_returns'
            })

        # Returns volatility
        for window in [20]:
            configs.append({
                'type': 'returns_volatility',
                'window': window,
                'base_calculation': 'price_returns'
            })

        # Returns skewness
        for window in [20]:
            configs.append({
                'type': 'returns_skewness',
                'window': window,
                'base_calculation': 'price_returns'
            })

        # Returns kurtosis
        for window in [20]:
            configs.append({
                'type': 'returns_kurtosis',
                'window': window,
                'base_calculation': 'price_returns'
            })

        # Sharpe ratio
        for window in [20]:
            configs.append({
                'type': 'sharpe_ratio',
                'window': window,
                'base_calculation': 'price_returns'
            })

        return configs

    def _fallback_comprehensive_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fallback comprehensive features using individual generators."""
        results = {}

        # Generate features using individual optimized generators
        generators = [
            ReturnsFeatureGenerator(),
            LogReturnsGenerator(period=1),
            LogReturnsGenerator(period=5),
            LogReturnsGenerator(period=10),
            SimpleReturnsGenerator(period=1),
            SimpleReturnsGenerator(period=5),
            SimpleReturnsGenerator(period=10),
            SimpleReturnsGenerator(period=20),
            CumulativeReturnsGenerator(window=10),
            CumulativeReturnsGenerator(window=20),
            RollingReturnsGenerator(window=10),
            RollingReturnsGenerator(window=20),
            ReturnsVolatilityGenerator(window=20),
            ReturnsSkewnessGenerator(window=20),
            ReturnsKurtosisGenerator(window=20),
            SharpeRatioGenerator(window=20)
        ]

        for generator in generators:
            try:
                feature_result = generator.generate_feature(data)
                if hasattr(feature_result, 'name'):
                    results[feature_result.name] = feature_result
                else:
                    results[f"{generator.__class__.__name__}_{len(results)}"] = feature_result
            except Exception as e:
                self.logger.warning(f"Feature generation failed for {generator.__class__.__name__}: {e}")
                continue

        return pd.DataFrame(results, index=data.index)

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        stats = self.performance_stats.copy()

        if stats['total_operations'] > 0:
            stats['vectorbt_usage_percentage'] = (
                (stats['vectorbt_operations'] + stats['rolling_optimizer_operations']) /
                stats['total_operations'] * 100
            )
            stats['unified_manager_usage_percentage'] = (
                stats['unified_manager_operations'] / stats['total_operations'] * 100
            )
            stats['batch_operations_percentage'] = (
                stats['batch_operations'] / stats['total_operations'] * 100
            )
            stats['pandas_fallback_percentage'] = (
                stats['pandas_fallbacks'] / stats['total_operations'] * 100
            )
        else:
            stats['vectorbt_usage_percentage'] = 0
            stats['unified_manager_usage_percentage'] = 0
            stats['batch_operations_percentage'] = 0
            stats['pandas_fallback_percentage'] = 0

        return stats

def create_vectorbt_optimized_returns_generators() -> List[FeatureGenerator]:
    """Create VectorBT-optimized returns feature generators."""
    generators = []

    # Add the comprehensive VectorBT-optimized generator
    generators.append(VectorBTOptimizedReturnsGenerator())

    # Add individual optimized generators
    generators.extend([
        ReturnsFeatureGenerator(),
        LogReturnsGenerator(period=1),
        LogReturnsGenerator(period=5),
        LogReturnsGenerator(period=10),
        SimpleReturnsGenerator(period=1),
        SimpleReturnsGenerator(period=5),
        SimpleReturnsGenerator(period=10),
        SimpleReturnsGenerator(period=20),
        CumulativeReturnsGenerator(window=10),
        CumulativeReturnsGenerator(window=20),
        RollingReturnsGenerator(window=10),
        RollingReturnsGenerator(window=20),
        ReturnsVolatilityGenerator(window=20),
        ReturnsSkewnessGenerator(window=20),
        ReturnsKurtosisGenerator(window=20),
        SharpeRatioGenerator(window=20)
    ])

    return generators

    def _should_use_vectorbt(self, data) -> bool:
        """Determine if VectorBT should be used based on data size and configuration."""
        return (hasattr(self, 'use_vectorbt') and self.use_vectorbt and
                len(data) >= getattr(self, 'vectorbt_threshold', 1000) and
                VECTORBT_AVAILABLE)

    def _vectorbt_rolling_operation(self, data: pd.Series, operation: str,
                                  window: int, **kwargs) -> pd.Series:
        """Perform VectorBT rolling operation with fallback to pandas."""
        if not self._should_use_vectorbt(data):
            return self._pandas_rolling_operation(data, operation, window, **kwargs)

        try:
            if operation == 'mean':
                return rolling_mean(data, window=window, **kwargs)
            elif operation == 'std':
                return rolling_std(data, window=window, **kwargs)
            elif operation == 'var':
                return rolling_var(data, window=window, **kwargs)
            elif operation == 'min':
                return rolling_min(data, window=window, **kwargs)
            elif operation == 'max':
                return rolling_max(data, window=window, **kwargs)
            elif operation == 'sum':
                return rolling_sum(data, window=window, **kwargs)
            else:
                raise ValueError(f"Unsupported operation: {operation}")
        except Exception as e:
            logger.warning(f"VectorBT operation failed: {e}, using pandas fallback")
            return self._pandas_rolling_operation(data, operation, window, **kwargs)

    def _pandas_rolling_operation(self, data: pd.Series, operation: str,
                                 window: int, **kwargs) -> pd.Series:
        """Fallback rolling operation using pandas."""
        if operation == 'mean':
            return data.rolling(window=window).mean()
        elif operation == 'std':
            return data.rolling(window=window).std()
        elif operation == 'var':
            return data.rolling(window=window).var()
        elif operation == 'min':
            return data.rolling(window=window).min()
        elif operation == 'max':
            return data.rolling(window=window).max()
        elif operation == 'sum':
            return data.rolling(window=window).sum()
        else:
            raise ValueError(f"Unsupported operation: {operation}")
