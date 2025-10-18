"""
Momentum Feature Generator

This module provides feature generators for momentum-based indicators,
including RSI, MACD, Stochastic, and other momentum oscillators.
Wires up existing momentum features from legacy and entropy modules.

Enhanced with VectorBT for maximum performance.
"""

import numpy as np
import pandas as pd
import warnings
import logging
from typing import Any, Dict, List, Optional, Union

# Import centralized logging and error handling
from src.feature_generation.utils.centralized_logging import tprint, log_function_execution, fast_fail_error
from src.feature_generation.utils.error_handling import (
    DataValidationError, ConfigurationError, ComputationError,
    validate_required_columns, validate_finite_values, safe_divide
)

from ..core.feature_generator import FeatureGenerator, FeatureResult, VectorizedFeatureGenerator, FeatureConfig, FeatureCategory
from ..core.vectorbt_feature_generator import VectorBTFeatureGenerator, VECTORBT_AVAILABLE

# Optimization utilities
try:
    from src.feature_generation.utils.vectorization_optimizer import get_vectorization_optimizer
    from src.feature_generation.utils.optimized_feature_pipeline import get_optimized_feature_pipeline
    OPTIMIZATION_AVAILABLE = True
except ImportError:
    OPTIMIZATION_AVAILABLE = False
from ..base_calculations import (
    BaseCalculator,
    BaseCalculationType,
    BaseCalculationConfig,
    create_base_calculator
)
from src.feature_generation.utils.math_validation import safe_divide, validate_finite, safe_percentage_change

# Legacy features are imported separately to avoid circular imports
from .entropy import (
    RSIEntropyGenerator,
    MACDEntropyGenerator
)

# VectorBT imports for native optimization
try:
    import vectorbt as vbt
    # VectorBT 0.28.1 doesn't have these functions in vectorbt.generic - will use pandas fallbacks
    rolling_mean = None
    rolling_std = None
    rolling_var = None
    rolling_min = None
    rolling_max = None
    rolling_sum = None
    rolling_apply = None
    rolling_corr = None
    rolling_cov = None
    scale = None
    rank = None
    zscore = None
    winsorize = None
    clip = None
    quantile = None
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None
    warnings.warn("VectorBT not available. Install with: pip install vectorbt for optimized performance")

# Import optimized rolling operations - NOW USING NEW OPTIMIZED VERSION
try:
    from src.feature_generation.utils.consolidated_rolling_optimizer import (
        ConsolidatedRollingOptimizer as VectorBTRollingOptimizer,
        get_global_rolling_optimizer as get_vectorbt_rolling_optimizer,
        RollingOperationConfig,
        RollingOperationType
    )
    from src.feature_generation.utils.statistical_calculations_optimizer import (
        StatisticalCalculationsOptimizer as VectorizationOptimizer,
        get_global_statistical_optimizer as get_vectorization_optimizer,
        StatisticalOperationConfig,
        StatisticalOperationType
    )
    ROLLING_OPTIMIZER_AVAILABLE = True
    OPTIMIZATION_AVAILABLE = True
except ImportError:
    # Fallback to legacy if new version not available
    try:
        from src.feature_generation.utils.vectorbt_rolling_optimizer import (
            get_vectorbt_rolling_optimizer,
            optimized_rolling_mean,
            optimized_rolling_std,
            optimized_rolling_var,
            optimized_rolling_min,
            optimized_rolling_max,
            optimized_rolling_sum,
            optimized_rolling_apply,
            optimized_rolling_corr,
            optimized_rolling_cov
        )
        ROLLING_OPTIMIZER_AVAILABLE = True
        OPTIMIZATION_AVAILABLE = False
    except ImportError:
        ROLLING_OPTIMIZER_AVAILABLE = False
        OPTIMIZATION_AVAILABLE = False

# Import Unified Vectorization Manager
try:
    from src.utils.ml_common.unified_vectorization_manager import (
        get_unified_vectorization_manager,
        OperationType,
        OperationConfig,
        optimize_financial_operation
    )
    UNIFIED_VECTORIZATION_AVAILABLE = True
except ImportError:
    UNIFIED_VECTORIZATION_AVAILABLE = False

logger = logging.getLogger(__name__)

# Centralized utility imports
from src.feature_generation.utils.vectorbt_rolling_optimizer import get_vectorbt_rolling_optimizer
# Removed VectorBTScaler import to avoid circular import
# Lazy import to avoid circular dependency
def get_global_feature_bank():
    from src.feature_generation.core.feature_bank import get_global_feature_bank as _get_global_feature_bank
    return _get_global_feature_bank()

# LegacyRSIGenerator, LegacyMACDGenerator, LegacyStochasticGenerator, and LegacyWilliamsRGenerator removed - use VectorBT versions instead

class MomentumFeatureGenerator(VectorizedFeatureGenerator):
    """Feature generator for momentum-based features with optimization."""

    def __init__(self, config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config()
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)

        # Initialize VectorBT rolling optimizer
        if ROLLING_OPTIMIZER_AVAILABLE:
            self.rolling_optimizer = get_vectorbt_rolling_optimizer(enable_gpu=False, enable_parallel=True)
        else:
            self.rolling_optimizer = None

        # Initialize Unified Vectorization Manager
        if UNIFIED_VECTORIZATION_AVAILABLE:
            self.unified_manager = get_unified_vectorization_manager()
        else:
            self.unified_manager = None

    @classmethod
    def _create_default_config(cls) -> FeatureConfig:
        return FeatureConfig(
            name="momentum_features",
            category=FeatureCategory.MOMENTUM,
            description="Comprehensive momentum-based features including RSI, MACD, Stochastic, and momentum oscillators",
            required_columns=["close"],
            optional_columns=["high", "low", "open", "volume"],
            default_lookback=20,
            min_lookback=2,
            max_lookback=50,
            parameters={
                "rsi_periods": [14],
                "macd_fast": [12],
                "macd_slow": [26],
                "stochastic_periods": [14],
                "momentum_windows": [10, 20]
            },
            matrix_optimized=True,
            gpu_accelerated=False
        )

    @classmethod
    def create_default(cls) -> 'MomentumFeatureGenerator':
        return cls()

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        # Optimize DataFrame for processing
        optimized_data = self.optimize_dataframe_processing(data)
        close_prices = optimized_data['close'].values
        momentum = self._calculate_momentum(close_prices, period=20)
        return pd.Series(momentum, index=data.index, name='momentum_20')

    def _calculate_momentum(self, prices: np.ndarray, period: int = 20) -> np.ndarray:
        if len(prices) < period:
            return np.full(len(prices), np.nan)

        prices_series = pd.Series(prices)

        # Use Unified Vectorization Manager for optimized momentum calculation
        if self.unified_manager and self._should_use_vectorbt(pd.DataFrame({'prices': prices_series})):
            try:
                # Use Unified Vectorization Manager for optimized momentum calculation
                momentum_result = self.unified_manager.optimize_operation(
                    OperationType.TECHNICAL_INDICATORS,
                    {
                        'data': prices_series,
                        'operation': 'momentum',
                        'period': period,
                        'indicator_configs': {'momentum': {'period': period}}
                    },
                    OperationConfig(
                        operation_type=OperationType.TECHNICAL_INDICATORS,
                        data_size=len(prices_series),
                        data_dimensions=prices_series.shape,
                        memory_budget_mb=256.0
                    )
                )
                momentum = momentum_result.result

                # Track performance
                if hasattr(self, 'performance_stats'):
                    self.performance_stats['vectorbt_operations'] += 1
                    self.performance_stats['unified_manager_used'] = self.performance_stats.get('unified_manager_used', 0) + 1

                return momentum
            except Exception as e:
                self.logger.warning(f"Unified Vectorization Manager momentum calculation failed: {e}, using VectorBT fallback")
                if self.rolling_optimizer:
                    try:
                        shifted_prices = prices_series.shift(period)
                        momentum_series = prices_series - shifted_prices
                        return momentum_series.values
                    except Exception as e2:
                        self.logger.warning(f"VectorBTRollingOptimizer momentum calculation failed: {e2}, using numpy fallback")
                        if hasattr(self, 'performance_stats'):
                            self.performance_stats['pandas_fallbacks'] += 1
                        momentum = prices - np.roll(prices, period)
                        return momentum
                else:
                    momentum = prices - np.roll(prices, period)
                    return momentum
        elif self.rolling_optimizer and self._should_use_vectorbt(pd.DataFrame({'prices': prices_series})):
            try:
                # Calculate momentum using VectorBT rolling operations
                shifted_prices = prices_series.shift(period)
                momentum_series = prices_series - shifted_prices
                if hasattr(self, 'performance_stats'):
                    self.performance_stats['vectorbt_operations'] += 1
                return momentum_series.values
            except Exception as e:
                self.logger.warning(f"VectorBTRollingOptimizer momentum calculation failed: {e}, using numpy fallback")
                if hasattr(self, 'performance_stats'):
                    self.performance_stats['pandas_fallbacks'] += 1
                momentum = prices - np.roll(prices, period)
                return momentum
        elif VECTORBT_AVAILABLE and self._should_use_vectorbt(pd.DataFrame({'prices': prices_series})):
            try:
                # Calculate momentum using VectorBT rolling operations
                shifted_prices = prices_series.shift(period)
                momentum_series = prices_series - shifted_prices
                if hasattr(self, 'performance_stats'):
                    self.performance_stats['vectorbt_operations'] += 1
                return momentum_series.values
            except Exception as e:
                self.logger.warning(f"VectorBT momentum calculation failed: {e}, using numpy fallback")
                if hasattr(self, 'performance_stats'):
                    self.performance_stats['pandas_fallbacks'] += 1
                momentum = prices - np.roll(prices, period)
                return momentum
        else:
            momentum = prices - np.roll(prices, period)
            return momentum

    def generate_optimized_momentum_features(self, data: pd.DataFrame,
                                           feature_configs: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Generate multiple momentum features using optimized batch processing.

        Args:
            data: OHLCV data
            feature_configs: List of feature configuration dictionaries

        Returns:
            DataFrame with generated momentum features
        """
        if self.unified_manager:
            try:
                # Use Unified Vectorization Manager for batch processing
                batch_result = self.unified_manager.optimize_operation(
                    OperationType.FEATURE_ENGINEERING,
                    {
                        'data': data,
                        'feature_configs': feature_configs,
                        'operation_type': 'momentum_batch'
                    },
                    OperationConfig(
                        operation_type=OperationType.FEATURE_ENGINEERING,
                        data_size=len(data),
                        data_dimensions=data.shape,
                        memory_budget_mb=1024.0
                    )
                )
                return batch_result.result
            except Exception as e:
                self.logger.warning(f"Unified Vectorization Manager batch processing failed: {e}, using fallback")
                # Fallback to individual processing
                return self._process_momentum_features_individually(data, feature_configs)
        elif self.rolling_optimizer:
            # Fallback to individual VectorBT operations
            return self._process_momentum_features_individually(data, feature_configs)
        else:
            # Fallback to pandas operations
            return self._process_momentum_features_individually(data, feature_configs)

    def _process_momentum_features_individually(self, data: pd.DataFrame, feature_configs: List[Dict[str, Any]]) -> pd.DataFrame:
        """Process momentum features individually as fallback when batch processing fails."""
        results = {}
        for config in feature_configs:
            feature_name = config['name']
            feature_type = config.get('type', 'momentum')
            params = config.get('params', {})

            try:
                if feature_type == 'momentum':
                    period = params.get('period', 20)
                    column = params.get('column', 'close')

                    if column in data.columns:
                        prices = data[column].values
                        momentum = self._calculate_momentum(prices, period)
                        results[feature_name] = pd.Series(momentum, index=data.index)

            except Exception as e:
                self.logger.warning(f"Momentum feature {feature_name} failed: {e}")
                results[feature_name] = pd.Series(np.nan, index=data.index)

        return pd.DataFrame(results, index=data.index)

# Analyst Features - Cross-timeframe momentum generators
class AnalystMomentum5mGenerator(VectorizedFeatureGenerator):
    """Generator for 5-minute timeframe momentum feature."""

    def __init__(self, lookback: int = 20):
        config = FeatureConfig(
            name="analyst_momentum_5m",
            category=FeatureCategory.MOMENTUM,
            description="Analyst 5-minute timeframe momentum (20-period rolling mean of returns)",
            required_columns=["close"],
            default_lookback=lookback,
            min_lookback=5,
            max_lookback=100,
            parameters={"lookback": lookback}
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.lookback = lookback

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate 5-minute momentum feature using VectorBT optimization."""
        returns = data['close'].pct_change()

        # Use VectorBT for optimized rolling mean
        if VECTORBT_AVAILABLE and len(returns) > 100:
            try:
                momentum = rolling_mean(returns, window=self.lookback)
                return momentum
            except Exception as e:
                logger.warning(f"VectorBT rolling mean failed: {e}, using pandas fallback")
                momentum = returns.rolling(self.lookback).mean()
                return momentum
        else:
            momentum = returns.rolling(self.lookback).mean()
            return momentum

class AnalystMomentum15mGenerator(VectorizedFeatureGenerator):
    """Generator for 15-minute timeframe momentum feature."""

    def __init__(self, lookback: int = 20):
        config = FeatureConfig(
            name="analyst_momentum_15m",
            category=FeatureCategory.MOMENTUM,
            description="Analyst 15-minute timeframe momentum (20-period rolling mean of returns)",
            required_columns=["close"],
            default_lookback=lookback,
            min_lookback=5,
            max_lookback=100,
            parameters={"lookback": lookback}
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.lookback = lookback

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate 15-minute momentum feature using VectorBT optimization."""
        returns = data['close'].pct_change()

        # Use VectorBT for optimized rolling mean
        if VECTORBT_AVAILABLE and len(returns) > 100:
            try:
                momentum = rolling_mean(returns, window=self.lookback)
                return momentum
            except Exception as e:
                logger.warning(f"VectorBT rolling mean failed: {e}, using pandas fallback")
                momentum = returns.rolling(self.lookback).mean()
                return momentum
        else:
            momentum = returns.rolling(self.lookback).mean()
            return momentum

class AnalystMomentum1hGenerator(VectorizedFeatureGenerator):
    """Generator for 1-hour timeframe momentum feature."""

    def __init__(self, lookback: int = 20):
        config = FeatureConfig(
            name="analyst_momentum_1h",
            category=FeatureCategory.MOMENTUM,
            description="Analyst 1-hour timeframe momentum (20-period rolling mean of returns)",
            required_columns=["close"],
            default_lookback=lookback,
            min_lookback=5,
            max_lookback=100,
            parameters={"lookback": lookback}
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.lookback = lookback

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate 1-hour momentum feature."""
        returns = data['close'].pct_change()
        momentum = returns.rolling(self.lookback).mean()
        return momentum

class AnalystMomentumAlignmentGenerator(VectorizedFeatureGenerator):
    """Generator for momentum alignment across timeframes."""

    def __init__(self, lookback: int = 20):
        config = FeatureConfig(
            name="analyst_momentum_alignment",
            category=FeatureCategory.MOMENTUM,
            description="Analyst momentum alignment across 5m, 15m, and 1h timeframes",
            required_columns=["close"],
            default_lookback=lookback,
            min_lookback=5,
            max_lookback=100,
            parameters={"lookback": lookback}
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.lookback = lookback

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate momentum alignment feature."""
        returns = data['close'].pct_change()

        # Calculate momentum for different timeframes
        mom_5m = returns.rolling(self.lookback).mean()
        mom_15m = returns.rolling(self.lookback).mean()
        mom_1h = returns.rolling(self.lookback).mean()

        # Check if all momentum signals have the same sign (alignment)
        alignment = ((np.sign(mom_5m) == np.sign(mom_15m)) &
                    (np.sign(mom_15m) == np.sign(mom_1h))).astype(int)

        return alignment

class RSIGenerator(VectorizedFeatureGenerator):
    """Generator for RSI (Relative Strength Index) with different base calculations."""

    def __init__(self,
                 period: int = 14,
                 base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.RETURNS_VWAP,
                 **base_kwargs):
        """
        Initialize RSI generator.

        Args:
            period: RSI period
            base_calculation: Base calculation type (price_levels, returns_vwap, etc.)
            **base_kwargs: Additional parameters for base calculation
        """
        tprint(f"Initializing RSIGenerator with period: {period}, base_calculation: {base_calculation}")
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)

        # Create base calculator
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)

        # Update required columns based on base calculation
        required_columns = self.base_calculator.get_required_columns()

        config = FeatureConfig(
            name=f"rsi_{period}_{base_calculation.value}",
            category=FeatureCategory.MOMENTUM,
            description=f"Relative Strength Index over {period} periods based on {base_calculation.value}",
            required_columns=required_columns,
            default_lookback=period * 2,
            min_lookback=period,
            max_lookback=period * 3,
            parameters={
                'period': period,
                'base_calculation': base_calculation.value,
                **base_kwargs
            }
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.period = period
        self.base_calculation = base_calculation

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate RSI based on the specified base calculation."""
        tprint(f"Generating RSI feature with period {self.period} and base calculation {self.base_calculation}")
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)
        if self.base_calculation == BaseCalculationType.PRICE_LEVELS:
            close = data['close']

            # Traditional RSI calculation using VectorBT optimization
            from ...utils.error_handling import safe_diff
            delta = safe_diff(close)
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)

            # Use VectorBTRollingOptimizer for maximum performance
            if ROLLING_OPTIMIZER_AVAILABLE and len(close) > 100:
                try:
                    # Get the global rolling optimizer
                    rolling_optimizer = get_vectorbt_rolling_optimizer()
                    avg_gain = rolling_optimizer.rolling_mean(gain, window=self.period)
                    avg_loss = rolling_optimizer.rolling_mean(loss, window=self.period)

                    # Track performance
                    if hasattr(self, 'performance_stats'):
                        self.performance_stats['rolling_optimizer_used'] = self.performance_stats.get('rolling_optimizer_used', 0) + 2
                except Exception as e:
                    logger.warning(f"VectorBTRollingOptimizer RSI calculation failed: {e}, using VectorBT fallback")
                    if VECTORBT_AVAILABLE:
                        avg_gain = rolling_mean(gain, window=self.period)
                        avg_loss = rolling_mean(loss, window=self.period)
                    else:
                        avg_gain = gain.rolling(window=self.period).mean()
                        avg_loss = loss.rolling(window=self.period).mean()
            elif VECTORBT_AVAILABLE and len(close) > 100:
                try:
                    avg_gain = rolling_mean(gain, window=self.period)
                    avg_loss = rolling_mean(loss, window=self.period)
                except Exception as e:
                    logger.warning(f"VectorBT RSI calculation failed: {e}, using pandas fallback")
                    avg_gain = gain.rolling(window=self.period).mean()
                    avg_loss = loss.rolling(window=self.period).mean()
            else:
                avg_gain = gain.rolling(window=self.period).mean()
                avg_loss = loss.rolling(window=self.period).mean()

            rs = avg_gain / avg_loss.replace(0, 1)
            rsi = 100 - (100 / (1 + rs))

            return rsi
        else:
            # For other base calculations, calculate RSI on base values
            base_values = self.base_calculator.calculate(data)

            from ...utils.error_handling import safe_diff
            delta = safe_diff(base_values)
            gain = (delta.where(delta > 0, 0)).rolling(window=self.period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=self.period).mean()
            rs = gain / loss.replace(0, 1)
            rsi = 100 - (100 / (1 + rs))

            return rsi

class MACDGenerator(VectorizedFeatureGenerator):
    """Generator for MACD (Moving Average Convergence Divergence) with different base calculations."""

    def __init__(self,
                 fast: int = 12,
                 slow: int = 26,
                 signal: int = 9,
                 base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.RETURNS_VWAP,
                 **base_kwargs):
        """
        Initialize MACD generator.

        Args:
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line EMA period
            base_calculation: Base calculation type (price_levels, returns_vwap, etc.)
            **base_kwargs: Additional parameters for base calculation
        """
        tprint(f"Initializing MACDGenerator with fast: {fast}, slow: {slow}, signal: {signal}, base_calculation: {base_calculation}")
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)

        # Create base calculator
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)

        # Update required columns based on base calculation
        required_columns = self.base_calculator.get_required_columns()

        config = FeatureConfig(
            name=f"macd_{fast}_{slow}_{signal}_{base_calculation.value}",
            category=FeatureCategory.MOMENTUM,
            description=f"MACD {fast}/{slow}/{signal} based on {base_calculation.value}",
            required_columns=required_columns,
            default_lookback=slow * 2,
            min_lookback=slow,
            max_lookback=slow * 3,
            parameters={
                'fast': fast,
                'slow': slow,
                'signal': signal,
                'base_calculation': base_calculation.value,
                **base_kwargs
            }
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.fast = fast
        self.slow = slow
        self.signal = signal
        self.base_calculation = base_calculation

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate MACD based on the specified base calculation."""
        tprint(f"Generating MACD feature with fast: {self.fast}, slow: {self.slow}, signal: {self.signal}")
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        # Calculate base values
        base_values = self.base_calculator.calculate(data)

        # Calculate MACD
        ema_fast = base_values.ewm(span=self.fast).mean()
        ema_slow = base_values.ewm(span=self.slow).mean()
        macd = ema_fast - ema_slow

        return macd

class StochasticGenerator(VectorizedFeatureGenerator):
    """Generator for Stochastic Oscillator with different base calculations."""

    def __init__(self,
                 k_period: int = 14,
                 d_period: int = 3,
                 base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS,
                 **base_kwargs):
        """
        Initialize Stochastic generator.

        Args:
            k_period: %K period
            d_period: %D period (smoothing)
            base_calculation: Base calculation type (price_levels, returns_vwap, etc.)
            **base_kwargs: Additional parameters for base calculation
        """
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)

        # Create base calculator
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)

        # Update required columns based on base calculation
        required_columns = self.base_calculator.get_required_columns()
        if base_calculation == BaseCalculationType.PRICE_LEVELS:
            required_columns.extend(["high", "low"])

        config = FeatureConfig(
            name=f"stochastic_{k_period}_{d_period}_{base_calculation.value}",
            category=FeatureCategory.MOMENTUM,
            description=f"Stochastic Oscillator {k_period}/{d_period} based on {base_calculation.value}",
            required_columns=required_columns,
            default_lookback=k_period,
            min_lookback=k_period,
            max_lookback=k_period,
            parameters={
                'k_period': k_period,
                'd_period': d_period,
                'base_calculation': base_calculation.value,
                **base_kwargs
            }
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.k_period = k_period
        self.d_period = d_period
        self.base_calculation = base_calculation

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate Stochastic Oscillator based on the specified base calculation."""
        if self.base_calculation == BaseCalculationType.PRICE_LEVELS:
            high = data['high']
            low = data['low']
            close = data['close']

            # Traditional Stochastic calculation using VectorBTRollingOptimizer
            if ROLLING_OPTIMIZER_AVAILABLE and len(close) > 100:
                try:
                    # Get the global rolling optimizer
                    rolling_optimizer = get_vectorbt_rolling_optimizer()
                    lowest_low = rolling_optimizer.rolling_min(low, window=self.k_period)
                    highest_high = rolling_optimizer.rolling_max(high, window=self.k_period)

                    # Track performance
                    if hasattr(self, 'performance_stats'):
                        self.performance_stats['rolling_optimizer_used'] = self.performance_stats.get('rolling_optimizer_used', 0) + 2
                except Exception as e:
                    logger.warning(f"VectorBTRollingOptimizer Stochastic calculation failed: {e}, using VectorBT fallback")
                    if VECTORBT_AVAILABLE:
                        lowest_low = rolling_min(low, window=self.k_period)
                        highest_high = rolling_max(high, window=self.k_period)
                    else:
                        lowest_low = low.rolling(window=self.k_period).min()
                        highest_high = high.rolling(window=self.k_period).max()
            elif VECTORBT_AVAILABLE and len(close) > 100:
                try:
                    lowest_low = rolling_min(low, window=self.k_period)
                    highest_high = rolling_max(high, window=self.k_period)
                except Exception as e:
                    logger.warning(f"VectorBT Stochastic calculation failed: {e}, using pandas fallback")
                    lowest_low = low.rolling(window=self.k_period).min()
                    highest_high = high.rolling(window=self.k_period).max()
            else:
                lowest_low = low.rolling(window=self.k_period).min()
                highest_high = high.rolling(window=self.k_period).max()

            # Handle division by zero when highest_high equals lowest_low
            with np.errstate(divide='ignore', invalid='ignore'):
                k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
                k_percent = np.where(highest_high == lowest_low, np.nan, k_percent)

            return k_percent
        else:
            # For other base calculations, use rolling min/max on base values
            base_values = self.base_calculator.calculate(data)

            lowest_low = base_values.rolling(window=self.k_period).min()
            highest_high = base_values.rolling(window=self.k_period).max()
            # Handle division by zero when highest_high equals lowest_low
            with np.errstate(divide='ignore', invalid='ignore'):
                k_percent = 100 * ((base_values - lowest_low) / (highest_high - lowest_low))
                k_percent = np.where(highest_high == lowest_low, np.nan, k_percent)

            return k_percent

class WilliamsRGenerator(VectorizedFeatureGenerator):
    """Generator for Williams %R with different base calculations."""

    def __init__(self,
                 period: int = 14,
                 base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS,
                 **base_kwargs):
        """
        Initialize Williams %R generator.

        Args:
            period: Williams %R period
            base_calculation: Base calculation type (price_levels, returns_vwap, etc.)
            **base_kwargs: Additional parameters for base calculation
        """
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)

        # Create base calculator
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)

        # Update required columns based on base calculation
        required_columns = self.base_calculator.get_required_columns()
        if base_calculation == BaseCalculationType.PRICE_LEVELS:
            required_columns.extend(["high", "low"])

        config = FeatureConfig(
            name=f"williams_r_{period}_{base_calculation.value}",
            category=FeatureCategory.MOMENTUM,
            description=f"Williams %R over {period} periods based on {base_calculation.value}",
            required_columns=required_columns,
            default_lookback=period,
            min_lookback=period,
            max_lookback=period,
            parameters={
                'period': period,
                'base_calculation': base_calculation.value,
                **base_kwargs
            }
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.period = period
        self.base_calculation = base_calculation

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate Williams %R based on the specified base calculation."""
        if self.base_calculation == BaseCalculationType.PRICE_LEVELS:
            high = data['high']
            low = data['low']
            close = data['close']

            # Williams %R calculation using VectorBTRollingOptimizer
            if ROLLING_OPTIMIZER_AVAILABLE and len(close) > 100:
                try:
                    # Get the global rolling optimizer
                    rolling_optimizer = get_vectorbt_rolling_optimizer()
                    highest_high = rolling_optimizer.rolling_max(high, window=self.period)
                    lowest_low = rolling_optimizer.rolling_min(low, window=self.period)

                    # Track performance
                    if hasattr(self, 'performance_stats'):
                        self.performance_stats['rolling_optimizer_used'] = self.performance_stats.get('rolling_optimizer_used', 0) + 2
                except Exception as e:
                    logger.warning(f"VectorBTRollingOptimizer Williams %R calculation failed: {e}, using VectorBT fallback")
                    if VECTORBT_AVAILABLE:
                        highest_high = rolling_max(high, window=self.period)
                        lowest_low = rolling_min(low, window=self.period)
                    else:
                        highest_high = high.rolling(window=self.period).max()
                        lowest_low = low.rolling(window=self.period).min()
            elif VECTORBT_AVAILABLE and len(close) > 100:
                try:
                    highest_high = rolling_max(high, window=self.period)
                    lowest_low = rolling_min(low, window=self.period)
                except Exception as e:
                    logger.warning(f"VectorBT Williams %R calculation failed: {e}, using pandas fallback")
                    highest_high = high.rolling(window=self.period).max()
                    lowest_low = low.rolling(window=self.period).min()
            else:
                highest_high = high.rolling(window=self.period).max()
                lowest_low = low.rolling(window=self.period).min()

            # Handle division by zero when highest_high equals lowest_low
            with np.errstate(divide='ignore', invalid='ignore'):
                williams_r = -100 * ((highest_high - close) / (highest_high - lowest_low))
                williams_r = np.where(highest_high == lowest_low, np.nan, williams_r)

            return williams_r
        else:
            # For other base calculations, use rolling min/max on base values
            base_values = self.base_calculator.calculate(data)

            highest_high = base_values.rolling(window=self.period).max()
            lowest_low = base_values.rolling(window=self.period).min()
            # Handle division by zero when highest_high equals lowest_low
            with np.errstate(divide='ignore', invalid='ignore'):
                williams_r = -100 * ((highest_high - base_values) / (highest_high - lowest_low))
                williams_r = np.where(highest_high == lowest_low, np.nan, williams_r)

            return williams_r

class MomentumOscillatorGenerator(VectorizedFeatureGenerator):
    """Generator for Momentum Oscillator with different base calculations."""

    def __init__(self,
                 period: int = 10,
                 base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS,
                 **base_kwargs):
        """
        Initialize Momentum Oscillator generator.

        Args:
            period: Momentum period
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
            name=f"momentum_{period}_{base_calculation.value}",
            category=FeatureCategory.MOMENTUM,
            description=f"Momentum Oscillator over {period} periods based on {base_calculation.value}",
            required_columns=required_columns,
            default_lookback=period,
            min_lookback=period,
            max_lookback=period,
            parameters={
                'period': period,
                'base_calculation': base_calculation.value,
                **base_kwargs
            }
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.period = period
        self.base_calculation = base_calculation

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate Momentum Oscillator based on the specified base calculation."""
        # Calculate base values
        base_values = self.base_calculator.calculate(data)

        # Calculate momentum with proper handling of initial NaN values
        momentum = base_values - base_values.shift(self.period)
        
        # Set the first 'period' values to 0 instead of NaN for better feature quality
        momentum.iloc[:self.period] = 0.0

        return momentum

class RateOfChangeGenerator(VectorizedFeatureGenerator):
    """Generator for Rate of Change (ROC) with different base calculations."""

    def __init__(self,
                 period: int = 10,
                 base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS,
                 **base_kwargs):
        """
        Initialize ROC generator.

        Args:
            period: ROC period
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
            name=f"roc_{period}_{base_calculation.value}",
            category=FeatureCategory.MOMENTUM,
            description=f"Rate of Change over {period} periods based on {base_calculation.value}",
            required_columns=required_columns,
            default_lookback=period,
            min_lookback=period,
            max_lookback=period,
            parameters={
                'period': period,
                'base_calculation': base_calculation.value,
                **base_kwargs
            }
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.period = period
        self.base_calculation = base_calculation

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate ROC based on the specified base calculation."""
        # Calculate base values
        base_values = self.base_calculator.calculate(data)

        # Calculate ROC with proper handling of initial NaN values
        shifted_values = base_values.shift(self.period)
        
        # Use pandas pct_change for more robust calculation
        roc_series = base_values.pct_change(periods=self.period) * 100
        
        # Replace non-finite values caused by zero denominators and set leading window to 0
        roc_series = roc_series.replace([np.inf, -np.inf], np.nan)
        roc_series = roc_series.fillna(0.0)
        roc_series.iloc[:self.period] = 0.0
        roc_series.name = f'roc_{self.period}_{self.base_calculation.value}'

        # Validate that all values are finite and provide detailed information
        try:
            validate_finite(roc_series.values, f"ROC_{self.period}_{self.base_calculation.value}")
        except ValueError as e:
            # Get detailed information about where the NaN/inf values are
            non_finite_mask = ~np.isfinite(roc_series.values)
            if np.any(non_finite_mask):
                non_finite_indices = np.where(non_finite_mask)[0]
                total_count = len(non_finite_indices)

                # Show first few and last few problematic indices
                if total_count <= 10:
                    indices_str = f"indices {non_finite_indices.tolist()}"
                else:
                    first_5 = non_finite_indices[:5].tolist()
                    last_5 = non_finite_indices[-5:].tolist()
                    indices_str = f"indices {first_5} ... {last_5} (total: {total_count})"

                # Only log once per feature globally to reduce verbosity
                feature_key = f"ROC_{self.period}_{self.base_calculation.value}"
                # Use class-level tracking to prevent duplicate warnings across all instances
                if not hasattr(RateOfChangeGenerator, '_logged_warnings'):
                    RateOfChangeGenerator._logged_warnings = set()
                if feature_key not in RateOfChangeGenerator._logged_warnings:
                    self.logger.warning(f"⚠️ {e} - {indices_str}")
                    RateOfChangeGenerator._logged_warnings.add(feature_key)
            else:
                self.logger.warning(f"⚠️ {e}")

        return roc_series

# NEW FEATURES - Advanced Momentum Analysis

class MomentumEndpointsGenerator(VectorizedFeatureGenerator):
    """Generator for momentum endpoints - price distance to moving averages as percentage."""

    def __init__(self, ma_period: int = 20, ma_type: str = 'SMA'):
        config = FeatureConfig(
            name=f"momentum_endpoints_{ma_type.lower()}_{ma_period}",
            category=FeatureCategory.MOMENTUM,
            description=f"Price distance to {ma_type} as percentage over {ma_period} periods",
            required_columns=["close"],
            default_lookback=ma_period,
            min_lookback=ma_period,
            max_lookback=ma_period,
            parameters={'ma_period': ma_period, 'ma_type': ma_type},
            matrix_optimized=True,
            gpu_accelerated=False
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.ma_period = ma_period
        self.ma_type = ma_type

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        close = data['close'].values
        if len(close) < self.ma_period:
            return pd.Series(np.full(len(close), np.nan), index=data.index)

        # Calculate moving average
        if self.ma_type.upper() == 'SMA':
            ma = self._calculate_sma(close, self.ma_period)
        elif self.ma_type.upper() == 'EMA':
            ma = self._calculate_ema(close, self.ma_period)
        else:  # KAMA
            ma = self._calculate_kama(close, self.ma_period)

        # Calculate percentage distance
        momentum_endpoints = np.full(len(close), np.nan)
        for i in range(self.ma_period - 1, len(close)):
            if not np.isnan(ma[i]) and ma[i] != 0:
                momentum_endpoints[i] = ((close[i] - ma[i]) / ma[i]) * 100

        return pd.Series(momentum_endpoints, index=data.index)

    def _calculate_sma(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Calculate Simple Moving Average."""
        sma = np.full(len(prices), np.nan)
        for i in range(period - 1, len(prices)):
            sma[i] = np.mean(prices[i - period + 1:i + 1])
        return sma

    def _calculate_ema(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Calculate Exponential Moving Average."""
        alpha = 2.0 / (period + 1)
        ema = np.full(len(prices), np.nan)
        ema[period - 1] = np.mean(prices[:period])
        for i in range(period, len(prices)):
            ema[i] = alpha * prices[i] + (1 - alpha) * ema[i - 1]
        return ema

    def _calculate_kama(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Calculate Kaufman's Adaptive Moving Average (simplified)."""
        # Simplified KAMA implementation
        return self._calculate_ema(prices, period)

class MACDDeltaGenerator(VectorizedFeatureGenerator):
    """Generator for MACD delta and signal crossover flags."""

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        config = FeatureConfig(
            name=f"macd_delta_{fast}_{slow}_{signal}",
            category=FeatureCategory.MOMENTUM,
            description=f"MACD delta and signal crossover flags ({fast}, {slow}, {signal})",
            required_columns=["close"],
            default_lookback=slow + signal,
            min_lookback=slow + signal,
            max_lookback=slow + signal,
            parameters={'fast': fast, 'slow': slow, 'signal': signal},
            matrix_optimized=True,
            gpu_accelerated=False
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.fast = fast
        self.slow = slow
        self.signal = signal

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        close = data['close']
        if len(close) < self.slow + self.signal:
            return pd.Series(np.full(len(close), np.nan), index=data.index)

        # Calculate MACD using pandas ewm for better reliability
        try:
            # Calculate EMA fast and slow using pandas ewm
            ema_fast = close.ewm(span=self.fast).mean()
            ema_slow = close.ewm(span=self.slow).mean()

            # Check for NaN values in EMAs
            if ema_fast.isna().all() or ema_slow.isna().all():
                return pd.Series(np.full(len(close), np.nan), index=data.index)

            macd_line = ema_fast - ema_slow

            # Calculate signal line
            signal_line = macd_line.ewm(span=self.signal).mean()

            # Check for NaN values in signal line
            if signal_line.isna().all():
                return pd.Series(np.full(len(close), np.nan), index=data.index)

            # Calculate MACD delta
            macd_delta = macd_line - signal_line

        except Exception as e:
            logger.warning(f"MACD calculation failed: {e}")
            return pd.Series(np.full(len(close), np.nan), index=data.index)

        # Calculate crossover flags
        crossover_flags = np.zeros(len(close))
        for i in range(1, len(close)):
            if not (pd.isna(macd_line.iloc[i]) or pd.isna(signal_line.iloc[i]) or
                   pd.isna(macd_line.iloc[i-1]) or pd.isna(signal_line.iloc[i-1])):
                # Bullish crossover
                if macd_line.iloc[i-1] <= signal_line.iloc[i-1] and macd_line.iloc[i] > signal_line.iloc[i]:
                    crossover_flags[i] = 1
                # Bearish crossover
                elif macd_line.iloc[i-1] >= signal_line.iloc[i-1] and macd_line.iloc[i] < signal_line.iloc[i]:
                    crossover_flags[i] = -1

        return pd.Series(macd_delta, index=data.index)


class RSIZScoreGenerator(VectorizedFeatureGenerator):
    """Generator for RSI z-score (enhancement to existing RSI)."""

    def __init__(self, rsi_period: int = 14, zscore_window: int = 20):
        config = FeatureConfig(
            name=f"rsi_zscore_{rsi_period}_{zscore_window}",
            category=FeatureCategory.MOMENTUM,
            description=f"RSI z-score over {zscore_window} periods (RSI period {rsi_period})",
            required_columns=["close"],
            default_lookback=rsi_period + zscore_window,
            min_lookback=rsi_period + zscore_window,
            max_lookback=rsi_period + zscore_window,
            parameters={'rsi_period': rsi_period, 'zscore_window': zscore_window},
            matrix_optimized=True,
            gpu_accelerated=False
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.rsi_period = rsi_period
        self.zscore_window = zscore_window

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        close = data['close'].values
        if len(close) < self.rsi_period + self.zscore_window:
            return pd.Series(np.full(len(close), np.nan), index=data.index)

        # Calculate RSI
        rsi = self._calculate_rsi(close, self.rsi_period)

        # Calculate RSI z-score
        rsi_zscore = np.full(len(close), np.nan)
        for i in range(self.rsi_period + self.zscore_window - 1, len(close)):
            window_rsi = rsi[i - self.zscore_window + 1:i + 1]
            valid_rsi = window_rsi[np.isfinite(window_rsi)]
            if len(valid_rsi) > 1:
                mean_rsi = np.mean(valid_rsi)
                std_rsi = np.std(valid_rsi, ddof=1)
                if std_rsi > 0:
                    rsi_zscore[i] = (rsi[i] - mean_rsi) / std_rsi

        return pd.Series(rsi_zscore, index=data.index)

    def _calculate_rsi(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Calculate RSI."""
        if len(prices) < period + 1:
            return np.full(len(prices), np.nan)

        delta = np.diff(prices, prepend=prices[0])
        gains = np.where(delta > 0, delta, 0)
        losses = np.where(delta < 0, -delta, 0)

        avg_gains = self._rolling_mean(gains, period)
        avg_losses = self._rolling_mean(losses, period)

        rs = np.divide(avg_gains, avg_losses, out=np.ones_like(avg_gains), where=avg_losses!=0)
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def _rolling_mean(self, data: np.ndarray, window: int) -> np.ndarray:
        """Calculate rolling mean."""
        if len(data) < window:
            return np.full(len(data), np.nan)

        rolling_mean = np.full(len(data), np.nan)
        for i in range(window - 1, len(data)):
            rolling_mean[i] = np.mean(data[i - window + 1:i + 1])
        return rolling_mean

class StochasticKDGenerator(VectorizedFeatureGenerator):
    """Generator for Stochastic %K and %D oscillators."""

    def __init__(self, k_period: int = 14, d_period: int = 3):
        config = FeatureConfig(
            name=f"stochastic_kd_{k_period}_{d_period}",
            category=FeatureCategory.MOMENTUM,
            description=f"Stochastic %K and %D oscillators ({k_period}, {d_period})",
            required_columns=["close", "high", "low"],
            default_lookback=k_period + d_period,
            min_lookback=k_period + d_period,
            max_lookback=k_period + d_period,
            parameters={'k_period': k_period, 'd_period': d_period},
            matrix_optimized=True,
            gpu_accelerated=False
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.k_period = k_period
        self.d_period = d_period

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        close = data['close'].values
        high = data['high'].values
        low = data['low'].values

        if len(close) < self.k_period + self.d_period:
            return pd.Series(np.full(len(close), np.nan), index=data.index)

        # Calculate %K
        k_percent = np.full(len(close), np.nan)
        for i in range(self.k_period - 1, len(close)):
            period_high = np.max(high[i - self.k_period + 1:i + 1])
            period_low = np.min(low[i - self.k_period + 1:i + 1])
            if period_high != period_low:
                k_percent[i] = ((close[i] - period_low) / (period_high - period_low)) * 100

        # Calculate %D (smoothed %K)
        d_percent = np.full(len(close), np.nan)
        for i in range(self.k_period + self.d_period - 2, len(close)):
            k_window = k_percent[i - self.d_period + 1:i + 1]
            valid_k = k_window[np.isfinite(k_window)]
            if len(valid_k) > 0:
                d_percent[i] = np.mean(valid_k)

        return pd.Series(k_percent, index=data.index)

class DonchianChannelGenerator(VectorizedFeatureGenerator):
    """Generator for Donchian channel %b (position within rolling min-max)."""

    def __init__(self, period: int = 20):
        config = FeatureConfig(
            name=f"donchian_channel_{period}",
            category=FeatureCategory.MOMENTUM,
            description=f"Donchian channel %b over {period} periods",
            required_columns=["close", "high", "low"],
            default_lookback=period,
            min_lookback=period,
            max_lookback=period,
            parameters={'period': period},
            matrix_optimized=True,
            gpu_accelerated=False
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.period = period

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        close = data['close'].values
        high = data['high'].values
        low = data['low'].values

        if len(close) < self.period:
            return pd.Series(np.full(len(close), np.nan), index=data.index)

        # Calculate Donchian channel %b
        donchian_b = np.full(len(close), np.nan)
        for i in range(self.period - 1, len(close)):
            period_high = np.max(high[i - self.period + 1:i + 1])
            period_low = np.min(low[i - self.period + 1:i + 1])
            if period_high != period_low:
                donchian_b[i] = (close[i] - period_low) / (period_high - period_low)

        return pd.Series(donchian_b, index=data.index)

def create_momentum_generators(periods: Dict[str, List[int]] = None) -> List[FeatureGenerator]:
    """Create a set of momentum feature generators."""
    if periods is None:
        periods = {
            'rsi': [14],
            'macd_fast': [12],
            'macd_slow': [26],
            'stochastic': [14],
            'williams_r': [14],
            'momentum': [10],
            'roc': [10]
        }

    generators = []

    # RSI generators
    for period in periods.get('rsi', [14]):
        generators.append(RSIGenerator(period))

    # MACD generators
    fast_periods = periods.get('macd_fast', [12])
    slow_periods = periods.get('macd_slow', [26])
    for fast in fast_periods:
        for slow in slow_periods:
            generators.append(MACDGenerator(fast, slow))

    # Stochastic generators
    for period in periods.get('stochastic', [14]):
        generators.append(StochasticGenerator(period))

    # Williams %R generators
    for period in periods.get('williams_r', [14]):
        generators.append(WilliamsRGenerator(period))

    # Momentum generators
    for period in periods.get('momentum', [10]):
        generators.append(MomentumOscillatorGenerator(period))

    # ROC generators
    for period in periods.get('roc', [10]):
        generators.append(RateOfChangeGenerator(period))

    # NEW FEATURES - Advanced Momentum Analysis
    # Momentum endpoints generators
    for ma_type in periods.get('momentum_endpoints_types', ['SMA', 'EMA']):
        for ma_period in periods.get('momentum_endpoints_periods', [20]):
            generators.append(MomentumEndpointsGenerator(ma_period, ma_type))

    # MACD delta generators
    for fast in periods.get('macd_delta_fast', [12]):
        for slow in periods.get('macd_delta_slow', [26]):
            for signal in periods.get('macd_delta_signal', [9]):
                generators.append(MACDDeltaGenerator(fast, slow, signal))

    # RSI z-score generators
    for rsi_period in periods.get('rsi_zscore_periods', [14]):
        for zscore_window in periods.get('rsi_zscore_windows', [20]):
            generators.append(RSIZScoreGenerator(rsi_period, zscore_window))

    # Stochastic KD generators
    for k_period in periods.get('stochastic_k_periods', [14]):
        for d_period in periods.get('stochastic_d_periods', [3]):
            generators.append(StochasticKDGenerator(k_period, d_period))

    # Donchian channel generators
    for period in periods.get('donchian_periods', [20]):
        generators.append(DonchianChannelGenerator(period))

    return generators

class AdvancedMomentumGenerator(VectorizedFeatureGenerator):
    """Generator for advanced momentum indicators for regime detection."""

    def __init__(self, fast_period: int = 5, slow_period: int = 20):
        config = FeatureConfig(
            name=f"advanced_momentum_{fast_period}_{slow_period}",
            category=FeatureCategory.MOMENTUM,
            description=f"Advanced momentum indicator ({fast_period}/{slow_period}) for regime detection",
            required_columns=["close"],
            default_lookback=slow_period,
            min_lookback=slow_period,
            max_lookback=slow_period * 2,
            parameters={"fast_period": fast_period, "slow_period": slow_period}
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Calculate enhanced momentum indicator with regime persistence."""
        close = data['close']

        # Fast and slow momentum with regime smoothing
        fast_ma = close.rolling(window=self.config.parameters["fast_period"]).mean()
        slow_ma = close.rolling(window=self.config.parameters["slow_period"]).mean()

        # Enhanced momentum ratio with regime strength
        momentum_ratio = (fast_ma - slow_ma) / (slow_ma + 1e-8)  # Avoid division by zero

        # Regime persistence measure
        momentum_volatility = self._vectorbt_rolling_operation(momentum_ratio, "std", 10)
        momentum_trend = self._vectorbt_rolling_operation(momentum_ratio, "mean", 5)

        # Regime strength: higher when momentum is consistent and trending
        regime_strength = np.abs(momentum_trend) / (momentum_volatility + 1e-8)

        # Enhanced regime indicator combining momentum and persistence
        enhanced_momentum = momentum_ratio * (1 + regime_strength)

        # Add regime transition detection
        momentum_change = momentum_ratio.diff().abs()
        regime_transition = self._vectorbt_rolling_operation(momentum_change, "mean", 3)

        # Combine momentum with regime transition awareness
        regime_aware_momentum = enhanced_momentum * (1 - regime_transition)

        return regime_aware_momentum

class PriceAccelerationGenerator(VectorizedFeatureGenerator):
    """Generator for price acceleration indicators."""

    def __init__(self, period: int = 10):
        config = FeatureConfig(
            name=f"price_acceleration_{period}",
            category=FeatureCategory.MOMENTUM,
            description=f"Price acceleration indicator over {period} periods",
            required_columns=["close"],
            default_lookback=period,
            min_lookback=period,
            max_lookback=period * 2,
            parameters={"period": period}
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Calculate price acceleration."""
        close = data['close']

        # Velocity (rate of change)
        velocity = close.pct_change(self.config.parameters["period"])

        # Acceleration (change in velocity)
        acceleration = velocity.diff(self.config.parameters["period"])

        return acceleration

class VolumeMomentumGenerator(VectorizedFeatureGenerator):
    """Generator for volume-based momentum indicators."""

    def __init__(self, period: int = 10):
        config = FeatureConfig(
            name=f"volume_momentum_{period}",
            category=FeatureCategory.VOLUME,
            description=f"Volume momentum indicator over {period} periods",
            required_columns=["volume"],
            default_lookback=period,
            min_lookback=period,
            max_lookback=period * 2,
            parameters={"period": period}
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Calculate volume momentum."""
        volume = data['volume']

        # Volume momentum
        volume_ma = volume.rolling(window=self.config.parameters["period"]).mean()
        # Handle division by zero for volume momentum
        with np.errstate(divide='ignore', invalid='ignore'):
            volume_momentum = (volume - volume_ma) / volume_ma
            volume_momentum = np.where(volume_ma == 0, np.nan, volume_momentum)

        return volume_momentum

# UnifiedMomentumFeatureGenerator removed - functionality consolidated into MomentumFeatureGenerator

class VectorBTMomentumFeatureGenerator(VectorBTFeatureGenerator):
    """VectorBT-optimized momentum feature generator with comprehensive indicators."""

    def __init__(self, period: int = 14, config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config(period)
        super().__init__(config, enable_gpu=True, enable_parallel=True)
        self.period = period

        # Initialize Unified Vectorization Manager for advanced optimizations
        if UNIFIED_VECTORIZATION_AVAILABLE:
            self.unified_manager = get_unified_vectorization_manager()
        else:
            self.unified_manager = None

    @classmethod
    def _create_default_config(cls, period: int = 14) -> FeatureConfig:
        return FeatureConfig(
            name=f"vectorbt_momentum_comprehensive_{period}",
            category=FeatureCategory.MOMENTUM,
            description=f"VectorBT-optimized comprehensive momentum features over {period} periods",
            required_columns=["close"],
            optional_columns=["high", "low", "open", "volume"],
            default_lookback=period,
            min_lookback=period,
            max_lookback=period,
            parameters={"period": period},
            matrix_optimized=True,
            gpu_accelerated=True
        )

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate comprehensive momentum features using VectorBT."""
        if data.empty:
            return pd.Series(dtype=float, index=data.index, name=f'vectorbt_momentum_{self.period}')

        # Generate multiple momentum indicators using VectorBT
        operations = [
            {'type': 'indicator', 'name': 'rsi', 'params': {'indicator': 'rsi', 'window': self.period}},
            {'type': 'indicator', 'name': 'macd', 'params': {'indicator': 'macd', 'fast': 12, 'slow': 26}},
            {'type': 'indicator', 'name': 'macd_signal', 'params': {'indicator': 'macd_signal', 'fast': 12, 'slow': 26}},
            {'type': 'indicator', 'name': 'macd_histogram', 'params': {'indicator': 'macd_histogram', 'fast': 12, 'slow': 26}},
            {'type': 'indicator', 'name': 'stoch_k', 'params': {'indicator': 'stoch_k', 'window': self.period}},
            {'type': 'indicator', 'name': 'stoch_d', 'params': {'indicator': 'stoch_d', 'window': self.period}},
            {'type': 'indicator', 'name': 'willr', 'params': {'indicator': 'willr', 'window': self.period}},
            {'type': 'indicator', 'name': 'cci', 'params': {'indicator': 'cci', 'window': self.period}},
            {'type': 'indicator', 'name': 'mfi', 'params': {'indicator': 'mfi', 'window': self.period}},
            {'type': 'indicator', 'name': 'roc', 'params': {'indicator': 'roc', 'window': self.period}},
            {'type': 'indicator', 'name': 'mom', 'params': {'indicator': 'mom', 'window': self.period}}
        ]

        # Use batch operations for efficiency
        results = self._vectorbt_batch_operations(data, operations)

        # Combine results into a single momentum measure
        if not results.empty:
            # Weighted combination of different momentum measures
            momentum = (
                0.25 * results.get('rsi', 0) +
                0.15 * results.get('macd', 0) +
                0.15 * results.get('macd_signal', 0) +
                0.10 * results.get('macd_histogram', 0) +
                0.10 * results.get('stoch_k', 0) +
                0.10 * results.get('stoch_d', 0) +
                0.05 * results.get('willr', 0) +
                0.05 * results.get('cci', 0) +
                0.05 * results.get('mfi', 0)
            )
        else:
            # Fallback to simple RSI
            momentum = self._vectorbt_technical_indicator(data, 'rsi', window=self.period)

        return momentum.rename(f'vectorbt_momentum_{self.period}')

# VectorBTRSIGenerator removed - functionality consolidated into RSIGenerator

# VectorBTMACDGenerator removed - functionality consolidated into MACDGenerator
# VectorBTStochasticGenerator removed - functionality consolidated into StochasticGenerator

def create_default_momentum_generators() -> List[FeatureGenerator]:
    """Create default momentum generators including legacy and entropy features."""
    generators = []

    # Add unified momentum generator (highest priority)
    if UNIFIED_VECTORIZATION_AVAILABLE and ROLLING_OPTIMIZER_AVAILABLE:
        generators.append(MomentumFeatureGenerator())

    if VECTORBT_AVAILABLE:
        # VectorBT-optimized generators
        for period in [9, 14, 21, 30]:
            generators.append(VectorBTMomentumFeatureGenerator(period))

    # Standard momentum generators
    for period in [14, 21, 30]:
        generators.append(RSIGenerator(period))
        generators.append(MACDGenerator(12, 26, 9))
        generators.append(StochasticGenerator(period))
        generators.append(WilliamsRGenerator(period))
        generators.append(MomentumOscillatorGenerator(period))
        generators.append(RateOfChangeGenerator(period))

    # Advanced momentum generators
    generators.append(MomentumEndpointsGenerator())
    generators.append(MACDDeltaGenerator())
    generators.append(RSIZScoreGenerator())
    generators.append(StochasticKDGenerator())
    generators.append(DonchianChannelGenerator(20))

    # Advanced momentum generators for regime detection
    generators.append(AdvancedMomentumGenerator(5, 20))
    generators.append(AdvancedMomentumGenerator(10, 30))

    # Entropy-based momentum generators
    generators.append(RSIEntropyGenerator())
    generators.append(MACDEntropyGenerator())

    # Analyst momentum generators
    generators.append(AnalystMomentum5mGenerator())
    generators.append(AnalystMomentum15mGenerator())
    generators.append(AnalystMomentum1hGenerator())
    generators.append(AnalystMomentumAlignmentGenerator())

    return generators
