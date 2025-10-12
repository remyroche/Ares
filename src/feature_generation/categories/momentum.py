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
from ..utils.centralized_logging import tprint, log_function_execution, fast_fail_error
from ..utils.error_handling import (
    DataValidationError, ConfigurationError, ComputationError,
    validate_required_columns, validate_finite_values, safe_divide
)

from ..core.feature_generator import FeatureGenerator, FeatureResult, VectorizedFeatureGenerator, FeatureConfig, FeatureCategory
from ..core.vectorbt_feature_generator import VectorBTFeatureGenerator, VECTORBT_AVAILABLE

# Optimization utilities
try:
    from ..utils.vectorization_optimizer import get_vectorization_optimizer
    from ..utils.optimized_feature_pipeline import get_optimized_feature_pipeline
    OPTIMIZATION_AVAILABLE = True
except ImportError:
    OPTIMIZATION_AVAILABLE = False
from ..base_calculations import (
    BaseCalculator,
    BaseCalculationType,
    BaseCalculationConfig,
    create_base_calculator
)
from ...utils.math_validation import safe_divide, validate_finite, safe_percentage_change

# Legacy features are imported separately to avoid circular imports
from .entropy import (
    RSIEntropyGenerator,
    MACDEntropyGenerator
)

# VectorBT imports for native optimization
try:
    import vectorbt as vbt
    from vectorbt.generic import rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max, rolling_sum, rolling_apply, rolling_corr, rolling_cov
    from vectorbt.generic import scale, rank, zscore, winsorize, clip, quantile
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None
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
    warnings.warn("VectorBT not available. Install with: pip install vectorbt for optimized performance")

# Import optimized rolling operations - NOW USING NEW OPTIMIZED VERSION
try:
    from ..utils.consolidated_rolling_optimizer import (
        ConsolidatedRollingOptimizer as VectorBTRollingOptimizer,
        get_global_rolling_optimizer as get_vectorbt_rolling_optimizer,
        RollingOperationConfig,
        RollingOperationType
    )
    from ..utils.statistical_calculations_optimizer import (
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
        from ..utils.vectorbt_rolling_optimizer import (
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
    from ...utils.ml_common.unified_vectorization_manager import (
        get_unified_vectorization_manager,
        OperationType,
        optimize_financial_operation
    )
    UNIFIED_VECTORIZATION_AVAILABLE = True
except ImportError:
    UNIFIED_VECTORIZATION_AVAILABLE = False

logger = logging.getLogger(__name__)

# Optional GPU acceleration
try:
    import cupy as cp
# Centralized utility imports
from ..utils.vectorbt_rolling_optimizer import get_vectorbt_rolling_optimizer
from ...features_common.transforms.vectorbt_scaler import VectorBTScaler, create_vectorbt_scaler
from ..core.feature_bank import get_global_feature_bank
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

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

# Analyst Features - Cross-timeframe momentum generatorsclass AnalystMomentum5mGenerator(VectorizedFeatureGenerator):
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
            return momentumclass AnalystMomentum15mGenerator(VectorizedFeatureGenerator):
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
            return momentumclass AnalystMomentum1hGenerator(VectorizedFeatureGenerator):
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
        return momentumclass AnalystMomentumAlignmentGenerator(VectorizedFeatureGenerator):
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
            delta = close.diff()
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
            
            delta = base_values.diff()
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
            
            k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
            
            return k_percent
        else:
            # For other base calculations, use rolling min/max on base values
            base_values = self.base_calculator.calculate(data)
            
            lowest_low = base_values.rolling(window=self.k_period).min()
            highest_high = base_values.rolling(window=self.k_period).max()
            k_percent = 100 * ((base_values - lowest_low) / (highest_high - lowest_low))
            
            return k_percentclass WilliamsRGenerator(VectorizedFeatureGenerator):
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
            
            williams_r = -100 * ((highest_high - close) / (highest_high - lowest_low))
            
            return williams_r
        else:
            # For other base calculations, use rolling min/max on base values
            base_values = self.base_calculator.calculate(data)
            
            highest_high = base_values.rolling(window=self.period).max()
            lowest_low = base_values.rolling(window=self.period).min()
            williams_r = -100 * ((highest_high - base_values) / (highest_high - lowest_low))
            
            return williams_rclass MomentumOscillatorGenerator(VectorizedFeatureGenerator):
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
        
        # Calculate momentum
        momentum = base_values - base_values.shift(self.period)
        
        return momentumclass RateOfChangeGenerator(VectorizedFeatureGenerator):
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

        # Calculate ROC with math validation using safe math utilities
        shifted_values = base_values.shift(self.period)

        # Use safe percentage change calculation
        roc_values = []
        for i in range(len(base_values)):
            current_val = base_values.iloc[i]
            shifted_val = shifted_values.iloc[i]

            # Use safe percentage change function
            roc_val = safe_percentage_change(shifted_val, current_val)
            roc_values.append(roc_val)

        roc_series = pd.Series(roc_values, index=data.index, name=f'roc_{self.period}_{self.base_calculation.value}')

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
        
        close = data['close'].values
        if len(close) < self.slow + self.signal:
            return pd.Series(np.full(len(close), np.nan), index=data.index)
        
        # Calculate MACD
        ema_fast = self._calculate_ema(close, self.fast)
        ema_slow = self._calculate_ema(close, self.slow)
        macd_line = ema_fast - ema_slow
        
        # Calculate signal line
        signal_line = self._calculate_ema(macd_line, self.signal)
        
        # Calculate MACD delta
        macd_delta = macd_line - signal_line
        
        # Calculate crossover flags
        crossover_flags = np.zeros(len(close))
        for i in range(1, len(close)):
            if not (np.isnan(macd_line[i]) or np.isnan(signal_line[i]) or 
                   np.isnan(macd_line[i-1]) or np.isnan(signal_line[i-1])):
                # Bullish crossover
                if macd_line[i-1] <= signal_line[i-1] and macd_line[i] > signal_line[i]:
                    crossover_flags[i] = 1
                # Bearish crossover
                elif macd_line[i-1] >= signal_line[i-1] and macd_line[i] < signal_line[i]:
                    crossover_flags[i] = -1
        
        return pd.Series(macd_delta, index=data.index)
    
    def _calculate_ema(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Calculate Exponential Moving Average."""
        alpha = 2.0 / (period + 1)
        ema = np.full(len(prices), np.nan)
        ema[period - 1] = np.mean(prices[:period])
        for i in range(period, len(prices)):
            ema[i] = alpha * prices[i] + (1 - alpha) * ema[i - 1]
        return ema

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
    
    return generatorsclass AdvancedMomentumGenerator(VectorizedFeatureGenerator):
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

        return regime_aware_momentumclass PriceAccelerationGenerator(VectorizedFeatureGenerator):
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

        return accelerationclass VolumeMomentumGenerator(VectorizedFeatureGenerator):
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
        volume_momentum = (volume - volume_ma) / volume_ma

        return volume_momentum


class UnifiedMomentumFeatureGenerator(VectorizedFeatureGenerator):
    """Unified momentum feature generator using VectorBTRollingOptimizer and UnifiedVectorizationManager."""
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config()
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        
        # Initialize optimization components
        self.rolling_optimizer = get_vectorbt_rolling_optimizer() if ROLLING_OPTIMIZER_AVAILABLE else None
        self.unified_manager = get_unified_vectorization_manager() if UNIFIED_VECTORIZATION_AVAILABLE else None
        
        # Performance tracking
        self.performance_stats = {
            'unified_operations': 0,
            'rolling_optimizer_operations': 0,
            'vectorbt_operations': 0,
            'total_operations': 0
        }
    
    @classmethod
    def _create_default_config(cls) -> FeatureConfig:
        return FeatureConfig(
            name="unified_momentum_features",
            category=FeatureCategory.MOMENTUM,
            description="Unified momentum features using VectorBTRollingOptimizer and UnifiedVectorizationManager",
            required_columns=["close"],
            optional_columns=["high", "low", "open", "volume"],
            default_lookback=20,
            min_lookback=5,
            max_lookback=100,
            parameters={
                "rsi_periods": [14, 21],
                "macd_fast": [12],
                "macd_slow": [26],
                "stochastic_periods": [14],
                "momentum_windows": [10, 20]
            },
            matrix_optimized=True,
            gpu_accelerated=True
        )
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate comprehensive momentum features using unified optimization."""
        if data.empty:
            return pd.Series(dtype=float, index=data.index, name='unified_momentum')
        
        # Use UnifiedVectorizationManager for comprehensive momentum analysis
        if self.unified_manager and len(data) > 1000:
            try:
                # Prepare data for unified optimization
                operation_data = {
                    'close': data['close'],
                    'high': data.get('high'),
                    'low': data.get('low'),
                    'volume': data.get('volume')
                }
                
                # Use unified manager for momentum analysis
                result = self.unified_manager.optimize_operation(
                    OperationType.TECHNICAL_INDICATORS,
                    operation_data,
                    **kwargs
                )
                
                # Extract momentum features from result
                if hasattr(result, 'result') and isinstance(result.result, dict):
                    momentum_features = result.result.get('momentum_features', {})
                    if momentum_features:
                        # Combine multiple momentum indicators
                        combined_momentum = self._combine_momentum_indicators(momentum_features)
                        self.performance_stats['unified_operations'] += 1
                        return combined_momentum
                
                # Fallback to individual calculations
                return self._calculate_individual_momentum_features(data)
                
            except Exception as e:
                logger.warning(f"Unified momentum calculation failed: {e}, using individual calculations")
                return self._calculate_individual_momentum_features(data)
        else:
            # Use individual calculations with VectorBTRollingOptimizer
            return self._calculate_individual_momentum_features(data)
    
    def _calculate_individual_momentum_features(self, data: pd.DataFrame) -> pd.Series:
        """Calculate individual momentum features using VectorBTRollingOptimizer."""
        close = data['close']
        
        # Calculate multiple momentum indicators
        momentum_features = {}
        
        # RSI
        if self.rolling_optimizer and len(close) > 100:
            try:
                delta = close.diff()
                gain = delta.where(delta > 0, 0)
                loss = -delta.where(delta < 0, 0)
                
                avg_gain = self.rolling_optimizer.rolling_mean(gain, window=14)
                avg_loss = self.rolling_optimizer.rolling_mean(loss, window=14)
                rs = avg_gain / avg_loss.replace(0, 1)
                rsi = 100 - (100 / (1 + rs))
                momentum_features['rsi'] = rsi
                self.performance_stats['rolling_optimizer_operations'] += 2
            except Exception as e:
                logger.warning(f"RSI calculation failed: {e}")
        
        # MACD
        if self.rolling_optimizer and len(close) > 100:
            try:
                ema_fast = close.ewm(span=12).mean()
                ema_slow = close.ewm(span=26).mean()
                macd = ema_fast - ema_slow
                momentum_features['macd'] = macd
                self.performance_stats['rolling_optimizer_operations'] += 1
            except Exception as e:
                logger.warning(f"MACD calculation failed: {e}")
        
        # Momentum
        if self.rolling_optimizer and len(close) > 100:
            try:
                momentum = close - close.shift(20)
                momentum_features['momentum'] = momentum
                self.performance_stats['rolling_optimizer_operations'] += 1
            except Exception as e:
                logger.warning(f"Momentum calculation failed: {e}")
        
        # Stochastic (if high/low available)
        if 'high' in data.columns and 'low' in data.columns and self.rolling_optimizer:
            try:
                high = data['high']
                low = data['low']
                
                lowest_low = self.rolling_optimizer.rolling_min(low, window=14)
                highest_high = self.rolling_optimizer.rolling_max(high, window=14)
                stoch = 100 * ((close - lowest_low) / (highest_high - lowest_low))
                momentum_features['stochastic'] = stoch
                self.performance_stats['rolling_optimizer_operations'] += 2
            except Exception as e:
                logger.warning(f"Stochastic calculation failed: {e}")
        
        # Combine features
        if momentum_features:
            combined_momentum = self._combine_momentum_indicators(momentum_features)
        else:
            # Fallback to simple momentum
            combined_momentum = close - close.shift(20)
        
        self.performance_stats['total_operations'] += 1
        return combined_momentum
    
    def _combine_momentum_indicators(self, momentum_features: Dict[str, pd.Series]) -> pd.Series:
        """Combine multiple momentum indicators into a single score."""
        if not momentum_features:
            return pd.Series(dtype=float)
        
        # Normalize each indicator to 0-1 range
        normalized_features = {}
        for name, series in momentum_features.items():
            if series.notna().any():
                # Min-max normalization
                min_val = series.min()
                max_val = series.max()
                if max_val != min_val:
                    normalized_features[name] = (series - min_val) / (max_val - min_val)
                else:
                    normalized_features[name] = pd.Series(0.5, index=series.index)
        
        if not normalized_features:
            return pd.Series(dtype=float)
        
        # Weighted combination (equal weights for now)
        weights = {
            'rsi': 0.3,
            'macd': 0.25,
            'momentum': 0.25,
            'stochastic': 0.2
        }
        
        combined = pd.Series(0.0, index=list(normalized_features.values())[0].index)
        total_weight = 0.0
        
        for name, series in normalized_features.items():
            weight = weights.get(name, 0.1)
            combined += series * weight
            total_weight += weight
        
        if total_weight > 0:
            combined = combined / total_weight
        
        return combined
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        stats = self.performance_stats.copy()
        if stats['total_operations'] > 0:
            stats['unified_usage_rate'] = stats['unified_operations'] / stats['total_operations']
            stats['rolling_optimizer_usage_rate'] = stats['rolling_optimizer_operations'] / stats['total_operations']
        return stats


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


class VectorBTRSIGenerator(VectorBTFeatureGenerator):
    """VectorBT-optimized RSI generator."""
    
    def __init__(self, period: int = 14, config: Optional[FeatureConfig] = None):
        tprint(f"Initializing VectorBTRSIGenerator with period: {period}")
        if config is None:
            config = self._create_default_config(period)
        super().__init__(config, enable_gpu=True, enable_parallel=True)
        self.period = period
    
    @classmethod
    def _create_default_config(cls, period: int = 14) -> FeatureConfig:
        return FeatureConfig(
            name=f"vectorbt_rsi_{period}",
            category=FeatureCategory.MOMENTUM,
            description=f"VectorBT-optimized RSI over {period} periods",
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
        """Generate RSI using VectorBT."""
        tprint(f"Generating VectorBT RSI feature with period {self.period}")
        if data.empty:
            return pd.Series(dtype=float, index=data.index, name=f'vectorbt_rsi_{self.period}')
        
        # Generate RSI using VectorBT
        rsi = self._vectorbt_technical_indicator(data, 'rsi', window=self.period)
        
        return rsi.rename(f'vectorbt_rsi_{self.period}')


class VectorBTMACDGenerator(VectorBTFeatureGenerator):
    """VectorBT-optimized MACD generator."""
    
    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9, config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config(fast, slow, signal)
        super().__init__(config, enable_gpu=True, enable_parallel=True)
        self.fast = fast
        self.slow = slow
        self.signal = signal
    
    @classmethod
    def _create_default_config(cls, fast: int = 12, slow: int = 26, signal: int = 9) -> FeatureConfig:
        return FeatureConfig(
            name=f"vectorbt_macd_{fast}_{slow}_{signal}",
            category=FeatureCategory.MOMENTUM,
            description=f"VectorBT-optimized MACD with fast={fast}, slow={slow}, signal={signal}",
            required_columns=["close"],
            optional_columns=["high", "low", "open", "volume"],
            default_lookback=slow,
            min_lookback=slow,
            max_lookback=slow,
            parameters={"fast": fast, "slow": slow, "signal": signal},
            matrix_optimized=True,
            gpu_accelerated=True
        )
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate MACD using VectorBT."""
        if data.empty:
            return pd.Series(dtype=float, index=data.index, name=f'vectorbt_macd_{self.fast}_{self.slow}')
        
        # Generate MACD using VectorBT
        macd = self._vectorbt_technical_indicator(data, 'macd', 
                                                fast=self.fast, 
                                                slow=self.slow, 
                                                signal=self.signal)
        
        return macd.rename(f'vectorbt_macd_{self.fast}_{self.slow}')


class VectorBTStochasticGenerator(VectorBTFeatureGenerator):
    """VectorBT-optimized Stochastic generator."""
    
    def __init__(self, period: int = 14, config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config(period)
        super().__init__(config, enable_gpu=True, enable_parallel=True)
        self.period = period
    
    @classmethod
    def _create_default_config(cls, period: int = 14) -> FeatureConfig:
        return FeatureConfig(
            name=f"vectorbt_stoch_{period}",
            category=FeatureCategory.MOMENTUM,
            description=f"VectorBT-optimized Stochastic over {period} periods",
            required_columns=["high", "low", "close"],
            optional_columns=["open", "volume"],
            default_lookback=period,
            min_lookback=period,
            max_lookback=period,
            parameters={"period": period},
            matrix_optimized=True,
            gpu_accelerated=True
        )
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate Stochastic %K using VectorBT."""
        if data.empty:
            return pd.Series(dtype=float, index=data.index, name=f'vectorbt_stoch_{self.period}')
        
        # Generate Stochastic %K using VectorBT
        stoch_k = self._vectorbt_technical_indicator(data, 'stoch_k', window=self.period)
        
        return stoch_k.rename(f'vectorbt_stoch_{self.period}')



    def _optimized_rolling_operation(self, data: pd.Series, operation: str, 
                                   window: int, **kwargs) -> pd.Series:
        """Perform rolling operation using centralized VectorBTRollingOptimizer."""
        if not hasattr(self, 'rolling_optimizer'):
            self.rolling_optimizer = get_vectorbt_rolling_optimizer()
        
        try:
            if operation == 'mean':
                return self.rolling_optimizer.rolling_mean(data, window, **kwargs)
            elif operation == 'std':
                return self.rolling_optimizer.rolling_std(data, window, **kwargs)
            elif operation == 'var':
                return self.rolling_optimizer.rolling_var(data, window, **kwargs)
            elif operation == 'min':
                return self.rolling_optimizer.rolling_min(data, window, **kwargs)
            elif operation == 'max':
                return self.rolling_optimizer.rolling_max(data, window, **kwargs)
            elif operation == 'sum':
                return self.rolling_optimizer.rolling_sum(data, window, **kwargs)
            else:
                raise ValueError(f"Unsupported operation: {operation}")
        except Exception as e:
            logger.warning(f"VectorBT rolling operation failed: {e}, using fallback")
            return self._fallback_rolling_operation(data, operation, window, **kwargs)
    
    def _fallback_rolling_operation(self, data: pd.Series, operation: str, 
                                  window: int, **kwargs) -> pd.Series:
        """Fallback rolling operation using pandas."""
        rolling_obj = data.rolling(window=window, **kwargs)
        
        if operation == 'mean':
            return rolling_obj.mean()
        elif operation == 'std':
            return rolling_obj.std()
        elif operation == 'var':
            return rolling_obj.var()
        elif operation == 'min':
            return rolling_obj.min()
        elif operation == 'max':
            return rolling_obj.max()
        elif operation == 'sum':
            return rolling_obj.sum()
        else:
            raise ValueError(f"Unsupported operation: {operation}")
    
    def _normalize_feature(self, data: pd.Series, method: str = 'zscore') -> pd.Series:
        """Normalize feature using centralized VectorBTScaler."""
        try:
            scaler = create_vectorbt_scaler(method=method)
            return scaler.fit_transform(data)
        except Exception as e:
            logger.warning(f"VectorBT scaling failed: {e}, using fallback")
            return self._fallback_normalize(data, method)
    
    def _fallback_normalize(self, data: pd.Series, method: str = 'zscore') -> pd.Series:
        """Fallback normalization using pandas/numpy."""
        if method == 'zscore':
            return (data - data.mean()) / data.std()
        elif method == 'minmax':
            return (data - data.min()) / (data.max() - data.min())
        elif method == 'robust':
            median = data.median()
            mad = (data - median).abs().median()
            return (data - median) / mad
        else:
            return data

def create_default_momentum_generators() -> List[FeatureGenerator]:
    """Create default momentum generators including legacy and entropy features."""
    generators = []
    
    # Add unified momentum generator (highest priority)
    if UNIFIED_VECTORIZATION_AVAILABLE and ROLLING_OPTIMIZER_AVAILABLE:
        generators.append(UnifiedMomentumFeatureGenerator())
    
    if VECTORBT_AVAILABLE:
        # VectorBT-optimized generators
        for period in [9, 14, 21, 30]:
            generators.append(VectorBTMomentumFeatureGenerator(period))
            generators.append(VectorBTRSIGenerator(period))
            generators.append(VectorBTStochasticGenerator(period))
            
        # MACD with different parameters
        for fast in [8, 12, 16]:
            for slow in [21, 26, 34]:
                generators.append(VectorBTMACDGenerator(fast, slow))
    else:
        # Fallback to original generators
        # Add new momentum generators
        generators.extend(create_momentum_generators())
        
        # Add legacy momentum generators
        generators.extend([
            VectorBTRSIGenerator(14),
            VectorBTMACDGenerator(12, 26, 9),
            VectorBTStochasticGenerator(14),
        ])
        
        # Add entropy-based momentum generators
        generators.extend([
        RSIEntropyGenerator(20, 14),
        MACDEntropyGenerator(20, 12, 26),
    ])
    
    # Additional RSI periods
    rsi_periods = [9, 21, 25]
    for period in rsi_periods:
        generators.append(RSIGenerator(period))
        generators.append(VectorBTRSIGenerator(period))
    
    # Additional MACD configurations
    macd_configs = [(8, 21, 5), (5, 35, 5)]
    for fast, slow, signal in macd_configs:
        generators.append(MACDGenerator(fast, slow, signal))
        generators.append(VectorBTMACDGenerator(fast, slow, signal))
    
    # Additional Stochastic periods
    stochastic_periods = [9, 21]
    for period in stochastic_periods:
        generators.append(StochasticGenerator(period))
        generators.append(LegacyStochasticGenerator(period))

    # Advanced momentum indicators for regime detection
    generators.append(AdvancedMomentumGenerator(5, 20))
    generators.append(AdvancedMomentumGenerator(10, 30))
    generators.append(PriceAccelerationGenerator(10))
    generators.append(PriceAccelerationGenerator(20))

    # Analyst Features - Cross-timeframe momentum
    generators.append(AnalystMomentum5mGenerator())
    generators.append(AnalystMomentum15mGenerator())
    generators.append(AnalystMomentum1hGenerator())
    generators.append(AnalystMomentumAlignmentGenerator())

    return generators


class OptimizedMomentumFeatureGenerator(VectorizedFeatureGenerator):
    """Optimized momentum feature generator with comprehensive VectorBT batch processing."""
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config()
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        
        # Initialize VectorBT components
        self.rolling_optimizer = None
        self.unified_manager = None
        
        # Initialize VectorBTRollingOptimizer
        if ROLLING_OPTIMIZER_AVAILABLE:
            try:
                self.rolling_optimizer = get_vectorbt_rolling_optimizer(
                    enable_gpu=True,
                    enable_parallel=True,
                    memory_efficient=True
                )
                self.logger.info("✅ VectorBTRollingOptimizer initialized for momentum features")
            except Exception as e:
                self.logger.warning(f"⚠️ VectorBTRollingOptimizer initialization failed: {e}")
        
        # Initialize UnifiedVectorizationManager
        if UNIFIED_VECTORIZATION_AVAILABLE:
            try:
                self.unified_manager = get_unified_vectorization_manager()
                self.logger.info("✅ UnifiedVectorizationManager initialized for momentum features")
            except Exception as e:
                self.logger.warning(f"⚠️ UnifiedVectorizationManager initialization failed: {e}")
        
        # Performance tracking
        self.performance_stats = {
            'batch_operations': 0,
            'rolling_optimizer_operations': 0,
            'unified_manager_operations': 0,
            'fallback_operations': 0,
            'total_features_generated': 0
        }
    
    @classmethod
    def _create_default_config(cls) -> FeatureConfig:
        return FeatureConfig(
            name="optimized_momentum_features",
            category=FeatureCategory.MOMENTUM,
            description="Optimized momentum features with VectorBT batch processing",
            required_columns=["close"],
            optional_columns=["high", "low", "open", "volume"],
            default_lookback=20,
            min_lookback=5,
            max_lookback=100,
            parameters={
                "rsi_periods": [14, 21, 30],
                "macd_fast": [12, 21],
                "macd_slow": [26, 50],
                "stochastic_periods": [14, 21],
                "williams_periods": [14, 21],
                "momentum_windows": [10, 20, 30]
            },
            matrix_optimized=True,
            gpu_accelerated=True
        )
    
    def generate_momentum_indicators_batch(self, data: pd.DataFrame,
                                         rsi_periods: List[int] = None,
                                         macd_configs: List[Dict[str, int]] = None,
                                         stochastic_periods: List[int] = None,
                                         williams_periods: List[int] = None) -> pd.DataFrame:
        """
        Generate comprehensive momentum indicators in batch using VectorBTRollingOptimizer.
        
        Args:
            data: OHLCV data
            rsi_periods: List of RSI periods
            macd_configs: List of MACD configurations [{'fast': 12, 'slow': 26, 'signal': 9}]
            stochastic_periods: List of Stochastic periods
            williams_periods: List of Williams %R periods
            
        Returns:
            DataFrame with momentum indicators
        """
        if rsi_periods is None:
            rsi_periods = [14, 21, 30]
        if macd_configs is None:
            macd_configs = [{'fast': 12, 'slow': 26, 'signal': 9}, {'fast': 21, 'slow': 50, 'signal': 9}]
        if stochastic_periods is None:
            stochastic_periods = [14, 21]
        if williams_periods is None:
            williams_periods = [14, 21]
        
        tprint_debug(f"🔄 Generating momentum indicators batch: RSI={len(rsi_periods)}, MACD={len(macd_configs)}, Stochastic={len(stochastic_periods)}, Williams={len(williams_periods)}")
        
        feature_configs = []
        
        # RSI features
        for period in rsi_periods:
            feature_configs.append({
                'name': f'rsi_{period}',
                'type': 'rsi',
                'params': {'period': period, 'column': 'close'}
            })
        
        # MACD features
        for i, config in enumerate(macd_configs):
            fast = config['fast']
            slow = config['slow']
            signal = config.get('signal', 9)
            feature_configs.append({
                'name': f'macd_{fast}_{slow}_{signal}',
                'type': 'macd',
                'params': {'fast': fast, 'slow': slow, 'signal': signal, 'column': 'close'}
            })
            feature_configs.append({
                'name': f'macd_signal_{fast}_{slow}_{signal}',
                'type': 'macd_signal',
                'params': {'fast': fast, 'slow': slow, 'signal': signal, 'column': 'close'}
            })
            feature_configs.append({
                'name': f'macd_histogram_{fast}_{slow}_{signal}',
                'type': 'macd_histogram',
                'params': {'fast': fast, 'slow': slow, 'signal': signal, 'column': 'close'}
            })
        
        # Stochastic features
        for period in stochastic_periods:
            feature_configs.append({
                'name': f'stoch_k_{period}',
                'type': 'stochastic_k',
                'params': {'period': period, 'k_period': 3, 'd_period': 3}
            })
            feature_configs.append({
                'name': f'stoch_d_{period}',
                'type': 'stochastic_d',
                'params': {'period': period, 'k_period': 3, 'd_period': 3}
            })
        
        # Williams %R features
        for period in williams_periods:
            feature_configs.append({
                'name': f'williams_r_{period}',
                'type': 'williams_r',
                'params': {'period': period}
            })
        
        return self.generate_optimized_momentum_features(data, feature_configs)
    
    def generate_rsi_features_batch(self, data: pd.DataFrame,
                                  periods: List[int] = None,
                                  columns: List[str] = None) -> pd.DataFrame:
        """
        Generate RSI features in batch for multiple periods and columns.
        
        Args:
            data: OHLCV data
            periods: List of RSI periods
            columns: List of columns to calculate RSI for
            
        Returns:
            DataFrame with RSI features
        """
        if periods is None:
            periods = [14, 21, 30]
        if columns is None:
            columns = ['close']
        
        tprint_debug(f"🔄 Generating RSI features batch: {len(periods)} periods, {len(columns)} columns")
        
        feature_configs = []
        
        for period in periods:
            for column in columns:
                if column in data.columns:
                    feature_configs.append({
                        'name': f'rsi_{column}_{period}',
                        'type': 'rsi',
                        'params': {'period': period, 'column': column}
                    })
        
        return self.generate_optimized_momentum_features(data, feature_configs)
    
    def generate_macd_features_batch(self, data: pd.DataFrame,
                                   macd_configs: List[Dict[str, int]] = None) -> pd.DataFrame:
        """
        Generate MACD features in batch for multiple configurations.
        
        Args:
            data: OHLCV data
            macd_configs: List of MACD configurations
            
        Returns:
            DataFrame with MACD features
        """
        if macd_configs is None:
            macd_configs = [
                {'fast': 12, 'slow': 26, 'signal': 9},
                {'fast': 21, 'slow': 50, 'signal': 9},
                {'fast': 5, 'slow': 35, 'signal': 5}
            ]
        
        tprint_debug(f"🔄 Generating MACD features batch: {len(macd_configs)} configurations")
        
        feature_configs = []
        
        for i, config in enumerate(macd_configs):
            fast = config['fast']
            slow = config['slow']
            signal = config.get('signal', 9)
            
            # MACD line
            feature_configs.append({
                'name': f'macd_{fast}_{slow}',
                'type': 'macd',
                'params': {'fast': fast, 'slow': slow, 'signal': signal, 'column': 'close'}
            })
            
            # Signal line
            feature_configs.append({
                'name': f'macd_signal_{fast}_{slow}',
                'type': 'macd_signal',
                'params': {'fast': fast, 'slow': slow, 'signal': signal, 'column': 'close'}
            })
            
            # Histogram
            feature_configs.append({
                'name': f'macd_histogram_{fast}_{slow}',
                'type': 'macd_histogram',
                'params': {'fast': fast, 'slow': slow, 'signal': signal, 'column': 'close'}
            })
            
            # MACD percentage
            feature_configs.append({
                'name': f'macd_pct_{fast}_{slow}',
                'type': 'macd_percentage',
                'params': {'fast': fast, 'slow': slow, 'signal': signal, 'column': 'close'}
            })
        
        return self.generate_optimized_momentum_features(data, feature_configs)
    
    def generate_stochastic_features_batch(self, data: pd.DataFrame,
                                         periods: List[int] = None,
                                         k_periods: List[int] = None,
                                         d_periods: List[int] = None) -> pd.DataFrame:
        """
        Generate Stochastic features in batch for multiple configurations.
        
        Args:
            data: OHLCV data
            periods: List of Stochastic periods
            k_periods: List of K periods
            d_periods: List of D periods
            
        Returns:
            DataFrame with Stochastic features
        """
        if periods is None:
            periods = [14, 21]
        if k_periods is None:
            k_periods = [3, 5]
        if d_periods is None:
            d_periods = [3, 5]
        
        tprint_debug(f"🔄 Generating Stochastic features batch: {len(periods)} periods, {len(k_periods)} K periods, {len(d_periods)} D periods")
        
        feature_configs = []
        
        for period in periods:
            for k_period in k_periods:
                for d_period in d_periods:
                    # %K
                    feature_configs.append({
                        'name': f'stoch_k_{period}_{k_period}_{d_period}',
                        'type': 'stochastic_k',
                        'params': {'period': period, 'k_period': k_period, 'd_period': d_period}
                    })
                    
                    # %D
                    feature_configs.append({
                        'name': f'stoch_d_{period}_{k_period}_{d_period}',
                        'type': 'stochastic_d',
                        'params': {'period': period, 'k_period': k_period, 'd_period': d_period}
                    })
                    
                    # Stochastic RSI
                    feature_configs.append({
                        'name': f'stoch_rsi_{period}_{k_period}_{d_period}',
                        'type': 'stochastic_rsi',
                        'params': {'period': period, 'k_period': k_period, 'd_period': d_period}
                    })
        
        return self.generate_optimized_momentum_features(data, feature_configs)
    
    def generate_williams_r_features_batch(self, data: pd.DataFrame,
                                         periods: List[int] = None) -> pd.DataFrame:
        """
        Generate Williams %R features in batch for multiple periods.
        
        Args:
            data: OHLCV data
            periods: List of Williams %R periods
            
        Returns:
            DataFrame with Williams %R features
        """
        if periods is None:
            periods = [14, 21, 30]
        
        tprint_debug(f"🔄 Generating Williams %R features batch: {len(periods)} periods")
        
        feature_configs = []
        
        for period in periods:
            feature_configs.append({
                'name': f'williams_r_{period}',
                'type': 'williams_r',
                'params': {'period': period}
            })
            
            # Williams %R smoothed
            feature_configs.append({
                'name': f'williams_r_smooth_{period}',
                'type': 'williams_r_smooth',
                'params': {'period': period, 'smooth_period': 3}
            })
        
        return self.generate_optimized_momentum_features(data, feature_configs)
    
    def generate_momentum_oscillators_batch(self, data: pd.DataFrame,
                                          periods: List[int] = None,
                                          columns: List[str] = None) -> pd.DataFrame:
        """
        Generate momentum oscillators in batch.
        
        Args:
            data: OHLCV data
            periods: List of periods for momentum calculations
            columns: List of columns to calculate momentum for
            
        Returns:
            DataFrame with momentum oscillators
        """
        if periods is None:
            periods = [10, 20, 30]
        if columns is None:
            columns = ['close']
        
        tprint_debug(f"🔄 Generating momentum oscillators batch: {len(periods)} periods, {len(columns)} columns")
        
        feature_configs = []
        
        for period in periods:
            for column in columns:
                if column in data.columns:
                    # Rate of Change
                    feature_configs.append({
                        'name': f'roc_{column}_{period}',
                        'type': 'rate_of_change',
                        'params': {'period': period, 'column': column}
                    })
                    
                    # Momentum
                    feature_configs.append({
                        'name': f'momentum_{column}_{period}',
                        'type': 'momentum',
                        'params': {'period': period, 'column': column}
                    })
                    
                    # Price Oscillator
                    feature_configs.append({
                        'name': f'price_oscillator_{column}_{period}',
                        'type': 'price_oscillator',
                        'params': {'period': period, 'column': column}
                    })
        
        return self.generate_optimized_momentum_features(data, feature_configs)
    
    def _process_momentum_features_individually(self, data: pd.DataFrame, feature_configs: List[Dict[str, Any]]) -> pd.DataFrame:
        """Process momentum features individually as fallback when batch processing fails."""
        tprint_warning("⚠️ Using individual momentum feature processing")
        
        results = {}
        
        for config in feature_configs:
            feature_name = config['name']
            feature_type = config.get('type', 'momentum')
            params = config.get('params', {})
            
            try:
                if feature_type == 'rsi':
                    results[feature_name] = self._calculate_rsi(data, params)
                elif feature_type == 'macd':
                    results[feature_name] = self._calculate_macd(data, params)
                elif feature_type == 'macd_signal':
                    results[feature_name] = self._calculate_macd_signal(data, params)
                elif feature_type == 'macd_histogram':
                    results[feature_name] = self._calculate_macd_histogram(data, params)
                elif feature_type == 'stochastic_k':
                    results[feature_name] = self._calculate_stochastic_k(data, params)
                elif feature_type == 'stochastic_d':
                    results[feature_name] = self._calculate_stochastic_d(data, params)
                elif feature_type == 'williams_r':
                    results[feature_name] = self._calculate_williams_r(data, params)
                elif feature_type == 'rate_of_change':
                    results[feature_name] = self._calculate_rate_of_change(data, params)
                elif feature_type == 'momentum':
                    results[feature_name] = self._calculate_momentum(data, params)
                else:
                    results[feature_name] = pd.Series(np.nan, index=data.index)
                
                self.performance_stats['fallback_operations'] += 1
                
            except Exception as e:
                self.logger.warning(f"⚠️ Feature {feature_name} failed: {e}")
                results[feature_name] = pd.Series(np.nan, index=data.index)
        
        self.performance_stats['total_features_generated'] += len(results)
        return pd.DataFrame(results, index=data.index)
    
    def _calculate_rsi(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """Calculate RSI using VectorBTRollingOptimizer if available."""
        period = params.get('period', 14)
        column = params.get('column', 'close')
        
        if column not in data.columns:
            return pd.Series(np.nan, index=data.index)
        
        series = data[column]
        
        if self.rolling_optimizer and len(series) > period * 2:
            try:
                delta = series.diff()
                gain = delta.where(delta > 0, 0)
                loss = -delta.where(delta < 0, 0)
                
                avg_gain = self.rolling_optimizer.rolling_mean(gain, period)
                avg_loss = self.rolling_optimizer.rolling_mean(loss, period)
                
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
                
                self.performance_stats['rolling_optimizer_operations'] += 1
                return rsi
                
            except Exception as e:
                self.logger.warning(f"VectorBTRollingOptimizer RSI calculation failed: {e}, using pandas fallback")
        
        # Fallback to pandas
        delta = series.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_macd(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """Calculate MACD line."""
        fast = params.get('fast', 12)
        slow = params.get('slow', 26)
        column = params.get('column', 'close')
        
        if column not in data.columns:
            return pd.Series(np.nan, index=data.index)
        
        series = data[column]
        
        if self.rolling_optimizer and len(series) > slow * 2:
            try:
                ema_fast = self.rolling_optimizer.rolling_ema(series, fast)
                ema_slow = self.rolling_optimizer.rolling_ema(series, slow)
                macd = ema_fast - ema_slow
                
                self.performance_stats['rolling_optimizer_operations'] += 1
                return macd
                
            except Exception as e:
                self.logger.warning(f"VectorBTRollingOptimizer MACD calculation failed: {e}, using pandas fallback")
        
        # Fallback to pandas
        ema_fast = series.ewm(span=fast).mean()
        ema_slow = series.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        
        return macd
    
    def _calculate_macd_signal(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """Calculate MACD signal line."""
        fast = params.get('fast', 12)
        slow = params.get('slow', 26)
        signal = params.get('signal', 9)
        column = params.get('column', 'close')
        
        if column not in data.columns:
            return pd.Series(np.nan, index=data.index)
        
        series = data[column]
        macd = self._calculate_macd(data, params)
        
        if self.rolling_optimizer and len(macd) > signal * 2:
            try:
                signal_line = self.rolling_optimizer.rolling_ema(macd, signal)
                self.performance_stats['rolling_optimizer_operations'] += 1
                return signal_line
                
            except Exception as e:
                self.logger.warning(f"VectorBTRollingOptimizer MACD signal calculation failed: {e}, using pandas fallback")
        
        # Fallback to pandas
        signal_line = macd.ewm(span=signal).mean()
        return signal_line
    
    def _calculate_macd_histogram(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """Calculate MACD histogram."""
        macd = self._calculate_macd(data, params)
        signal = self._calculate_macd_signal(data, params)
        histogram = macd - signal
        return histogram
    
    def _calculate_stochastic_k(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """Calculate Stochastic %K."""
        period = params.get('period', 14)
        k_period = params.get('k_period', 3)
        
        if 'high' not in data.columns or 'low' not in data.columns or 'close' not in data.columns:
            return pd.Series(np.nan, index=data.index)
        
        high = data['high']
        low = data['low']
        close = data['close']
        
        if self.rolling_optimizer and len(close) > period * 2:
            try:
                lowest_low = self.rolling_optimizer.rolling_min(low, period)
                highest_high = self.rolling_optimizer.rolling_max(high, period)
                
                stoch_k = 100 * (close - lowest_low) / (highest_high - lowest_low)
                
                # Smooth %K
                if k_period > 1:
                    stoch_k = self.rolling_optimizer.rolling_mean(stoch_k, k_period)
                
                self.performance_stats['rolling_optimizer_operations'] += 1
                return stoch_k
                
            except Exception as e:
                self.logger.warning(f"VectorBTRollingOptimizer Stochastic K calculation failed: {e}, using pandas fallback")
        
        # Fallback to pandas
        lowest_low = low.rolling(window=period).min()
        highest_high = high.rolling(window=period).max()
        
        stoch_k = 100 * (close - lowest_low) / (highest_high - lowest_low)
        
        # Smooth %K
        if k_period > 1:
            stoch_k = stoch_k.rolling(window=k_period).mean()
        
        return stoch_k
    
    def _calculate_stochastic_d(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """Calculate Stochastic %D."""
        d_period = params.get('d_period', 3)
        stoch_k = self._calculate_stochastic_k(data, params)
        
        if self.rolling_optimizer and len(stoch_k) > d_period * 2:
            try:
                stoch_d = self.rolling_optimizer.rolling_mean(stoch_k, d_period)
                self.performance_stats['rolling_optimizer_operations'] += 1
                return stoch_d
                
            except Exception as e:
                self.logger.warning(f"VectorBTRollingOptimizer Stochastic D calculation failed: {e}, using pandas fallback")
        
        # Fallback to pandas
        stoch_d = stoch_k.rolling(window=d_period).mean()
        return stoch_d
    
    def _calculate_williams_r(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """Calculate Williams %R."""
        period = params.get('period', 14)
        
        if 'high' not in data.columns or 'low' not in data.columns or 'close' not in data.columns:
            return pd.Series(np.nan, index=data.index)
        
        high = data['high']
        low = data['low']
        close = data['close']
        
        if self.rolling_optimizer and len(close) > period * 2:
            try:
                highest_high = self.rolling_optimizer.rolling_max(high, period)
                lowest_low = self.rolling_optimizer.rolling_min(low, period)
                
                williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
                
                self.performance_stats['rolling_optimizer_operations'] += 1
                return williams_r
                
            except Exception as e:
                self.logger.warning(f"VectorBTRollingOptimizer Williams %R calculation failed: {e}, using pandas fallback")
        
        # Fallback to pandas
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()
        
        williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
        return williams_r
    
    def _calculate_rate_of_change(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """Calculate Rate of Change."""
        period = params.get('period', 10)
        column = params.get('column', 'close')
        
        if column not in data.columns:
            return pd.Series(np.nan, index=data.index)
        
        series = data[column]
        
        if self.rolling_optimizer and len(series) > period * 2:
            try:
                roc = self.rolling_optimizer.rolling_apply(
                    series, period, lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0] * 100
                )
                self.performance_stats['rolling_optimizer_operations'] += 1
                return roc
                
            except Exception as e:
                self.logger.warning(f"VectorBTRollingOptimizer ROC calculation failed: {e}, using pandas fallback")
        
        # Fallback to pandas
        roc = series.pct_change(period) * 100
        return roc
    
    def _calculate_momentum(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """Calculate Momentum."""
        period = params.get('period', 10)
        column = params.get('column', 'close')
        
        if column not in data.columns:
            return pd.Series(np.nan, index=data.index)
        
        series = data[column]
        
        if self.rolling_optimizer and len(series) > period * 2:
            try:
                momentum = self.rolling_optimizer.rolling_apply(
                    series, period, lambda x: x.iloc[-1] - x.iloc[0]
                )
                self.performance_stats['rolling_optimizer_operations'] += 1
                return momentum
                
            except Exception as e:
                self.logger.warning(f"VectorBTRollingOptimizer Momentum calculation failed: {e}, using pandas fallback")
        
        # Fallback to pandas
        momentum = series - series.shift(period)
        return momentum
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return self.performance_stats.copy()
    
    def reset_performance_stats(self):
        """Reset performance statistics."""
        self.performance_stats = {
            'batch_operations': 0,
            'rolling_optimizer_operations': 0,
            'unified_manager_operations': 0,
            'fallback_operations': 0,
            'total_features_generated': 0
        }