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
from .legacy import (
    LegacyRSIGenerator,
    LegacyMACDGenerator,
    LegacyStochasticGenerator,
    LegacyWilliamsRGenerator
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

# Import optimized rolling operations
try:
    from ..utils.vectorbt_rolling_optimizer import (
        get_vectorbt_rolling_optimizer,
        optimized_rolling_mean,
        optimized_rolling_std,
        optimized_rolling_apply
    )
    ROLLING_OPTIMIZER_AVAILABLE = True
except ImportError:
    ROLLING_OPTIMIZER_AVAILABLE = False

logger = logging.getLogger(__name__)

# Optional GPU acceleration
try:
    import cupy as cp
# Centralized utility imports
from ..utils.vectorbt_rolling_optimizer import get_vectorbt_rolling_optimizer
from ...features_common.transforms.vectorbt_scaler import VectorBTScaler, create_vectorbt_scaler
from ..core.feature_bank import get_global_feature_bank

# Centralized indicators utility
from ..utils.centralized_indicators import get_centralized_indicators
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None
    LegacyRSIGenerator,
    LegacyMACDGenerator,
    LegacyStochasticGenerator,
    LegacyWilliamsRGenerator
)

class MomentumFeatureGenerator(VectorizedFeatureGenerator):
    """Feature generator for momentum-based features with optimization."""
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config()
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
    
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
        
        # Use optimized rolling operations for momentum calculation
        if ROLLING_OPTIMIZER_AVAILABLE and self._should_use_vectorbt(pd.DataFrame({'prices': prices_series})):
            try:
                # Calculate momentum using optimized rolling operations
                shifted_prices = prices_series.shift(period)
                momentum = prices_series - shifted_prices
                self.performance_stats['vectorbt_operations'] += 1
                return momentum.values
            except Exception as e:
                self.logger.warning(f"Optimized momentum calculation failed: {e}, using numpy fallback")
                self.performance_stats['pandas_fallbacks'] += 1
                momentum = prices - np.roll(prices, period)
                return momentum
        elif VECTORBT_AVAILABLE and self._should_use_vectorbt(pd.DataFrame({'prices': prices_series})):
            try:
                # Calculate momentum using VectorBT rolling operations
                shifted_prices = prices_series.shift(period)
                momentum = prices_series - shifted_prices
                self.performance_stats['vectorbt_operations'] += 1
                return momentum.values
            except Exception as e:
                self.logger.warning(f"VectorBT momentum calculation failed: {e}, using numpy fallback")
                self.performance_stats['pandas_fallbacks'] += 1
                momentum = prices - np.roll(prices, period)
                return momentum
        else:
            momentum = prices - np.roll(prices, period)
            return momentum

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

        return alignmentclass RSIGenerator(VectorizedFeatureGenerator):
    """Generator for RSI (Relative Strength Index) using centralized utilities."""

    def __init__(self,
                 period: int = 14,
                 base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_LEVELS,
                 normalize: bool = False,
                 normalization_method: str = 'zscore',
                 **base_kwargs):
        """
        Initialize RSI generator using centralized utilities.
        
        Args:
            period: RSI period
            base_calculation: Base calculation type (price_levels, returns_vwap, etc.)
            normalize: Whether to normalize the output
            normalization_method: Normalization method
            **base_kwargs: Additional parameters for base calculation
        """
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        # Create base calculator
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        
        # Update required columns based on base calculation
        required_columns = self.base_calculator.get_required_columns()
        
        config = FeatureConfig(
            name=f"rsi_{period}_{base_calculation.value}",
            category=FeatureCategory.MOMENTUM,
            description=f"RSI {period} using centralized utilities based on {base_calculation.value}",
            required_columns=required_columns,
            default_lookback=period * 2,
            min_lookback=period,
            max_lookback=period * 3,
            parameters={
                'period': period,
                'base_calculation': base_calculation.value,
                'normalize': normalize,
                'normalization_method': normalization_method,
                **base_kwargs
            }
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        
        self.period = period
        self.base_calculation = base_calculation
        self.normalize = normalize
        self.normalization_method = normalization_method
        
        # Initialize centralized utilities
        self.indicators = get_centralized_indicators()
        self.scaler = create_vectorbt_scaler(method=normalization_method) if normalize else None
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate RSI using centralized utilities."""
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        if self.base_calculation == BaseCalculationType.PRICE_LEVELS:
            # Use centralized RSI calculation for price levels
            rsi = self.indicators.calculate_rsi(data['close'], self.period)
        else:
            # For other base calculations, calculate RSI on base values
            base_values = self.base_calculator.calculate(data)
            rsi = self.indicators.calculate_rsi(base_values, self.period)
        
        # Apply normalization if requested
        if self.normalize and self.scaler:
            rsi = self.scaler.fit_transform(rsi)
        
        return rsiclass MACDGenerator(VectorizedFeatureGenerator):
    """Generator for MACD (Moving Average Convergence Divergence) using centralized utilities."""
    
    def __init__(self,
                 fast: int = 12,
                 slow: int = 26,
                 signal: int = 9,
                 base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_LEVELS,
                 normalize: bool = False,
                 normalization_method: str = 'zscore',
                 **base_kwargs):
        """
        Initialize MACD generator using centralized utilities.
        
        Args:
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line EMA period
            base_calculation: Base calculation type (price_levels, returns_vwap, etc.)
            normalize: Whether to normalize the output
            normalization_method: Normalization method
            **base_kwargs: Additional parameters for base calculation
        """
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        # Create base calculator
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        
        # Update required columns based on base calculation
        required_columns = self.base_calculator.get_required_columns()
        
        config = FeatureConfig(
            name=f"macd_{fast}_{slow}_{signal}_{base_calculation.value}",
            category=FeatureCategory.MOMENTUM,
            description=f"MACD {fast}/{slow}/{signal} using centralized utilities based on {base_calculation.value}",
            required_columns=required_columns,
            default_lookback=slow * 2,
            min_lookback=slow,
            max_lookback=slow * 3,
            parameters={
                'fast': fast,
                'slow': slow,
                'signal': signal,
                'base_calculation': base_calculation.value,
                'normalize': normalize,
                'normalization_method': normalization_method,
                **base_kwargs
            }
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        
        self.fast = fast
        self.slow = slow
        self.signal = signal
        self.base_calculation = base_calculation
        self.normalize = normalize
        self.normalization_method = normalization_method
        
        # Initialize centralized utilities
        self.indicators = get_centralized_indicators()
        self.scaler = create_vectorbt_scaler(method=normalization_method) if normalize else None
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate MACD using centralized utilities."""
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        # Calculate base values
        base_values = self.base_calculator.calculate(data)
        
        # Use centralized MACD calculation
        macd_line, signal_line, histogram = self.indicators.calculate_macd(
            base_values, self.fast, self.slow, self.signal
        )
        
        # Return MACD line (can be extended to return signal_line or histogram)
        result = macd_line
        
        # Apply normalization if requested
        if self.normalize and self.scaler:
            result = self.scaler.fit_transform(result)
        
        return resultclass StochasticGenerator(VectorizedFeatureGenerator):
    """Generator for Stochastic Oscillator using centralized utilities."""
    
    def __init__(self, 
                 k_period: int = 14, 
                 d_period: int = 3,
                 base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_LEVELS,
                 normalize: bool = False,
                 normalization_method: str = 'zscore',
                 **base_kwargs):
        """
        Initialize Stochastic generator using centralized utilities.
        
        Args:
            k_period: %K period
            d_period: %D period (smoothing)
            base_calculation: Base calculation type (price_levels, returns_vwap, etc.)
            normalize: Whether to normalize the output
            normalization_method: Normalization method
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
            description=f"Stochastic Oscillator {k_period}/{d_period} using centralized utilities based on {base_calculation.value}",
            required_columns=required_columns,
            default_lookback=k_period,
            min_lookback=k_period,
            max_lookback=k_period,
            parameters={
                'k_period': k_period,
                'd_period': d_period,
                'base_calculation': base_calculation.value,
                'normalize': normalize,
                'normalization_method': normalization_method,
                **base_kwargs
            }
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        
        self.k_period = k_period
        self.d_period = d_period
        self.base_calculation = base_calculation
        self.normalize = normalize
        self.normalization_method = normalization_method
        
        # Initialize centralized utilities
        self.indicators = get_centralized_indicators()
        self.scaler = create_vectorbt_scaler(method=normalization_method) if normalize else None
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate Stochastic Oscillator using centralized utilities."""
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        if self.base_calculation == BaseCalculationType.PRICE_LEVELS:
            # Use centralized Stochastic calculation for price levels
            k_percent, d_percent = self.indicators.calculate_stochastic(
                data['high'], data['low'], data['close'], self.k_period, self.d_period
            )
            result = k_percent
        else:
            # For other base calculations, calculate Stochastic on base values
            base_values = self.base_calculator.calculate(data)
            # For non-price-level base calculations, we need to create synthetic high/low
            # This is a simplified approach - in practice, you might want to handle this differently
            synthetic_high = base_values * 1.01  # 1% higher
            synthetic_low = base_values * 0.99   # 1% lower
            k_percent, d_percent = self.indicators.calculate_stochastic(
                synthetic_high, synthetic_low, base_values, self.k_period, self.d_period
            )
            result = k_percent
        
        # Apply normalization if requested
        if self.normalize and self.scaler:
            result = self.scaler.fit_transform(result)
        
        return resultclass WilliamsRGenerator(VectorizedFeatureGenerator):
    """Generator for Williams %R using centralized utilities."""
    
    def __init__(self, 
                 period: int = 14,
                 base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_LEVELS,
                 normalize: bool = False,
                 normalization_method: str = 'zscore',
                 **base_kwargs):
        """
        Initialize Williams %R generator using centralized utilities.
        
        Args:
            period: Williams %R period
            base_calculation: Base calculation type (price_levels, returns_vwap, etc.)
            normalize: Whether to normalize the output
            normalization_method: Normalization method
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
            description=f"Williams %R {period} using centralized utilities based on {base_calculation.value}",
            required_columns=required_columns,
            default_lookback=period,
            min_lookback=period,
            max_lookback=period,
            parameters={
                'period': period,
                'base_calculation': base_calculation.value,
                'normalize': normalize,
                'normalization_method': normalization_method,
                **base_kwargs
            }
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        
        self.period = period
        self.base_calculation = base_calculation
        self.normalize = normalize
        self.normalization_method = normalization_method
        
        # Initialize centralized utilities
        self.indicators = get_centralized_indicators()
        self.scaler = create_vectorbt_scaler(method=normalization_method) if normalize else None
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate Williams %R using centralized utilities."""
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        if self.base_calculation == BaseCalculationType.PRICE_LEVELS:
            # Use centralized Williams %R calculation for price levels
            williams_r = self.indicators.calculate_williams_r(
                data['high'], data['low'], data['close'], self.period
            )
        else:
            # For other base calculations, calculate Williams %R on base values
            base_values = self.base_calculator.calculate(data)
            # For non-price-level base calculations, we need to create synthetic high/low
            synthetic_high = base_values * 1.01  # 1% higher
            synthetic_low = base_values * 0.99   # 1% lower
            williams_r = self.indicators.calculate_williams_r(
                synthetic_high, synthetic_low, base_values, self.period
            )
        
        # Apply normalization if requested
        if self.normalize and self.scaler:
            williams_r = self.scaler.fit_transform(williams_r)
        
        return williams_rclass MomentumOscillatorGenerator(VectorizedFeatureGenerator):
    """Generator for Momentum Oscillator using centralized utilities."""
    
    def __init__(self, 
                 period: int = 10,
                 base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_LEVELS,
                 normalize: bool = False,
                 normalization_method: str = 'zscore',
                 **base_kwargs):
        """
        Initialize Momentum Oscillator generator using centralized utilities.
        
        Args:
            period: Momentum period
            base_calculation: Base calculation type (price_levels, returns_vwap, etc.)
            normalize: Whether to normalize the output
            normalization_method: Normalization method
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
            description=f"Momentum Oscillator {period} using centralized utilities based on {base_calculation.value}",
            required_columns=required_columns,
            default_lookback=period,
            min_lookback=period,
            max_lookback=period,
            parameters={
                'period': period,
                'base_calculation': base_calculation.value,
                'normalize': normalize,
                'normalization_method': normalization_method,
                **base_kwargs
            }
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        
        self.period = period
        self.base_calculation = base_calculation
        self.normalize = normalize
        self.normalization_method = normalization_method
        
        # Initialize centralized utilities
        self.indicators = get_centralized_indicators()
        self.scaler = create_vectorbt_scaler(method=normalization_method) if normalize else None
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate Momentum Oscillator using centralized utilities."""
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        # Calculate base values
        base_values = self.base_calculator.calculate(data)
        
        # Use centralized momentum calculation
        momentum = self.indicators.calculate_momentum(base_values, self.period)
        
        # Apply normalization if requested
        if self.normalize and self.scaler:
            momentum = self.scaler.fit_transform(momentum)
        
        return momentumclass RateOfChangeGenerator(VectorizedFeatureGenerator):
    """Generator for Rate of Change (ROC) using centralized utilities."""
    
    def __init__(self, 
                 period: int = 10,
                 base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_LEVELS,
                 normalize: bool = False,
                 normalization_method: str = 'zscore',
                 **base_kwargs):
        """
        Initialize ROC generator using centralized utilities.
        
        Args:
            period: ROC period
            base_calculation: Base calculation type (price_levels, returns_vwap, etc.)
            normalize: Whether to normalize the output
            normalization_method: Normalization method
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
            description=f"Rate of Change {period} using centralized utilities based on {base_calculation.value}",
            required_columns=required_columns,
            default_lookback=period,
            min_lookback=period,
            max_lookback=period,
            parameters={
                'period': period,
                'base_calculation': base_calculation.value,
                'normalize': normalize,
                'normalization_method': normalization_method,
                **base_kwargs
            }
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        
        self.period = period
        self.base_calculation = base_calculation
        self.normalize = normalize
        self.normalization_method = normalization_method
        
        # Initialize centralized utilities
        self.indicators = get_centralized_indicators()
        self.scaler = create_vectorbt_scaler(method=normalization_method) if normalize else None
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate ROC using centralized utilities."""
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        # Calculate base values
        base_values = self.base_calculator.calculate(data)
        
        # Use centralized ROC calculation
        roc = self.indicators.calculate_roc(base_values, self.period)
        
        # Apply normalization if requested
        if self.normalize and self.scaler:
            roc = self.scaler.fit_transform(roc)
        
        return roc

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


class VectorBTMomentumFeatureGenerator(VectorBTFeatureGenerator):
    """VectorBT-optimized momentum feature generator with comprehensive indicators."""
    
    def __init__(self, period: int = 14, config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config(period)
        super().__init__(config, enable_gpu=True, enable_parallel=True)
        self.period = period
    
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
    """Create default momentum generators using centralized utilities."""
    generators = []
    
    # Use refactored generators with centralized utilities
    # RSI generators with different periods
    rsi_periods = [9, 14, 21, 25, 30]
    for period in rsi_periods:
        generators.append(RSIGenerator(period, normalize=True))
        generators.append(RSIGenerator(period, normalize=False))
    
    # MACD generators with different configurations
    macd_configs = [(8, 21, 5), (12, 26, 9), (5, 35, 5), (16, 34, 9)]
    for fast, slow, signal in macd_configs:
        generators.append(MACDGenerator(fast, slow, signal, normalize=True))
        generators.append(MACDGenerator(fast, slow, signal, normalize=False))
    
    # Stochastic generators with different periods
    stochastic_periods = [9, 14, 21]
    for period in stochastic_periods:
        generators.append(StochasticGenerator(period, normalize=True))
        generators.append(StochasticGenerator(period, normalize=False))
    
    # Williams %R generators
    williams_periods = [9, 14, 21]
    for period in williams_periods:
        generators.append(WilliamsRGenerator(period, normalize=True))
        generators.append(WilliamsRGenerator(period, normalize=False))
    
    # Momentum generators
    momentum_periods = [5, 10, 20]
    for period in momentum_periods:
        generators.append(MomentumOscillatorGenerator(period, normalize=True))
        generators.append(MomentumOscillatorGenerator(period, normalize=False))
    
    # ROC generators
    roc_periods = [5, 10, 20]
    for period in roc_periods:
        generators.append(RateOfChangeGenerator(period, normalize=True))
        generators.append(RateOfChangeGenerator(period, normalize=False))
    
    # VectorBT-optimized generators (if available)
    if VECTORBT_AVAILABLE:
        for period in [9, 14, 21, 30]:
            generators.append(VectorBTMomentumFeatureGenerator(period))
            generators.append(VectorBTRSIGenerator(period))
            generators.append(VectorBTStochasticGenerator(period))
            
        # MACD with different parameters
        for fast in [8, 12, 16]:
            for slow in [21, 26, 34]:
                generators.append(VectorBTMACDGenerator(fast, slow))
    
    # Legacy generators for backward compatibility
    generators.extend([
        LegacyRSIGenerator(14),
        LegacyMACDGenerator(12, 26, 9),
        LegacyStochasticGenerator(14, 3),
    ])
    
    # Entropy-based momentum generators
    try:
        generators.extend([
            RSIEntropyGenerator(20, 14),
            MACDEntropyGenerator(20, 12, 26),
        ])
    except ImportError:
        pass  # Entropy generators might not be available
    
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