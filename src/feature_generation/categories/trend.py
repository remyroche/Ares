"""
Trend Feature Generator

This module provides feature generators for trend-based indicators,
including moving averages, trend lines, and trend strength measures.
Supports different base calculations: price returns, returns-based VWAP, etc.

Enhanced with VectorBT for maximum performance.
"""

import numpy as np
import pandas as pd
import warnings
import logging
from typing import Any, Dict, List, Optional, Union

from ..core.feature_generator import FeatureGenerator, FeatureResult, VectorizedFeatureGenerator, FeatureConfig, FeatureCategory
from ..core.vectorbt_optimization_mixin import VectorBTOptimizationMixin

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

# VectorBT Rolling Optimizer
try:
    from ..utils.vectorbt_rolling_optimizer import get_vectorbt_rolling_optimizer, VectorBTRollingOptimizer
    ROLLING_OPTIMIZER_AVAILABLE = True
except ImportError:
    ROLLING_OPTIMIZER_AVAILABLE = False
    get_vectorbt_rolling_optimizer = None
    VectorBTRollingOptimizer = None
    from ..utils.vectorbt_rolling_optimizer import get_vectorbt_rolling_optimizer
    VECTORBT_OPTIMIZER_AVAILABLE = True
except ImportError:
    VECTORBT_OPTIMIZER_AVAILABLE = False

# Optimization utilities
try:
    from ..utils.vectorization_optimizer import get_vectorization_optimizer
    from ..utils.optimized_feature_pipeline import get_optimized_feature_pipeline
    OPTIMIZATION_AVAILABLE = True
except ImportError:
    OPTIMIZATION_AVAILABLE = False

# Optional GPU acceleration
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

from ..base_calculations import (
    BaseCalculator,
    BaseCalculationType,
    BaseCalculationConfig,
    create_base_calculator
)

logger = logging.getLogger(__name__)

class TrendFeatureGenerator(VectorizedFeatureGenerator, VectorBTOptimizationMixin):
    """Feature generator for trend-based features with VectorBT optimization."""
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config()
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        
        # Initialize VectorBT optimizer
        if VECTORBT_OPTIMIZER_AVAILABLE:
            self.vectorbt_optimizer = get_vectorbt_rolling_optimizer(enable_gpu=False, enable_parallel=True)
        else:
            self.vectorbt_optimizer = None
    
    @classmethod
    def _create_default_config(cls) -> FeatureConfig:
        return FeatureConfig(
            name="trend_features",
            category=FeatureCategory.TREND,
            description="Comprehensive trend-based features including moving averages and trend indicators",
            required_columns=["close"],
            optional_columns=["high", "low", "open"],
            default_lookback=20,
            min_lookback=2,
            max_lookback=50,
            parameters={
                "sma_periods": [5, 10, 20, 50],
                "ema_periods": [12, 26],
                "trend_windows": [10, 20]
            },
            matrix_optimized=True,
            gpu_accelerated=False
        )
    
    @classmethod
    def create_default(cls) -> 'TrendFeatureGenerator':
        return cls()
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        close_prices = data['close']
        
        # Use VectorBT for SMA calculation
        if self.vectorbt_optimizer and self._should_use_vectorbt(close_prices):
            try:
                sma = self.vectorbt_optimizer.rolling_mean(close_prices, window=20)
                return sma.rename('sma_20')
            except Exception as e:
                self.logger.warning(f"VectorBT SMA calculation failed: {e}, using pandas fallback")
                sma = self._calculate_sma(close_prices.values, period=20)
                return pd.Series(sma, index=data.index, name='sma_20')
        else:
            sma = self._calculate_sma(close_prices.values, period=20)
            return pd.Series(sma, index=data.index, name='sma_20')
    
    def _calculate_sma(self, prices: np.ndarray, period: int = 20) -> np.ndarray:
        if len(prices) < period:
            return np.full(len(prices), np.nan)

        prices_series = pd.Series(prices)
        
        # Use VectorBT if available and data is large enough
        if self.vectorbt_optimizer and self._should_use_vectorbt(prices_series):
            try:
                sma = self.vectorbt_optimizer.rolling_mean(prices_series, window=period)
                return sma.values
            except Exception as e:
                self.logger.warning(f"VectorBT SMA calculation failed: {e}, using pandas fallback")
                return prices_series.rolling(window=period).mean().values
                self.performance_stats['pandas_fallbacks'] += 1
                return self._calculate_sma_vectorized(prices_series, period).values
        else:
            return self._calculate_sma_vectorized(prices_series, period).values

    def _calculate_ema(self, prices: np.ndarray, period: int = 20) -> np.ndarray:
        """Calculate Exponential Moving Average."""
        if len(prices) < period:
            return np.full(len(prices), np.nan)

        ema = pd.Series(prices).ewm(span=period).mean().values
        return ema

    def _calculate_adx(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
        """
        Calculate Average Directional Index (ADX).

        Args:
            high: High prices
            low: Low prices
            close: Close prices
            period: ADX period (default 14)

        Returns:
            ADX values
        """
        if len(high) < period or len(low) < period or len(close) < period:
            return np.full(len(close), np.nan)

        # Calculate True Range
        tr = np.maximum.reduce([
            high - low,
            np.abs(high - np.roll(close, 1)),
            np.abs(low - np.roll(close, 1))
        ])
        tr[0] = np.nan  # First value is NaN

        # Calculate Directional Movement
        dm_plus = np.maximum(high - np.roll(high, 1), 0)
        dm_minus = np.maximum(np.roll(low, 1) - low, 0)

        # Convert to pandas Series for rolling operations
        tr_series = pd.Series(tr)
        dm_plus_series = pd.Series(dm_plus)
        dm_minus_series = pd.Series(dm_minus)

        # Calculate Directional Indicators using VectorBT optimization
        if self.vectorbt_optimizer and self._should_use_vectorbt(tr_series):
            try:
                dm_plus_mean = self.vectorbt_optimizer.rolling_mean(dm_plus_series, window=period)
                dm_minus_mean = self.vectorbt_optimizer.rolling_mean(dm_minus_series, window=period)
                tr_mean = self.vectorbt_optimizer.rolling_mean(tr_series, window=period)
            except Exception as e:
                logger.warning(f"VectorBT ADX calculation failed: {e}, using pandas fallback")
                dm_plus_mean = dm_plus_series.rolling(period).mean()
                dm_minus_mean = dm_minus_series.rolling(period).mean()
                tr_mean = tr_series.rolling(period).mean()
        else:
            dm_plus_mean = dm_plus_series.rolling(period).mean()
            dm_minus_mean = dm_minus_series.rolling(period).mean()
            tr_mean = tr_series.rolling(period).mean()
        
        di_plus = 100 * (dm_plus_mean / tr_mean)
        di_minus = 100 * (dm_minus_mean / tr_mean)

        # Calculate ADX
        dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
        
        if self.vectorbt_optimizer and self._should_use_vectorbt(pd.Series(dx)):
            try:
                adx = self.vectorbt_optimizer.rolling_mean(pd.Series(dx), window=period)
            except Exception as e:
                logger.warning(f"VectorBT ADX rolling mean failed: {e}, using pandas fallback")
                adx = pd.Series(dx).rolling(period).mean()
        else:
            adx = pd.Series(dx).rolling(period).mean()

        return adx.values

    def _calculate_directional_signal(self, prices: np.ndarray) -> np.ndarray:
        """
        Calculate directional signal as EMA_8 - EMA_20.

        Args:
            prices: Price data

        Returns:
            Directional signal values
        """
        ema_8 = self._calculate_ema(prices, period=8)
        ema_20 = self._calculate_ema(prices, period=20)

        # Calculate directional signal
        directional_signal = ema_8 - ema_20

        return directional_signal

    def _calculate_trend_score(self, prices: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
        """
        Calculate trend score as normalized directional signal multiplied by ADX.

        Args:
            prices: Price data
            high: High prices
            low: Low prices
            close: Close prices

        Returns:
            Trend score values
        """
        # Calculate directional signal
        directional_signal = self._calculate_directional_signal(prices)

        # Calculate ADX
        adx = self._calculate_adx(high, low, close, period=14)

        # Normalize directional signal to [-1, 1] range
        signal_max = np.nanmax(np.abs(directional_signal))
        if signal_max > 0:
            normalized_signal = directional_signal / signal_max
        else:
            normalized_signal = directional_signal

        # Calculate trend score
        trend_score = normalized_signal * adx

        return trend_score    def _should_use_vectorbt(self, data) -> bool:
        """Determine if VectorBT should be used based on data size and configuration."""
        return (VECTORBT_AVAILABLE and 
                len(data) >= getattr(self, 'vectorbt_threshold', 1000))
    
    def _vectorbt_rolling_operation(self, data: pd.Series, operation: str, 
                                  window: int, **kwargs) -> pd.Series:
        """Perform VectorBT rolling operation with fallback to pandas."""
        if not self._should_use_vectorbt(data):
            return self._pandas_rolling_operation(data, operation, window, **kwargs)
        
        try:
            if operation == 'mean':
                return self.vectorbt_optimizer.rolling_mean(data, window=window, **kwargs)
            elif operation == 'std':
                return self.vectorbt_optimizer.rolling_std(data, window=window, **kwargs)
            elif operation == 'var':
                return self.vectorbt_optimizer.rolling_var(data, window=window, **kwargs)
            elif operation == 'min':
                return self.vectorbt_optimizer.rolling_min(data, window=window, **kwargs)
            elif operation == 'max':
                return self.vectorbt_optimizer.rolling_max(data, window=window, **kwargs)
            elif operation == 'sum':
                return self.vectorbt_optimizer.rolling_sum(data, window=window, **kwargs)
            else:
                raise ValueError(f"Unsupported operation: {operation}")
        except Exception as e:
            self.logger.warning(f"VectorBT operation failed: {e}, using pandas fallback")
            return self._pandas_rolling_operation(data, operation, window, **kwargs)
    
    def _pandas_rolling_operation(self, data: pd.Series, operation: str, 
                                 window: int, **kwargs) -> pd.Series:
        """Fallback rolling operation using pandas."""
        if operation == 'mean':
            return self._calculate_sma_vectorized(data, window)
        elif operation == 'std':
            return self._calculate_rolling_std_vectorized(data, window)
        elif operation == 'var':
            return data.rolling(window=window).var()
        elif operation == 'min':
            return self._calculate_rolling_min_vectorized(data, window)
        elif operation == 'max':
            return self._calculate_rolling_max_vectorized(data, window)
        elif operation == 'sum':
            return self._calculate_rolling_sum_vectorized(data, window)
        else:
            raise ValueError(f"Unsupported operation: {operation}")

class ADXGenerator(VectorizedFeatureGenerator, VectorBTOptimizationMixin):
    """Generator for Average Directional Index (ADX) with VectorBT optimization."""

    def __init__(self, period: int = 14):
        """
        Initialize ADX generator.

        Args:
            period: ADX period (default 14)
        """
        config = FeatureConfig(
            name=f"adx_{period}",
            category=FeatureCategory.TREND,
            description=f"Average Directional Index over {period} periods",
            required_columns=["high", "low", "close"],
            default_lookback=period * 2,  # Need more data for ADX calculation
            min_lookback=period * 2,
            max_lookback=period * 2,
            parameters={
                'period': period
            }
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.period = period
        
        # Initialize VectorBT optimizer
        if VECTORBT_OPTIMIZER_AVAILABLE:
            self.vectorbt_optimizer = get_vectorbt_rolling_optimizer(enable_gpu=False, enable_parallel=True)
        else:
            self.vectorbt_optimizer = None

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        high = data['high']
        low = data['low']
        close = data['close']

        # Use VectorBT for ADX calculation
        if self.vectorbt_optimizer and self._should_use_vectorbt(close):
            try:
                # VectorBT doesn't have direct ADX, so we use our custom calculation
                adx = self._calculate_adx(high.values, low.values, close.values, period=self.period)
                return pd.Series(adx, index=data.index, name=f'adx_{self.period}')
            except Exception as e:
                self.logger.warning(f"VectorBT ADX calculation failed: {e}, using pandas fallback")
                adx = self._calculate_adx(high.values, low.values, close.values, period=self.period)
                return pd.Series(adx, index=data.index, name=f'adx_{self.period}')
        else:
            adx = self._calculate_adx(high.values, low.values, close.values, period=self.period)
            return pd.Series(adx, index=data.index, name=f'adx_{self.period}')

    def _calculate_adx(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
        """Calculate Average Directional Index (ADX)."""
        if len(high) < period or len(low) < period or len(close) < period:
            return np.full(len(close), np.nan)

        # Calculate True Range
        tr = np.maximum.reduce([
            high - low,
            np.abs(high - np.roll(close, 1)),
            np.abs(low - np.roll(close, 1))
        ])
        tr[0] = np.nan  # First value is NaN

        # Calculate Directional Movement
        dm_plus = np.maximum(high - np.roll(high, 1), 0)
        dm_minus = np.maximum(np.roll(low, 1) - low, 0)

        # Convert to pandas Series for rolling operations
        dm_plus_series = pd.Series(dm_plus)
        dm_minus_series = pd.Series(dm_minus)
        tr_series = pd.Series(tr)

        # Calculate Directional Indicators using VectorBT optimization
        if self.vectorbt_optimizer and self._should_use_vectorbt(tr_series):
            try:
                dm_plus_mean = self.vectorbt_optimizer.rolling_mean(dm_plus_series, window=period)
                dm_minus_mean = self.vectorbt_optimizer.rolling_mean(dm_minus_series, window=period)
                tr_mean = self.vectorbt_optimizer.rolling_mean(tr_series, window=period)
            except Exception as e:
                logger.warning(f"VectorBT ADX calculation failed: {e}, using pandas fallback")
                dm_plus_mean = dm_plus_series.rolling(period).mean()
                dm_minus_mean = dm_minus_series.rolling(period).mean()
                tr_mean = tr_series.rolling(period).mean()
        else:
            dm_plus_mean = dm_plus_series.rolling(period).mean()
            dm_minus_mean = dm_minus_series.rolling(period).mean()
            tr_mean = tr_series.rolling(period).mean()

        # Calculate Directional Indicators
        di_plus = 100 * (dm_plus_mean / tr_mean)
        di_minus = 100 * (dm_minus_mean / tr_mean)

        # Calculate ADX
        dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
        
        if self.vectorbt_optimizer and self._should_use_vectorbt(pd.Series(dx)):
            try:
                adx = self.vectorbt_optimizer.rolling_mean(pd.Series(dx), window=period)
            except Exception as e:
                logger.warning(f"VectorBT ADX rolling mean failed: {e}, using pandas fallback")
                adx = pd.Series(dx).rolling(period).mean()
        else:
            adx = pd.Series(dx).rolling(period).mean()

        return adx.values

    
    def optimize_dataframe_processing(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame for vectorized processing."""
        if hasattr(self, 'vectorization_optimizer') and self.vectorization_optimizer:
            return self.vectorization_optimizer.optimize_dataframe_processing(data)
        return data
    
    def vectorized_rolling_operations(self, data: pd.DataFrame, operations: List[str], 
                                    windows: List[int], columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Perform vectorized rolling operations with hardware optimization."""
        if hasattr(self, 'vectorization_optimizer') and self.vectorization_optimizer:
            return self.vectorization_optimizer.vectorized_rolling_operations(
                data, operations, windows, columns
            )
        return data

class DirectionalSignalGenerator(VectorizedFeatureGenerator, VectorBTOptimizationMixin):
    """Generator for Directional Signal (EMA_8 - EMA_20) with VectorBT optimization."""
# ADXGenerator moved to oscillator.py to avoid duplication
# Import from oscillator module instead
from .oscillator import ADXGenerator
class DirectionalSignalGenerator(VectorizedFeatureGenerator):
    """Generator for Directional Signal (EMA_8 - EMA_20)."""

    def __init__(self):
        """Initialize Directional Signal generator."""
        config = FeatureConfig(
            name="directional_signal",
            category=FeatureCategory.TREND,
            description="Directional signal calculated as EMA_8 - EMA_20",
            required_columns=["close"],
            default_lookback=20,  # Need enough data for both EMAs
            min_lookback=20,
            max_lookback=20,
            parameters={}
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        
        # Initialize VectorBT optimizer
        if VECTORBT_OPTIMIZER_AVAILABLE:
            self.vectorbt_optimizer = get_vectorbt_rolling_optimizer(enable_gpu=False, enable_parallel=True)
        else:
            self.vectorbt_optimizer = None

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        prices = data['close']

        # Use VectorBT for directional signal calculation
        if self.vectorbt_optimizer and self._should_use_vectorbt(prices):
            try:
                # Calculate EMA using VectorBT
                ema_8 = self._calculate_ema_vectorized(prices, 8)
                ema_20 = self._calculate_ema_vectorized(prices, 20)
                directional_signal = ema_8 - ema_20
                return directional_signal.rename('directional_signal')
            except Exception as e:
                self.logger.warning(f"VectorBT directional signal calculation failed: {e}, using pandas fallback")
                directional_signal = self._calculate_directional_signal(prices.values)
                return pd.Series(directional_signal, index=data.index, name='directional_signal')
        else:
            directional_signal = self._calculate_directional_signal(prices.values)
            return pd.Series(directional_signal, index=data.index, name='directional_signal')

    def _calculate_ema(self, prices: np.ndarray, period: int = 20) -> np.ndarray:
        """Calculate Exponential Moving Average."""
        if len(prices) < period:
            return np.full(len(prices), np.nan)

        ema = pd.Series(prices).ewm(span=period).mean().values
        return ema

    def _calculate_directional_signal(self, prices: np.ndarray) -> np.ndarray:
        """Calculate directional signal as EMA_8 - EMA_20."""
        ema_8 = self._calculate_ema(prices, period=8)
        ema_20 = self._calculate_ema(prices, period=20)

        # Calculate directional signal
        directional_signal = ema_8 - ema_20

        return directional_signal

    
    def optimize_dataframe_processing(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame for vectorized processing."""
        if hasattr(self, 'vectorization_optimizer') and self.vectorization_optimizer:
            return self.vectorization_optimizer.optimize_dataframe_processing(data)
        return data
    
    def vectorized_rolling_operations(self, data: pd.DataFrame, operations: List[str], 
                                    windows: List[int], columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Perform vectorized rolling operations with hardware optimization."""
        if hasattr(self, 'vectorization_optimizer') and self.vectorization_optimizer:
            return self.vectorization_optimizer.vectorized_rolling_operations(
                data, operations, windows, columns
            )
        return data

class TrendScoreGenerator(VectorizedFeatureGenerator, VectorBTOptimizationMixin):
    """Generator for Trend Score (normalized directional signal * ADX) with VectorBT optimization."""
        class TrendScoreGenerator(VectorizedFeatureGenerator):
    """Generator for Trend Score (normalized directional signal * ADX)."""

    def __init__(self, adx_period: int = 14):
        """
        Initialize Trend Score generator.

        Args:
            adx_period: ADX period (default 14)
        """
        config = FeatureConfig(
            name=f"trend_score_{adx_period}",
            category=FeatureCategory.TREND,
            description=f"Trend score calculated as normalized directional signal multiplied by ADX ({adx_period})",
            required_columns=["close", "high", "low"],
            default_lookback=adx_period * 2,  # Need enough data for both calculations
            min_lookback=adx_period * 2,
            max_lookback=adx_period * 2,
            parameters={
                'adx_period': adx_period
            }
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.adx_period = adx_period
        
        # Initialize VectorBT optimizer
        if VECTORBT_OPTIMIZER_AVAILABLE:
            self.vectorbt_optimizer = get_vectorbt_rolling_optimizer(enable_gpu=False, enable_parallel=True)
        else:
            self.vectorbt_optimizer = None

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        prices = data['close']
        high = data['high']
        low = data['low']
        close = data['close']

        # Use VectorBT for trend score calculation
        if self.vectorbt_optimizer and self._should_use_vectorbt(prices):
            try:
                # Calculate directional signal using VectorBT
                ema_8 = self._calculate_ema_vectorized(prices, 8)
                ema_20 = self._calculate_ema_vectorized(prices, 20)
                directional_signal = ema_8 - ema_20

                # Calculate ADX using our custom method
                adx = self._calculate_adx(high.values, low.values, close.values, period=self.adx_period)
                adx_series = pd.Series(adx, index=data.index)

                # Normalize directional signal to [-1, 1] range
                signal_max = np.nanmax(np.abs(directional_signal))
                if signal_max > 0:
                    normalized_signal = directional_signal / signal_max
                else:
                    normalized_signal = directional_signal

                # Calculate trend score
                trend_score = normalized_signal * adx_series
                return trend_score.rename(f'trend_score_{self.adx_period}')
            except Exception as e:
                self.logger.warning(f"VectorBT trend score calculation failed: {e}, using pandas fallback")
                trend_score = self._calculate_trend_score(prices.values, high.values, low.values, close.values)
                return pd.Series(trend_score, index=data.index, name=f'trend_score_{self.adx_period}')
        else:
            trend_score = self._calculate_trend_score(prices.values, high.values, low.values, close.values)
            return pd.Series(trend_score, index=data.index, name=f'trend_score_{self.adx_period}')

    def _calculate_ema(self, prices: np.ndarray, period: int = 20) -> np.ndarray:
        """Calculate Exponential Moving Average."""
        if len(prices) < period:
            return np.full(len(prices), np.nan)

        ema = pd.Series(prices).ewm(span=period).mean().values
        return ema

    def _calculate_adx(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:


    def _calculate_directional_signal(self, prices: np.ndarray) -> np.ndarray:
        """Calculate directional signal as EMA_8 - EMA_20."""
        ema_8 = self._calculate_ema(prices, period=8)
        ema_20 = self._calculate_ema(prices, period=20)

        # Calculate directional signal
        directional_signal = ema_8 - ema_20

        return directional_signal

    def _calculate_trend_score(self, prices: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
        """Calculate trend score as normalized directional signal multiplied by ADX."""
        # Calculate directional signal
        directional_signal = self._calculate_directional_signal(prices)

        # Calculate ADX
        adx = self._calculate_adx(high, low, close, period=self.adx_period)

        # Normalize directional signal to [-1, 1] range
        signal_max = np.nanmax(np.abs(directional_signal))
        if signal_max > 0:
            normalized_signal = directional_signal / signal_max
        else:
            normalized_signal = directional_signal

        # Calculate trend score
        trend_score = normalized_signal * adx

        return trend_score

    
    def optimize_dataframe_processing(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame for vectorized processing."""
        if hasattr(self, 'vectorization_optimizer') and self.vectorization_optimizer:
            return self.vectorization_optimizer.optimize_dataframe_processing(data)
        return data
    
    def vectorized_rolling_operations(self, data: pd.DataFrame, operations: List[str], 
                                    windows: List[int], columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Perform vectorized rolling operations with hardware optimization."""
        if hasattr(self, 'vectorization_optimizer') and self.vectorization_optimizer:
            return self.vectorization_optimizer.vectorized_rolling_operations(
                data, operations, windows, columns
            )
        return data

class SMAGenerator(VectorizedFeatureGenerator, VectorBTOptimizationMixin):
    """Generator for Simple Moving Average with different base calculations and VectorBT optimization."""
        return trend_scoreclass SMAGenerator(VectorizedFeatureGenerator):
    """Generator for Simple Moving Average with different base calculations."""

    def __init__(self,
                 period: int = 20,
                 base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.RETURNS_VWAP,
                 **base_kwargs):
        """
        Initialize SMA generator.
        
        Args:
            period: SMA period
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
            name=f"sma_{period}_{base_calculation.value}",
            category=FeatureCategory.TREND,
            description=f"Simple Moving Average over {period} periods based on {base_calculation.value}",
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
        
        # Initialize VectorBT optimizer
        if VECTORBT_OPTIMIZER_AVAILABLE:
            self.vectorbt_optimizer = get_vectorbt_rolling_optimizer(enable_gpu=False, enable_parallel=True)
        else:
            self.vectorbt_optimizer = None
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        # Calculate base values
        base_values = self.base_calculator.calculate(data)
        
        # Use VectorBT for SMA calculation
        if self.vectorbt_optimizer and self._should_use_vectorbt(base_values):
            try:
                sma = self.vectorbt_optimizer.rolling_mean(base_values, window=self.period)
                return sma
            except Exception as e:
                self.logger.warning(f"VectorBT SMA calculation failed: {e}, using pandas fallback")
                sma = base_values.rolling(window=self.period).mean()
                return sma
        else:
            sma = base_values.rolling(window=self.period).mean()
            return sma

    
    def optimize_dataframe_processing(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame for vectorized processing."""
        if hasattr(self, 'vectorization_optimizer') and self.vectorization_optimizer:
            return self.vectorization_optimizer.optimize_dataframe_processing(data)
        return data
    
    def vectorized_rolling_operations(self, data: pd.DataFrame, operations: List[str], 
                                    windows: List[int], columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Perform vectorized rolling operations with hardware optimization."""
        if hasattr(self, 'vectorization_optimizer') and self.vectorization_optimizer:
            return self.vectorization_optimizer.vectorized_rolling_operations(
                data, operations, windows, columns
            )
        return data

class EMAGenerator(VectorizedFeatureGenerator, VectorBTOptimizationMixin):
    """Generator for Exponential Moving Average with different base calculations and VectorBT optimization."""
            return smaclass EMAGenerator(VectorizedFeatureGenerator):
    """Generator for Exponential Moving Average with different base calculations."""

    def __init__(self,
                 period: int = 20,
                 base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.RETURNS_VWAP,
                 **base_kwargs):
        """
        Initialize EMA generator.
        
        Args:
            period: EMA period
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
            name=f"ema_{period}_{base_calculation.value}",
            category=FeatureCategory.TREND,
            description=f"Exponential Moving Average over {period} periods based on {base_calculation.value}",
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
        
        # Initialize VectorBT optimizer
        if VECTORBT_OPTIMIZER_AVAILABLE:
            self.vectorbt_optimizer = get_vectorbt_rolling_optimizer(enable_gpu=False, enable_parallel=True)
        else:
            self.vectorbt_optimizer = None
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        # Calculate base values
        base_values = self.base_calculator.calculate(data)
        
        # Use VectorBT for EMA calculation
        if self.vectorbt_optimizer and self._should_use_vectorbt(base_values):
            try:
                # VectorBT doesn't have direct EMA, so we use ewm
                ema = base_values.ewm(span=self.period).mean()
                return ema
            except Exception as e:
                self.logger.warning(f"VectorBT EMA calculation failed: {e}, using pandas fallback")
                ema = base_values.ewm(span=self.period).mean()
                return ema
        else:
            ema = base_values.ewm(span=self.period).mean()
            return ema

def create_trend_generators(periods: Dict[str, List[int]] = None) -> List[FeatureGenerator]:
    """Create a set of trend feature generators."""
    if periods is None:
        periods = {
            'sma': [5, 10, 20, 50],
            'ema': [12, 26]
        }
    
    generators = []
    
    # SMA generators
    for period in periods.get('sma', [20]):
        generators.append(SMAGenerator(period))
    
    # EMA generators
    for period in periods.get('ema', [12, 26]):
        generators.append(EMAGenerator(period))
    
    return generators

def create_default_trend_generators() -> List[FeatureGenerator]:
    return create_trend_generators()

# WMA (Weighted Moving Average)class WMAGenerator(VectorizedFeatureGenerator):
    """Generator for WMA (Weighted Moving Average) with different base calculations."""
    
    def __init__(self, 
                 period: int = 20,
                 base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS,
                 **base_kwargs):
        """
        Initialize WMA generator.
        
        Args:
            period: WMA period
            base_calculation: Base calculation type (price_returns, returns_vwap, etc.)
            **base_kwargs: Additional parameters for base calculation
        """
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        # Create base calculator
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        
        # Update required columns based on base calculation
        required_columns = self.base_calculator.get_required_columns()
        
        config = FeatureConfig(
            name=f"wma_{period}_{base_calculation.value}",
            category=FeatureCategory.TREND,
            description=f"Weighted Moving Average over {period} periods based on {base_calculation.value}",
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

        """Generate WMA based on the specified base calculation."""
        base_values = self.base_calculator.calculate(data)
        
        # Use VectorBT for WMA calculation
        if self.vectorbt_optimizer and self._should_use_vectorbt(base_values):
            try:
                # VectorBT doesn't have direct WMA, so we use rolling apply
                weights = np.arange(1, self.period + 1)
                wma = base_values.rolling(window=self.period).apply(
                    lambda x: np.average(x, weights=weights)
                )
                return wma
            except Exception as e:
                self.logger.warning(f"VectorBT WMA calculation failed: {e}, using pandas fallback")
                weights = np.arange(1, self.period + 1)
                wma = base_values.rolling(window=self.period).apply(
                    lambda x: np.average(x, weights=weights)
                )
                return wma
        else:
            weights = np.arange(1, self.period + 1)
            wma = base_values.rolling(window=self.period).apply(
                lambda x: np.average(x, weights=weights)
            )
            return wma

# DEMA (Double Exponential Moving Average)class DEMAGenerator(VectorizedFeatureGenerator):
    """Generator for DEMA (Double Exponential Moving Average) with different base calculations."""
    
    def __init__(self, 
                 period: int = 20,
                 base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS,
                 **base_kwargs):
        """
        Initialize DEMA generator.
        
        Args:
            period: DEMA period
            base_calculation: Base calculation type (price_returns, returns_vwap, etc.)
            **base_kwargs: Additional parameters for base calculation
        """
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        # Create base calculator
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        
        # Update required columns based on base calculation
        required_columns = self.base_calculator.get_required_columns()
        
        config = FeatureConfig(
            name=f"dema_{period}_{base_calculation.value}",
            category=FeatureCategory.TREND,
            description=f"Double Exponential Moving Average over {period} periods based on {base_calculation.value}",
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

        """Generate DEMA based on the specified base calculation."""
        base_values = self.base_calculator.calculate(data)
        
        # Use VectorBT for DEMA calculation
        if self.vectorbt_optimizer and self._should_use_vectorbt(base_values):
            try:
                # Calculate DEMA using ewm
                ema1 = base_values.ewm(span=self.period).mean()
                ema2 = ema1.ewm(span=self.period).mean()
                dema = 2 * ema1 - ema2
                return dema
            except Exception as e:
                self.logger.warning(f"VectorBT DEMA calculation failed: {e}, using pandas fallback")
                ema1 = base_values.ewm(span=self.period).mean()
                ema2 = ema1.ewm(span=self.period).mean()
                dema = 2 * ema1 - ema2
                return dema
        else:
            ema1 = base_values.ewm(span=self.period).mean()
            ema2 = ema1.ewm(span=self.period).mean()
            dema = 2 * ema1 - ema2
            return dema

# TEMA (Triple Exponential Moving Average)class TEMAGenerator(VectorizedFeatureGenerator):
    """Generator for TEMA (Triple Exponential Moving Average) with different base calculations."""
    
    def __init__(self, 
                 period: int = 20,
                 base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS,
                 **base_kwargs):
        """
        Initialize TEMA generator.
        
        Args:
            period: TEMA period
            base_calculation: Base calculation type (price_returns, returns_vwap, etc.)
            **base_kwargs: Additional parameters for base calculation
        """
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        # Create base calculator
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        
        # Update required columns based on base calculation
        required_columns = self.base_calculator.get_required_columns()
        
        config = FeatureConfig(
            name=f"tema_{period}_{base_calculation.value}",
            category=FeatureCategory.TREND,
            description=f"Triple Exponential Moving Average over {period} periods based on {base_calculation.value}",
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

        """Generate TEMA based on the specified base calculation."""
        base_values = self.base_calculator.calculate(data)
        
        # Use VectorBT for TEMA calculation
        if self.vectorbt_optimizer and self._should_use_vectorbt(base_values):
            try:
                # Calculate TEMA using ewm
                ema1 = base_values.ewm(span=self.period).mean()
                ema2 = ema1.ewm(span=self.period).mean()
                ema3 = ema2.ewm(span=self.period).mean()
                tema = 3 * ema1 - 3 * ema2 + ema3
                return tema
            except Exception as e:
                self.logger.warning(f"VectorBT TEMA calculation failed: {e}, using pandas fallback")
                ema1 = base_values.ewm(span=self.period).mean()
                ema2 = ema1.ewm(span=self.period).mean()
                ema3 = ema2.ewm(span=self.period).mean()
                tema = 3 * ema1 - 3 * ema2 + ema3
                return tema
        else:
            ema1 = base_values.ewm(span=self.period).mean()
            ema2 = ema1.ewm(span=self.period).mean()
            ema3 = ema2.ewm(span=self.period).mean()
            tema = 3 * ema1 - 3 * ema2 + ema3
            return tema

# TRIMA (Triangular Moving Average)class TRIMAGenerator(VectorizedFeatureGenerator):
    """Generator for TRIMA (Triangular Moving Average) with different base calculations."""
    
    def __init__(self, 
                 period: int = 20,
                 base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS,
                 **base_kwargs):
        """
        Initialize TRIMA generator.
        
        Args:
            period: TRIMA period
            base_calculation: Base calculation type (price_returns, returns_vwap, etc.)
            **base_kwargs: Additional parameters for base calculation
        """
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        # Create base calculator
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        
        # Update required columns based on base calculation
        required_columns = self.base_calculator.get_required_columns()
        
        config = FeatureConfig(
            name=f"trima_{period}_{base_calculation.value}",
            category=FeatureCategory.TREND,
            description=f"Triangular Moving Average over {period} periods based on {base_calculation.value}",
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

        """Generate TRIMA based on the specified base calculation."""
        base_values = self.base_calculator.calculate(data)
        
        # Use VectorBT for TRIMA calculation
        if self.vectorbt_optimizer and self._should_use_vectorbt(base_values):
            try:
                # Calculate TRIMA using VectorBT rolling mean
                half_period = self.period // 2
                trima = self.vectorbt_optimizer.rolling_mean(base_values, window=half_period).rolling(window=half_period).mean()
                return trima
            except Exception as e:
                self.logger.warning(f"VectorBT TRIMA calculation failed: {e}, using pandas fallback")
                half_period = self.period // 2
                trima = self._calculate_sma_vectorized(base_values, half_period).rolling(window=half_period).mean()
                return trima
        else:
            half_period = self.period // 2
            trima = self._calculate_sma_vectorized(base_values, half_period).rolling(window=half_period).mean()
            return trima

# MAMA (MESA Adaptive Moving Average)class MAMAGenerator(VectorizedFeatureGenerator):
    """Generator for MAMA (MESA Adaptive Moving Average) with different base calculations."""
    
    def __init__(self, 
                 fast_limit: float = 0.5,
                 slow_limit: float = 0.05,
                 base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS,
                 **base_kwargs):
        """
        Initialize MAMA generator.
        
        Args:
            fast_limit: Fast limit
            slow_limit: Slow limit
            base_calculation: Base calculation type (price_returns, returns_vwap, etc.)
            **base_kwargs: Additional parameters for base calculation
        """
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        # Create base calculator
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        
        # Update required columns based on base calculation
        required_columns = self.base_calculator.get_required_columns()
        
        config = FeatureConfig(
            name=f"mama_{fast_limit}_{slow_limit}_{base_calculation.value}",
            category=FeatureCategory.TREND,
            description=f"MESA Adaptive Moving Average with fast_limit={fast_limit}, slow_limit={slow_limit} based on {base_calculation.value}",
            required_columns=required_columns,
            default_lookback=20,
            min_lookback=1,
            max_lookback=50,
            parameters={
                'fast_limit': fast_limit,
                'slow_limit': slow_limit,
                'base_calculation': base_calculation.value,
                **base_kwargs
            }
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.fast_limit = fast_limit
        self.slow_limit = slow_limit
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate MAMA based on the specified base calculation."""
        base_values = self.base_calculator.calculate(data)
        
        # Use VectorBT for MAMA calculation
        if self.vectorbt_optimizer and self._should_use_vectorbt(base_values):
            try:
                # Calculate MAMA (simplified version) using ewm
                mama = self._calculate_ema_vectorized(base_values, 20)
                return mama
            except Exception as e:
                self.logger.warning(f"VectorBT MAMA calculation failed: {e}, using pandas fallback")
                mama = self._calculate_ema_vectorized(base_values, 20)
                return mama
        else:
            mama = self._calculate_ema_vectorized(base_values, 20)
            return mama

# VWMA (Volume Weighted Moving Average)class VWMAGenerator(VectorizedFeatureGenerator):
    """Generator for VWMA (Volume Weighted Moving Average) with different base calculations."""
    
    def __init__(self, 
                 period: int = 20,
                 base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS,
                 **base_kwargs):
        """
        Initialize VWMA generator.
        
        Args:
            period: VWMA period
            base_calculation: Base calculation type (price_returns, returns_vwap, etc.)
            **base_kwargs: Additional parameters for base calculation
        """
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        # Create base calculator
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        
        # Update required columns based on base calculation
        required_columns = self.base_calculator.get_required_columns()
        if 'volume' not in required_columns:
            required_columns.append('volume')
        
        config = FeatureConfig(
            name=f"vwma_{period}_{base_calculation.value}",
            category=FeatureCategory.TREND,
            description=f"Volume Weighted Moving Average over {period} periods based on {base_calculation.value}",
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

        """Generate VWMA based on the specified base calculation."""
        base_values = self.base_calculator.calculate(data)
        volume = data['volume']
        
        # Calculate VWMA using VectorBT optimization
        if self.vectorbt_optimizer and self._should_use_vectorbt(base_values):
            try:
                numerator = self.vectorbt_optimizer.rolling_sum(base_values * volume, window=self.period)
                denominator = self.vectorbt_optimizer.rolling_sum(volume, window=self.period)
                vwma = numerator / denominator
            except Exception as e:
                logger.warning(f"VectorBT VWMA calculation failed: {e}, using pandas fallback")
                vwma = (base_values * volume).rolling(window=self.period).sum() / volume.rolling(window=self.period).sum()
        else:
            vwma = (base_values * volume).rolling(window=self.period).sum() / volume.rolling(window=self.period).sum()
        
        return vwma


    
    def optimize_dataframe_processing(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame for vectorized processing."""
        if hasattr(self, 'vectorization_optimizer') and self.vectorization_optimizer:
            return self.vectorization_optimizer.optimize_dataframe_processing(data)
        return data
    
    def vectorized_rolling_operations(self, data: pd.DataFrame, operations: List[str], 
                                    windows: List[int], columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Perform vectorized rolling operations with hardware optimization."""
        if hasattr(self, 'vectorization_optimizer') and self.vectorization_optimizer:
            return self.vectorization_optimizer.vectorized_rolling_operations(
                data, operations, windows, columns
            )
        return data

class KeltnerChannelsGenerator(VectorizedFeatureGenerator):
        # Use VectorBT for VWMA calculation
        if VECTORBT_AVAILABLE:
            try:
                # Calculate VWMA using VectorBT rolling sum
                price_volume = base_values * volume
                price_volume_sum = rolling_sum(price_volume, window=self.period)
                volume_sum = rolling_sum(volume, window=self.period)
                vwma = price_volume_sum / volume_sum
                return vwma
            except Exception as e:
                self.logger.warning(f"VectorBT VWMA calculation failed: {e}, using pandas fallback")
                vwma = (base_values * volume).rolling(window=self.period).sum() / volume.rolling(window=self.period).sum()
                return vwma
        else:
            vwma = (base_values * volume).rolling(window=self.period).sum() / volume.rolling(window=self.period).sum()
            return vwmaclass KeltnerChannelsGenerator(VectorizedFeatureGenerator):
    """Generator for Keltner Channels with different base calculations."""
    
    def __init__(self,
                 period: int = 20,
                 atr_period: int = 14,
                 multiplier: float = 2.0,
                 base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS,
                 **base_kwargs):
        """
        Initialize Keltner Channels generator.
        
        Args:
            period: EMA period for middle line
            atr_period: ATR period for channel width
            multiplier: ATR multiplier for channel width
            base_calculation: Base calculation type
            **base_kwargs: Additional parameters for base calculation
        """
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        # Create base calculator
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        
        # Update required columns based on base calculation
        required_columns = self.base_calculator.get_required_columns()
        if base_calculation == BaseCalculationType.PRICE_LEVELS:
            required_columns.extend(["high", "low"])  # ATR requires high/low
        
        config = FeatureConfig(
            name=f"keltner_channels_{period}_{atr_period}_{base_calculation.value}",
            category=FeatureCategory.TREND,
            description=f"Keltner Channels with EMA period={period}, ATR period={atr_period}, multiplier={multiplier} based on {base_calculation.value}",
            required_columns=required_columns,
            default_lookback=max(period, atr_period),
            min_lookback=max(period, atr_period),
            max_lookback=max(period, atr_period),
            parameters={
                'period': period,
                'atr_period': atr_period,
                'multiplier': multiplier,
                'base_calculation': base_calculation.value,
                **base_kwargs
            }
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.period = period
        self.atr_period = atr_period
        self.multiplier = multiplier
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate Keltner Channels middle line (EMA) based on the specified base calculation."""
        if self.base_calculator.config.calculation_type == BaseCalculationType.PRICE_LEVELS:
            # Traditional Keltner Channels calculation on price levels
            close = data['close']
            high = data['high']
            low = data['low']
            
            # Calculate EMA of close prices (middle line)
            ema = close.ewm(span=self.period).mean()
            
            # Calculate ATR
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            # Use VectorBT for optimized ATR calculation
            if self.vectorbt_optimizer and self._should_use_vectorbt(true_range):
                try:
                    atr = self.vectorbt_optimizer.rolling_mean(true_range, window=self.atr_period)
                except Exception as e:
                    logger.warning(f"VectorBT ATR calculation failed: {e}, using pandas fallback")
                    atr = true_range.rolling(window=self.atr_period).mean()
            else:
                atr = true_range.rolling(window=self.atr_period).mean()
            
            # Return middle line (EMA) as the main feature
            # Upper and lower bands would be: ema ± (multiplier * atr)
            return ema
        else:
            # For other base calculations, use EMA of base values
            base_values = self.base_calculator.calculate(data)
            ema = base_values.ewm(span=self.period).mean()
            return ema


def create_trend_generators(periods: Dict[str, List[int]] = None) -> List[FeatureGenerator]:
    """Create a set of trend feature generators."""
    if periods is None:
        periods = {
            'sma': [5, 10, 20, 50],
            'ema': [12, 26],
            'wma': [20],
            'dema': [21],
            'tema': [21],
            'trima': [21],
            'mama': [0.5, 0.05],
            'vwma': [20],
            'keltner_channels': [20],
            'adx': [14],
            'trend_score': [14]
        }
    
    generators = []
    
    # SMA generators
    for period in periods.get('sma', [20]):
        generators.append(SMAGenerator(period))
    
    # EMA generators  
    for period in periods.get('ema', [12, 26]):
        generators.append(EMAGenerator(period))
    
    # WMA generators
    for period in periods.get('wma', [20]):
        generators.append(WMAGenerator(period))
    
    # DEMA generators
    for period in periods.get('dema', [21]):
        generators.append(DEMAGenerator(period))
    
    # TEMA generators
    for period in periods.get('tema', [21]):
        generators.append(TEMAGenerator(period))
    
    # TRIMA generators
    for period in periods.get('trima', [21]):
        generators.append(TRIMAGenerator(period))
    
    # VWMA generators
    for period in periods.get('vwma', [20]):
        generators.append(VWMAGenerator(period))
    
    # Keltner Channels generators
    for period in periods.get('keltner_channels', [20]):
        generators.append(KeltnerChannelsGenerator(period))

    # ADX generators
    for period in periods.get('adx', [14]):
        generators.append(ADXGenerator(period))

    # Directional Signal generators
    generators.append(DirectionalSignalGenerator())

    # Trend Score generators
    for period in periods.get('trend_score', [14]):
        generators.append(TrendScoreGenerator(adx_period=period))

    return generators


class VectorBTTrendFeatureGenerator(VectorBTFeatureGenerator):
    """VectorBT-optimized trend feature generator with comprehensive indicators."""
    
    def __init__(self, period: int = 20, config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config(period)
        super().__init__(config, enable_gpu=True, enable_parallel=True)
        self.period = period
    
    @classmethod
    def _create_default_config(cls, period: int = 20) -> FeatureConfig:
        return FeatureConfig(
            name=f"vectorbt_trend_comprehensive_{period}",
            category=FeatureCategory.TREND,
            description=f"VectorBT-optimized comprehensive trend features over {period} periods",
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
        """Generate comprehensive trend features using VectorBT."""
        if data.empty:
            return pd.Series(dtype=float, index=data.index, name=f'vectorbt_trend_{self.period}')
        
        # Generate multiple trend indicators using VectorBT
        operations = [
            {'type': 'rolling', 'name': 'sma', 'params': {'operation': 'mean', 'window': self.period, 'column': 'close'}},
            {'type': 'indicator', 'name': 'ema', 'params': {'indicator': 'ema', 'window': self.period}},
            {'type': 'indicator', 'name': 'wma', 'params': {'indicator': 'wma', 'window': self.period}},
            {'type': 'indicator', 'name': 'dema', 'params': {'indicator': 'dema', 'window': self.period}},
            {'type': 'indicator', 'name': 'tema', 'params': {'indicator': 'tema', 'window': self.period}},
            {'type': 'indicator', 'name': 'kama', 'params': {'indicator': 'kama', 'window': self.period}},
            {'type': 'indicator', 'name': 'adx', 'params': {'indicator': 'adx', 'window': self.period}},
            {'type': 'indicator', 'name': 'plus_di', 'params': {'indicator': 'plus_di', 'window': self.period}},
            {'type': 'indicator', 'name': 'minus_di', 'params': {'indicator': 'minus_di', 'window': self.period}},
            {'type': 'indicator', 'name': 'aroon_up', 'params': {'indicator': 'aroon_up', 'window': self.period}},
            {'type': 'indicator', 'name': 'aroon_down', 'params': {'indicator': 'aroon_down', 'window': self.period}}
        ]
        
        # Use batch operations for efficiency
        results = self._vectorbt_batch_operations(data, operations)
        
        # Combine results into a single trend measure
        if not results.empty:
            # Weighted combination of different trend measures
            trend = (
                0.20 * results.get('sma', 0) +
                0.15 * results.get('ema', 0) +
                0.15 * results.get('wma', 0) +
                0.10 * results.get('dema', 0) +
                0.10 * results.get('tema', 0) +
                0.10 * results.get('kama', 0) +
                0.10 * results.get('adx', 0) +
                0.05 * results.get('plus_di', 0) +
                0.05 * results.get('minus_di', 0)
            )
        else:
            # Fallback to simple SMA
            trend = self._vectorbt_rolling_operation(data['close'], 'mean', window=self.period)
        
        return trend.rename(f'vectorbt_trend_{self.period}')


class VectorBTSMAGenerator(VectorBTFeatureGenerator):
    """VectorBT-optimized Simple Moving Average generator."""
    
    def __init__(self, period: int = 20, config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config(period)
        super().__init__(config, enable_gpu=True, enable_parallel=True)
        self.period = period
    
    @classmethod
    def _create_default_config(cls, period: int = 20) -> FeatureConfig:
        return FeatureConfig(
            name=f"vectorbt_sma_{period}",
            category=FeatureCategory.TREND,
            description=f"VectorBT-optimized SMA over {period} periods",
            required_columns=["close"],
            optional_columns=["high", "low", "open"],
            default_lookback=period,
            min_lookback=period,
            max_lookback=period,
            parameters={"period": period},
            matrix_optimized=True,
            gpu_accelerated=True
        )
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate SMA using VectorBT rolling mean."""
        if data.empty:
            return pd.Series(dtype=float, index=data.index, name=f'vectorbt_sma_{self.period}')
        
        # Generate SMA using VectorBT rolling mean
        sma = self._vectorbt_rolling_operation(data['close'], 'mean', window=self.period)
        
        return sma.rename(f'vectorbt_sma_{self.period}')


class VectorBTEMAGenerator(VectorBTFeatureGenerator):
    """VectorBT-optimized Exponential Moving Average generator."""
    
    def __init__(self, period: int = 20, config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config(period)
        super().__init__(config, enable_gpu=True, enable_parallel=True)
        self.period = period
    
    @classmethod
    def _create_default_config(cls, period: int = 20) -> FeatureConfig:
        return FeatureConfig(
            name=f"vectorbt_ema_{period}",
            category=FeatureCategory.TREND,
            description=f"VectorBT-optimized EMA over {period} periods",
            required_columns=["close"],
            optional_columns=["high", "low", "open"],
            default_lookback=period,
            min_lookback=period,
            max_lookback=period,
            parameters={"period": period},
            matrix_optimized=True,
            gpu_accelerated=True
        )
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate EMA using VectorBT."""
        if data.empty:
            return pd.Series(dtype=float, index=data.index, name=f'vectorbt_ema_{self.period}')
        
        # Generate EMA using VectorBT
        ema = self._vectorbt_technical_indicator(data, 'ema', window=self.period)
        
        return ema.rename(f'vectorbt_ema_{self.period}')


class VectorBTADXGenerator(VectorBTFeatureGenerator):
    """VectorBT-optimized ADX generator."""
    
    def __init__(self, period: int = 14, config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config(period)
        super().__init__(config, enable_gpu=True, enable_parallel=True)
        self.period = period
    
    @classmethod
    def _create_default_config(cls, period: int = 14) -> FeatureConfig:
        return FeatureConfig(
            name=f"vectorbt_adx_{period}",
            category=FeatureCategory.TREND,
            description=f"VectorBT-optimized ADX over {period} periods",
            required_columns=["high", "low", "close"],
            optional_columns=["open"],
            default_lookback=period,
            min_lookback=period,
            max_lookback=period,
            parameters={"period": period},
            matrix_optimized=True,
            gpu_accelerated=True
        )
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate ADX using VectorBT."""
        if data.empty:
            return pd.Series(dtype=float, index=data.index, name=f'vectorbt_adx_{self.period}')
        
        # Generate ADX using VectorBT
        adx = self._vectorbt_technical_indicator(data, 'adx', window=self.period)
        
        return adx.rename(f'vectorbt_adx_{self.period}')


def create_default_trend_generators() -> List[FeatureGenerator]:
    """Create default trend generators with VectorBT optimization."""
    generators = []
    
    if VECTORBT_AVAILABLE:
        # VectorBT-optimized generators
        for period in [5, 10, 20, 50, 100]:
            generators.append(VectorBTTrendFeatureGenerator(period))
            generators.append(VectorBTSMAGenerator(period))
            generators.append(VectorBTEMAGenerator(period))
            
        # ADX with different periods
        for period in [9, 14, 21]:
            generators.append(VectorBTADXGenerator(period))
        
        # Advanced trend indicators
        # Ichimoku Cloud with different parameters
        for tenkan in [9, 12]:
            for kijun in [26, 30]:
                generators.append(VectorBTIchimokuCloudGenerator(tenkan, kijun, 52, 26))
        
        # Parabolic SAR with different parameters
        for acc in [0.02, 0.05, 0.1]:
            for max_acc in [0.2, 0.3]:
                generators.append(VectorBTParabolicSARGenerator(acc, max_acc))
        
        # ZigZag with different parameters
        for deviation in [3.0, 5.0, 7.0, 10.0]:
            for backstep in [2, 3, 5]:
                generators.append(VectorBTZigZagGenerator(deviation, backstep))
    else:
        # Fallback to original generators
        periods = {
            'sma': [5, 10, 20, 50, 100],
            'ema': [12, 26, 50],
            'wma': [20],
            'dema': [21],
            'tema': [21],
            'trima': [21],
            'vwma': [20],
            'keltner_channels': [20],
            'adx': [14],
            'trend_score': [14]
        }
        
        # SMA generators
        for period in periods.get('sma', [20]):
            generators.append(SMAGenerator(period))
        
        # EMA generators
        for period in periods.get('ema', [12, 26]):
            generators.append(EMAGenerator(period))
        
        # WMA generators
        for period in periods.get('wma', [20]):
            generators.append(WMAGenerator(period))
        
        # DEMA generators
        for period in periods.get('dema', [21]):
            generators.append(DEMAGenerator(period))
        
        # TEMA generators
        for period in periods.get('tema', [21]):
            generators.append(TEMAGenerator(period))
        
        # TRIMA generators
        for period in periods.get('trima', [21]):
            generators.append(TRIMAGenerator(period))
        
        # VWMA generators
        for period in periods.get('vwma', [20]):
            generators.append(VWMAGenerator(period))
        
        # Keltner Channels generators
        for period in periods.get('keltner_channels', [20]):
            generators.append(KeltnerChannelsGenerator(period))
    # All generators now use VectorBT optimization through the mixin
    periods = {
        'sma': [5, 10, 20, 50, 100],
        'ema': [12, 26, 50],
        'wma': [20],
        'dema': [21],
        'tema': [21],
        'trima': [21],
        'vwma': [20],
        'keltner_channels': [20],
        'adx': [14],
        'trend_score': [14]
    }
    
    # SMA generators
    for period in periods.get('sma', [20]):
        generators.append(SMAGenerator(period))
    
    # EMA generators
    for period in periods.get('ema', [12, 26]):
        generators.append(EMAGenerator(period))
    
    # WMA generators
    for period in periods.get('wma', [20]):
        generators.append(WMAGenerator(period))
    
    # DEMA generators
    for period in periods.get('dema', [21]):
        generators.append(DEMAGenerator(period))
    
    # TEMA generators
    for period in periods.get('tema', [21]):
        generators.append(TEMAGenerator(period))
    
    # TRIMA generators
    for period in periods.get('trima', [21]):
        generators.append(TRIMAGenerator(period))
    
    # VWMA generators
    for period in periods.get('vwma', [20]):
        generators.append(VWMAGenerator(period))
    
    # Keltner Channels generators
    for period in periods.get('keltner_channels', [20]):
        generators.append(KeltnerChannelsGenerator(period))

    # ADX generators
    for period in periods.get('adx', [14]):
        generators.append(ADXGenerator(period))

    # Directional Signal generators
    generators.append(DirectionalSignalGenerator())

    # Trend Score generators
    for period in periods.get('trend_score', [14]):
        generators.append(TrendScoreGenerator(adx_period=period))

    return generators


class VectorBTIchimokuCloudGenerator(VectorBTFeatureGenerator):
    """VectorBT-optimized Ichimoku Cloud generator."""
    
    def __init__(self, tenkan_period: int = 9, kijun_period: int = 26, senkou_span_b_period: int = 52, 
                 displacement: int = 26, config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config(tenkan_period, kijun_period, senkou_span_b_period, displacement)
        super().__init__(config, enable_gpu=True, enable_parallel=True)
        self.tenkan_period = tenkan_period
        self.kijun_period = kijun_period
        self.senkou_span_b_period = senkou_span_b_period
        self.displacement = displacement
        
        # Initialize VectorBT rolling optimizer
        if ROLLING_OPTIMIZER_AVAILABLE:
            self.rolling_optimizer = get_vectorbt_rolling_optimizer(enable_gpu=True, enable_parallel=True)
        else:
            self.rolling_optimizer = None
    
    @classmethod
    def _create_default_config(cls, tenkan_period: int = 9, kijun_period: int = 26, 
                              senkou_span_b_period: int = 52, displacement: int = 26) -> FeatureConfig:
        return FeatureConfig(
            name=f"vectorbt_ichimoku_cloud_{tenkan_period}_{kijun_period}_{senkou_span_b_period}",
            category=FeatureCategory.TREND,
            description=f"VectorBT-optimized Ichimoku Cloud with Tenkan={tenkan_period}, Kijun={kijun_period}, Senkou Span B={senkou_span_b_period}",
            required_columns=["high", "low", "close"],
            optional_columns=["open"],
            default_lookback=max(tenkan_period, kijun_period, senkou_span_b_period) + displacement,
            min_lookback=max(tenkan_period, kijun_period, senkou_span_b_period) + displacement,
            max_lookback=max(tenkan_period, kijun_period, senkou_span_b_period) + displacement,
            parameters={
                "tenkan_period": tenkan_period,
                "kijun_period": kijun_period,
                "senkou_span_b_period": senkou_span_b_period,
                "displacement": displacement
            },
            matrix_optimized=True,
            gpu_accelerated=True
        )
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate Ichimoku Cloud features using VectorBT."""
        if data.empty or not all(col in data.columns for col in ['high', 'low', 'close']):
            return pd.Series(dtype=float, index=data.index, name=f'vectorbt_ichimoku_cloud_{self.tenkan_period}')
        
        try:
            high = data['high']
            low = data['low']
            close = data['close']
            
            # Calculate Tenkan-sen (Conversion Line)
            if self.rolling_optimizer:
                try:
                    tenkan_high = self.rolling_optimizer.rolling_max(high, window=self.tenkan_period)
                    tenkan_low = self.rolling_optimizer.rolling_min(low, window=self.tenkan_period)
                except Exception as e:
                    self.logger.warning(f"VectorBT rolling optimizer failed: {e}, using fallback")
                    tenkan_high = high.rolling(window=self.tenkan_period).max()
                    tenkan_low = low.rolling(window=self.tenkan_period).min()
            else:
                tenkan_high = high.rolling(window=self.tenkan_period).max()
                tenkan_low = low.rolling(window=self.tenkan_period).min()
            
            tenkan_sen = (tenkan_high + tenkan_low) / 2
            
            # Calculate Kijun-sen (Base Line)
            if self.rolling_optimizer:
                try:
                    kijun_high = self.rolling_optimizer.rolling_max(high, window=self.kijun_period)
                    kijun_low = self.rolling_optimizer.rolling_min(low, window=self.kijun_period)
                except Exception as e:
                    self.logger.warning(f"VectorBT rolling optimizer failed: {e}, using fallback")
                    kijun_high = high.rolling(window=self.kijun_period).max()
                    kijun_low = low.rolling(window=self.kijun_period).min()
            else:
                kijun_high = high.rolling(window=self.kijun_period).max()
                kijun_low = low.rolling(window=self.kijun_period).min()
            
            kijun_sen = (kijun_high + kijun_low) / 2
            
            # Calculate Senkou Span A (Leading Span A)
            senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(self.displacement)
            
            # Calculate Senkou Span B (Leading Span B)
            if self.rolling_optimizer:
                try:
                    senkou_high = self.rolling_optimizer.rolling_max(high, window=self.senkou_span_b_period)
                    senkou_low = self.rolling_optimizer.rolling_min(low, window=self.senkou_span_b_period)
                except Exception as e:
                    self.logger.warning(f"VectorBT rolling optimizer failed: {e}, using fallback")
                    senkou_high = high.rolling(window=self.senkou_span_b_period).max()
                    senkou_low = low.rolling(window=self.senkou_span_b_period).min()
            else:
                senkou_high = high.rolling(window=self.senkou_span_b_period).max()
                senkou_low = low.rolling(window=self.senkou_span_b_period).min()
            
            senkou_span_b = ((senkou_high + senkou_low) / 2).shift(self.displacement)
            
            # Calculate Chikou Span (Lagging Span)
            chikou_span = close.shift(-self.displacement)
            
            # Return the cloud position indicator (price relative to cloud)
            cloud_position = close - ((senkou_span_a + senkou_span_b) / 2)
            
            return cloud_position.rename(f'vectorbt_ichimoku_cloud_{self.tenkan_period}')
            
        except Exception as e:
            self.logger.error(f"Error generating Ichimoku Cloud: {e}")
            return pd.Series(np.nan, index=data.index, name=f'vectorbt_ichimoku_cloud_{self.tenkan_period}')


class VectorBTParabolicSARGenerator(VectorBTFeatureGenerator):
    """VectorBT-optimized Parabolic SAR generator."""
    
    def __init__(self, acceleration: float = 0.02, maximum: float = 0.2, config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config(acceleration, maximum)
        super().__init__(config, enable_gpu=True, enable_parallel=True)
        self.acceleration = acceleration
        self.maximum = maximum
        
        # Initialize VectorBT rolling optimizer
        if ROLLING_OPTIMIZER_AVAILABLE:
            self.rolling_optimizer = get_vectorbt_rolling_optimizer(enable_gpu=True, enable_parallel=True)
        else:
            self.rolling_optimizer = None
    
    @classmethod
    def _create_default_config(cls, acceleration: float = 0.02, maximum: float = 0.2) -> FeatureConfig:
        return FeatureConfig(
            name=f"vectorbt_parabolic_sar_{acceleration}_{maximum}",
            category=FeatureCategory.TREND,
            description=f"VectorBT-optimized Parabolic SAR with acceleration={acceleration}, maximum={maximum}",
            required_columns=["high", "low", "close"],
            optional_columns=["open"],
            default_lookback=50,  # Need enough data for SAR calculation
            min_lookback=10,
            max_lookback=100,
            parameters={
                "acceleration": acceleration,
                "maximum": maximum
            },
            matrix_optimized=True,
            gpu_accelerated=True
        )
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate Parabolic SAR using VectorBT."""
        if data.empty or not all(col in data.columns for col in ['high', 'low', 'close']):
            return pd.Series(dtype=float, index=data.index, name=f'vectorbt_parabolic_sar_{self.acceleration}')
        
        try:
            high = data['high']
            low = data['low']
            close = data['close']
            
            # Initialize SAR arrays
            sar = np.zeros(len(close))
            trend = np.zeros(len(close))
            af = np.zeros(len(close))
            ep = np.zeros(len(close))
            
            # Initialize first values
            sar[0] = low.iloc[0]
            trend[0] = 1  # 1 for uptrend, -1 for downtrend
            af[0] = self.acceleration
            ep[0] = high.iloc[0]
            
            # Calculate SAR using VectorBT-optimized operations where possible
            for i in range(1, len(close)):
                # Calculate previous SAR
                prev_sar = sar[i-1]
                prev_trend = trend[i-1]
                prev_af = af[i-1]
                prev_ep = ep[i-1]
                
                # Calculate new SAR
                new_sar = prev_sar + prev_af * (prev_ep - prev_sar)
                
                # Check for trend reversal
                if prev_trend == 1:  # Uptrend
                    if low.iloc[i] <= new_sar:
                        # Trend reversal to downtrend
                        trend[i] = -1
                        sar[i] = prev_ep
                        ep[i] = low.iloc[i]
                        af[i] = self.acceleration
                    else:
                        # Continue uptrend
                        trend[i] = 1
                        sar[i] = new_sar
                        if high.iloc[i] > prev_ep:
                            ep[i] = high.iloc[i]
                            af[i] = min(prev_af + self.acceleration, self.maximum)
                        else:
                            ep[i] = prev_ep
                            af[i] = prev_af
                else:  # Downtrend
                    if high.iloc[i] >= new_sar:
                        # Trend reversal to uptrend
                        trend[i] = 1
                        sar[i] = prev_ep
                        ep[i] = high.iloc[i]
                        af[i] = self.acceleration
                    else:
                        # Continue downtrend
                        trend[i] = -1
                        sar[i] = new_sar
                        if low.iloc[i] < prev_ep:
                            ep[i] = low.iloc[i]
                            af[i] = min(prev_af + self.acceleration, self.maximum)
                        else:
                            ep[i] = prev_ep
                            af[i] = prev_af
            
            # Calculate SAR signal (price relative to SAR)
            sar_signal = close - pd.Series(sar, index=close.index)
            
            return sar_signal.rename(f'vectorbt_parabolic_sar_{self.acceleration}')
            
        except Exception as e:
            self.logger.error(f"Error generating Parabolic SAR: {e}")
            return pd.Series(np.nan, index=data.index, name=f'vectorbt_parabolic_sar_{self.acceleration}')


class VectorBTZigZagGenerator(VectorBTFeatureGenerator):
    """VectorBT-optimized ZigZag indicator generator."""
    
    def __init__(self, deviation: float = 5.0, backstep: int = 3, config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config(deviation, backstep)
        super().__init__(config, enable_gpu=True, enable_parallel=True)
        self.deviation = deviation
        self.backstep = backstep
        
        # Initialize VectorBT rolling optimizer
        if ROLLING_OPTIMIZER_AVAILABLE:
            self.rolling_optimizer = get_vectorbt_rolling_optimizer(enable_gpu=True, enable_parallel=True)
        else:
            self.rolling_optimizer = None
    
    @classmethod
    def _create_default_config(cls, deviation: float = 5.0, backstep: int = 3) -> FeatureConfig:
        return FeatureConfig(
            name=f"vectorbt_zigzag_{deviation}_{backstep}",
            category=FeatureCategory.TREND,
            description=f"VectorBT-optimized ZigZag indicator with deviation={deviation}%, backstep={backstep}",
            required_columns=["high", "low", "close"],
            optional_columns=["open"],
            default_lookback=50,  # Need enough data for ZigZag calculation
            min_lookback=10,
            max_lookback=200,
            parameters={
                "deviation": deviation,
                "backstep": backstep
            },
            matrix_optimized=True,
            gpu_accelerated=True
        )
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate ZigZag indicator using VectorBT."""
        if data.empty or not all(col in data.columns for col in ['high', 'low', 'close']):
            return pd.Series(dtype=float, index=data.index, name=f'vectorbt_zigzag_{self.deviation}')
        
        try:
            high = data['high']
            low = data['low']
            close = data['close']
            
            # Initialize ZigZag arrays
            zigzag = np.zeros(len(close))
            last_high_idx = 0
            last_low_idx = 0
            last_high = high.iloc[0]
            last_low = low.iloc[0]
            direction = 0  # 0 = neutral, 1 = up, -1 = down
            
            # Calculate ZigZag points
            for i in range(1, len(close)):
                current_high = high.iloc[i]
                current_low = low.iloc[i]
                
                if direction == 0:  # Neutral - looking for first significant move
                    if current_high > last_high * (1 + self.deviation / 100):
                        direction = 1
                        last_high_idx = i
                        last_high = current_high
                        zigzag[i] = current_high
                    elif current_low < last_low * (1 - self.deviation / 100):
                        direction = -1
                        last_low_idx = i
                        last_low = current_low
                        zigzag[i] = current_low
                elif direction == 1:  # Uptrend - looking for peak
                    if current_high > last_high:
                        # New high found
                        zigzag[last_high_idx] = 0  # Remove previous high
                        last_high_idx = i
                        last_high = current_high
                        zigzag[i] = current_high
                    elif current_low < last_high * (1 - self.deviation / 100):
                        # Significant pullback - start downtrend
                        direction = -1
                        last_low_idx = i
                        last_low = current_low
                        zigzag[i] = current_low
                else:  # Downtrend - looking for trough
                    if current_low < last_low:
                        # New low found
                        zigzag[last_low_idx] = 0  # Remove previous low
                        last_low_idx = i
                        last_low = current_low
                        zigzag[i] = current_low
                    elif current_high > last_low * (1 + self.deviation / 100):
                        # Significant bounce - start uptrend
                        direction = 1
                        last_high_idx = i
                        last_high = current_high
                        zigzag[i] = current_high
            
            # Calculate ZigZag trend strength
            zigzag_series = pd.Series(zigzag, index=close.index)
            non_zero_points = zigzag_series[zigzag_series != 0]
            
            if len(non_zero_points) > 1:
                # Calculate trend strength as the slope of the last ZigZag segment
                last_two_points = non_zero_points.tail(2)
                if len(last_two_points) == 2:
                    trend_strength = (last_two_points.iloc[-1] - last_two_points.iloc[0]) / len(last_two_points)
                else:
                    trend_strength = 0
            else:
                trend_strength = 0
            
            # Create trend strength series
            trend_strength_series = pd.Series(trend_strength, index=close.index)
            
            return trend_strength_series.rename(f'vectorbt_zigzag_{self.deviation}')
            
        except Exception as e:
            self.logger.error(f"Error generating ZigZag indicator: {e}")
            return pd.Series(np.nan, index=data.index, name=f'vectorbt_zigzag_{self.deviation}')


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
            return self._calculate_sma_vectorized(data, window)
        elif operation == 'std':
            return self._calculate_rolling_std_vectorized(data, window)
        elif operation == 'var':
            return data.rolling(window=window).var()
        elif operation == 'min':
            return self._calculate_rolling_min_vectorized(data, window)
        elif operation == 'max':
            return self._calculate_rolling_max_vectorized(data, window)
        elif operation == 'sum':
            return self._calculate_rolling_sum_vectorized(data, window)
        else:
            raise ValueError(f"Unsupported operation: {operation}")
