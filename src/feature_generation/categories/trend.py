"""
Trend Feature Generator

This module provides feature generators for trend-based indicators,
including moving averages, trend lines, and trend strength measures.
Supports different base calculations: price returns, returns-based VWAP, etc.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union

from ..core.feature_generator import FeatureGenerator, FeatureResult, VectorizedFeatureGenerator, FeatureConfig, FeatureCategory

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

class TrendFeatureGenerator(VectorizedFeatureGenerator):
    """Feature generator for trend-based features."""
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config()
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
    
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

        close_prices = data['close'].values
        sma = self._calculate_sma(close_prices, period=20)
        return pd.Series(sma, index=data.index, name='sma_20')
    
    def _calculate_sma(self, prices: np.ndarray, period: int = 20) -> np.ndarray:
        if len(prices) < period:
            return np.full(len(prices), np.nan)

        sma = pd.Series(prices).rolling(window=period).mean().values
        return sma

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

        # Calculate Directional Indicators
        di_plus = 100 * (dm_plus_series.rolling(period).mean() / tr_series.rolling(period).mean())
        di_minus = 100 * (dm_minus_series.rolling(period).mean() / tr_series.rolling(period).mean())

        # Calculate ADX
        dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
        adx = dx.rolling(period).mean()

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

class ADXGenerator(FeatureGenerator):
    """Generator for Average Directional Index (ADX)."""

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

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate ADX values."""
        high = data['high'].values
        low = data['low'].values
        close = data['close'].values

        adx = self._calculate_adx(high, low, close, period=self.period)

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

        # Calculate Directional Indicators using pandas rolling
        dm_plus_series = pd.Series(dm_plus)
        dm_minus_series = pd.Series(dm_minus)
        tr_series = pd.Series(tr)

        # Calculate Directional Indicators
        di_plus = 100 * (dm_plus_series.rolling(period).mean() / tr_series.rolling(period).mean())
        di_minus = 100 * (dm_minus_series.rolling(period).mean() / tr_series.rolling(period).mean())

        # Calculate ADX
        dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
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

class DirectionalSignalGenerator(FeatureGenerator):
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

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate directional signal values."""
        prices = data['close'].values

        directional_signal = self._calculate_directional_signal(prices)

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

class TrendScoreGenerator(FeatureGenerator):
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

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate trend score values."""
        prices = data['close'].values
        high = data['high'].values
        low = data['low'].values
        close = data['close'].values

        trend_score = self._calculate_trend_score(prices, high, low, close)

        return pd.Series(trend_score, index=data.index, name=f'trend_score_{self.adx_period}')

    def _calculate_ema(self, prices: np.ndarray, period: int = 20) -> np.ndarray:
        """Calculate Exponential Moving Average."""
        if len(prices) < period:
            return np.full(len(prices), np.nan)

        ema = pd.Series(prices).ewm(span=period).mean().values
        return ema

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

        # Calculate Directional Indicators
        # Convert to pandas Series for rolling operations
        dm_plus_series = pd.Series(dm_plus)
        dm_minus_series = pd.Series(dm_minus)
        tr_series = pd.Series(tr)
        
        di_plus = 100 * (dm_plus_series.rolling(period).mean() / tr_series.rolling(period).mean())
        di_minus = 100 * (dm_minus_series.rolling(period).mean() / tr_series.rolling(period).mean())

        # Calculate ADX
        dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
        dx_series = pd.Series(dx)
        adx = dx_series.rolling(period).mean()

        return adx.values

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

class SMAGenerator(FeatureGenerator):
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
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate SMA based on the specified base calculation."""
        # Calculate base values
        base_values = self.base_calculator.calculate(data)
        
        # Calculate SMA on base values
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

class EMAGenerator(FeatureGenerator):
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
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate EMA based on the specified base calculation."""
        # Calculate base values
        base_values = self.base_calculator.calculate(data)
        
        # Calculate EMA on base values
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

# WMA (Weighted Moving Average)
    
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

class WMAGenerator(FeatureGenerator):
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
        
        # Calculate WMA
        weights = np.arange(1, self.period + 1)
        wma = base_values.rolling(window=self.period).apply(
            lambda x: np.average(x, weights=weights)
        )
        
        return wma

# DEMA (Double Exponential Moving Average)
    
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

class DEMAGenerator(FeatureGenerator):
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
        
        # Calculate DEMA
        ema1 = base_values.ewm(span=self.period).mean()
        ema2 = ema1.ewm(span=self.period).mean()
        dema = 2 * ema1 - ema2
        
        return dema

# TEMA (Triple Exponential Moving Average)
    
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

class TEMAGenerator(FeatureGenerator):
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
        
        # Calculate TEMA
        ema1 = base_values.ewm(span=self.period).mean()
        ema2 = ema1.ewm(span=self.period).mean()
        ema3 = ema2.ewm(span=self.period).mean()
        tema = 3 * ema1 - 3 * ema2 + ema3
        
        return tema

# TRIMA (Triangular Moving Average)
    
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

class TRIMAGenerator(FeatureGenerator):
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
        
        # Calculate TRIMA
        half_period = self.period // 2
        trima = base_values.rolling(window=half_period).mean().rolling(window=half_period).mean()
        
        return trima

# MAMA (MESA Adaptive Moving Average)
    
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

class MAMAGenerator(FeatureGenerator):
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
        
        # Calculate MAMA (simplified version)
        mama = base_values.ewm(span=20).mean()
        
        return mama

# VWMA (Volume Weighted Moving Average)
    
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

class VWMAGenerator(FeatureGenerator):
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
        
        # Calculate VWMA
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

class KeltnerChannelsGenerator(FeatureGenerator):
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

