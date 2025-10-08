"""
Momentum Feature Generator

This module provides feature generators for momentum-based indicators,
including RSI, MACD, Stochastic, and other momentum oscillators.
Wires up existing momentum features from legacy and entropy modules.
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
        
        momentum = prices - np.roll(prices, period)
        return momentum

# Analyst Features - Cross-timeframe momentum generators
    
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

        """Generate 5-minute momentum feature."""
        returns = data['close'].pct_change()
        momentum = returns.rolling(self.lookback).mean()
        return momentum

    
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

        """Generate 15-minute momentum feature."""
        returns = data['close'].pct_change()
        momentum = returns.rolling(self.lookback).mean()
        return momentum

    
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
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate RSI based on the specified base calculation."""
        if self.base_calculation == BaseCalculationType.PRICE_LEVELS:
            close = data['close']
            
            # Traditional RSI calculation
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=self.period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=self.period).mean()
            rs = gain / loss.replace(0, 1)
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
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate MACD based on the specified base calculation."""
        # Calculate base values
        base_values = self.base_calculator.calculate(data)
        
        # Calculate MACD
        ema_fast = base_values.ewm(span=self.fast).mean()
        ema_slow = base_values.ewm(span=self.slow).mean()
        macd = ema_fast - ema_slow
        
        return macd

    
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
            
            # Traditional Stochastic calculation
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
            
            return k_percent

    
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
            
            # Williams %R calculation
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
            
            return williams_r

    
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
        
        # Calculate momentum
        momentum = base_values - base_values.shift(self.period)
        
        return momentum

    
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
        momentum_volatility = momentum_ratio.rolling(window=10).std()
        momentum_trend = momentum_ratio.rolling(window=5).mean()
        
        # Regime strength: higher when momentum is consistent and trending
        regime_strength = np.abs(momentum_trend) / (momentum_volatility + 1e-8)
        
        # Enhanced regime indicator combining momentum and persistence
        enhanced_momentum = momentum_ratio * (1 + regime_strength)
        
        # Add regime transition detection
        momentum_change = momentum_ratio.diff().abs()
        regime_transition = momentum_change.rolling(window=3).mean()
        
        # Combine momentum with regime transition awareness
        regime_aware_momentum = enhanced_momentum * (1 - regime_transition)

        return regime_aware_momentum


    
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
        volume_momentum = (volume - volume_ma) / volume_ma

        return volume_momentum


def create_default_momentum_generators() -> List[FeatureGenerator]:
    """Create default momentum generators including legacy and entropy features."""
    generators = []
    
    # Add new momentum generators
    generators.extend(create_momentum_generators())
    
    # Add legacy momentum generators
    generators.extend([
        LegacyRSIGenerator(14),
        LegacyMACDGenerator(12, 26, 9),
        LegacyStochasticGenerator(14, 3),
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
        generators.append(LegacyRSIGenerator(period))
    
    # Additional MACD configurations
    macd_configs = [(8, 21, 5), (5, 35, 5)]
    for fast, slow, signal in macd_configs:
        generators.append(MACDGenerator(fast, slow, signal))
        generators.append(LegacyMACDGenerator(fast, slow, signal))
    
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

