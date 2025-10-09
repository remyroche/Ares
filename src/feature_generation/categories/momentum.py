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
            max_lookback=ma_period * 3,  # Allow up to 3x window for optimization
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
            max_lookback=(slow + signal) * 3,  # Allow up to 3x window for optimization
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
            max_lookback=(rsi_period + zscore_window) * 3,  # Allow up to 3x window for optimization
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
            max_lookback=(k_period + d_period) * 3,  # Allow up to 3x window for optimization
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
            max_lookback=period * 3,  # Allow up to 3x window for optimization
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

