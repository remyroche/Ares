"""
Momentum Feature Generator

This module provides feature generators for momentum-based indicators,
including RSI, MACD, Stochastic, and other momentum oscillators.
Supports different base calculations: price returns, returns-based VWAP, etc.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union, Literal

from ..core.feature_generator import (
    FeatureGenerator, 
    FeatureConfig, 
    FeatureCategory,
    VectorizedFeatureGenerator
)
from ..base_calculations import (
    BaseCalculator,
    BaseCalculationType,
    BaseCalculationConfig,
    create_base_calculator
)

class MomentumFeatureGenerator(VectorizedFeatureGenerator):
    """
    Feature generator for momentum-based features.
    
    This generator creates various momentum indicators including:
    - RSI (Relative Strength Index)
    - MACD (Moving Average Convergence Divergence)
    - Stochastic Oscillator
    - Williams %R
    - Rate of Change (ROC)
    - Momentum
    """
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        """
        Initialize the momentum feature generator.
        
        Args:
            config: Feature configuration (uses default if None)
        """
        if config is None:
            config = self._create_default_config()
        
        super().__init__(config, enable_matrix_ops=True)
    
    @classmethod
    def _create_default_config(cls) -> FeatureConfig:
        """Create default configuration for momentum features."""
        return FeatureConfig(
            name="momentum_features",
            category=FeatureCategory.MOMENTUM,
            description="Comprehensive momentum-based features including RSI, MACD, and other momentum indicators",
            required_columns=["close"],
            optional_columns=["high", "low", "volume"],
            default_lookback=14,
            min_lookback=2,
            max_lookback=50,
            parameters={
                "rsi_periods": [14, 21],
                "macd_fast": 12,
                "macd_slow": 26,
                "macd_signal": 9,
                "stochastic_k": 14,
                "stochastic_d": 3,
                "williams_period": 14,
                "roc_periods": [10, 20]
            },
            matrix_optimized=True,
            gpu_accelerated=False
        )
    
    @classmethod
    def create_default(cls) -> 'MomentumFeatureGenerator':
        """Create a default momentum feature generator."""
        return cls()
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        Generate momentum features.
        
        Args:
            data: Input data with OHLCV columns
            **kwargs: Additional parameters
            
        Returns:
            Combined momentum features (placeholder - actual implementation would return multiple features)
        """
        # This is a simplified implementation that returns a single feature
        # In practice, this would generate multiple momentum features
        
        close_prices = data['close'].values
        
        # Generate RSI as the main feature
        rsi = self._calculate_rsi(close_prices, period=14)
        
        return pd.Series(rsi, index=data.index, name='rsi_14')
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> np.ndarray:
        """Calculate RSI (Relative Strength Index)."""
        if len(prices) < period + 1:
            return np.full(len(prices), np.nan)
        
        # Calculate price changes
        price_changes = np.diff(prices)
        
        # Separate gains and losses
        gains = np.where(price_changes > 0, price_changes, 0)
        losses = np.where(price_changes < 0, -price_changes, 0)
        
        # Calculate initial average gain and loss
        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])
        
        rsi = np.full(len(prices), np.nan)
        
        # Calculate RSI for the first valid period
        if avg_loss != 0:
            rs = avg_gain / avg_loss
            rsi[period] = 100 - (100 / (1 + rs))
        
        # Calculate RSI for remaining periods using Wilder's smoothing
        for i in range(period + 1, len(prices)):
            if i - period - 1 < len(gains):
                gain = gains[i - period - 1]
                loss = losses[i - period - 1]
                
                avg_gain = (avg_gain * (period - 1) + gain) / period
                avg_loss = (avg_loss * (period - 1) + loss) / period
                
                if avg_loss != 0:
                    rs = avg_gain / avg_loss
                    rsi[i] = 100 - (100 / (1 + rs))
                else:
                    rsi[i] = 100
        
        return rsi
    
    def _calculate_macd(self, prices: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, np.ndarray]:
        """Calculate MACD (Moving Average Convergence Divergence)."""
        if len(prices) < slow:
            return {
                'macd': np.full(len(prices), np.nan),
                'signal': np.full(len(prices), np.nan),
                'histogram': np.full(len(prices), np.nan)
            }
        
        # Calculate EMAs
        ema_fast = self._calculate_ema(prices, fast)
        ema_slow = self._calculate_ema(prices, slow)
        
        # Calculate MACD line
        macd = ema_fast - ema_slow
        
        # Calculate signal line (EMA of MACD)
        signal_line = self._calculate_ema(macd, signal)
        
        # Calculate histogram
        histogram = macd - signal_line
        
        return {
            'macd': macd,
            'signal': signal_line,
            'histogram': histogram
        }
    
    def _calculate_ema(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Calculate Exponential Moving Average."""
        if len(prices) < period:
            return np.full(len(prices), np.nan)
        
        alpha = 2.0 / (period + 1)
        ema = np.full(len(prices), np.nan)
        
        # Initialize with SMA
        ema[period - 1] = np.mean(prices[:period])
        
        # Calculate EMA for remaining periods
        for i in range(period, len(prices)):
            ema[i] = alpha * prices[i] + (1 - alpha) * ema[i - 1]
        
        return ema
    
    def _generate_feature_with_lookback(self, data: pd.DataFrame, lookback: int, **kwargs) -> pd.Series:
        """
        Generate momentum features with specific lookback period.
        
        Args:
            data: Input data
            lookback: Lookback period
            **kwargs: Additional parameters
            
        Returns:
            Momentum features with specified lookback
        """
        close_prices = data['close'].values
        rsi = self._calculate_rsi(close_prices, period=lookback)
        
        return pd.Series(rsi, index=data.index, name=f'rsi_{lookback}')

class RSIGenerator(FeatureGenerator):
    """Generator for RSI (Relative Strength Index) with different base calculations."""
    
    def __init__(self, 
                 period: int = 14,
                 base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS,
                 **base_kwargs):
        """
        Initialize RSI generator.
        
        Args:
            period: RSI period
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
            name=f"rsi_{period}_{base_calculation.value}",
            category=FeatureCategory.MOMENTUM,
            description=f"RSI over {period} periods based on {base_calculation.value}",
            required_columns=required_columns,
            default_lookback=period,
            min_lookback=2,
            max_lookback=50,
            parameters={
                'period': period,
                'base_calculation': base_calculation.value,
                **base_kwargs
            }
        )
        super().__init__(config)
        self.period = period
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate RSI based on the specified base calculation."""
        # Calculate base values (returns, returns_vwap, etc.)
        base_values = self.base_calculator.calculate(data)
        
        # Calculate RSI based on base values
        if self.base_calculation == BaseCalculationType.PRICE_RETURNS:
            # For price returns, RSI is calculated on the returns themselves
            return self._calculate_rsi_from_returns(base_values)
        elif self.base_calculation == BaseCalculationType.RETURNS_VWAP:
            # For returns-based VWAP, RSI is calculated on the VWAP returns
            return self._calculate_rsi_from_returns(base_values)
        else:
            # For other base calculations, calculate RSI on the values
            return self._calculate_rsi_from_values(base_values)
    
    def _calculate_rsi_from_returns(self, returns: pd.Series) -> pd.Series:
        """Calculate RSI from returns."""
        # Separate gains and losses
        gains = returns.where(returns > 0, 0)
        losses = -returns.where(returns < 0, 0)
        
        # Calculate average gains and losses using Wilder's smoothing
        avg_gains = gains.ewm(alpha=1/self.period, adjust=False).mean()
        avg_losses = losses.ewm(alpha=1/self.period, adjust=False).mean()
        
        # Calculate RS and RSI
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_rsi_from_values(self, values: pd.Series) -> pd.Series:
        """Calculate RSI from values (like price levels)."""
        # Calculate price changes
        delta = values.diff()
        
        # Separate gains and losses
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)
        
        # Calculate average gains and losses using Wilder's smoothing
        avg_gains = gains.ewm(alpha=1/self.period, adjust=False).mean()
        avg_losses = losses.ewm(alpha=1/self.period, adjust=False).mean()
        
        # Calculate RS and RSI
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))
        
        return rsi

class MACDGenerator(FeatureGenerator):
    """Generator for MACD (Moving Average Convergence Divergence) with different base calculations."""
    
    def __init__(self, 
                 fast: int = 12, 
                 slow: int = 26, 
                 signal: int = 9,
                 base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_LEVELS,
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
            description=f"MACD with fast={fast}, slow={slow}, signal={signal} based on {base_calculation.value}",
            required_columns=required_columns,
            default_lookback=slow,
            min_lookback=slow,
            max_lookback=slow,
            parameters={
                'fast': fast,
                'slow': slow,
                'signal': signal,
                'base_calculation': base_calculation.value,
                **base_kwargs
            }
        )
        super().__init__(config)
        self.fast = fast
        self.slow = slow
        self.signal = signal
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate MACD line based on the specified base calculation."""
        # Calculate base values
        base_values = self.base_calculator.calculate(data)
        
        # Calculate EMAs on base values
        ema_fast = base_values.ewm(span=self.fast).mean()
        ema_slow = base_values.ewm(span=self.slow).mean()
        
        # Calculate MACD line
        macd = ema_fast - ema_slow
        
        return macd

class MACDSignalGenerator(FeatureGenerator):
    """Generator for MACD signal line."""
    
    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        """Initialize MACD signal generator."""
        config = FeatureConfig(
            name=f"macd_signal_{fast}_{slow}_{signal}",
            category=FeatureCategory.MOMENTUM,
            description=f"MACD signal line with fast={fast}, slow={slow}, signal={signal}",
            required_columns=["close"],
            default_lookback=slow,
            min_lookback=slow,
            max_lookback=slow
        )
        super().__init__(config)
        self.fast = fast
        self.slow = slow
        self.signal = signal
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate MACD signal line."""
        close_prices = data['close']
        
        # Calculate EMAs
        ema_fast = close_prices.ewm(span=self.fast).mean()
        ema_slow = close_prices.ewm(span=self.slow).mean()
        
        # Calculate MACD line
        macd = ema_fast - ema_slow
        
        # Calculate signal line (EMA of MACD)
        signal = macd.ewm(span=self.signal).mean()
        
        return signal

class MACDHistogramGenerator(FeatureGenerator):
    """Generator for MACD histogram."""
    
    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        """Initialize MACD histogram generator."""
        config = FeatureConfig(
            name=f"macd_histogram_{fast}_{slow}_{signal}",
            category=FeatureCategory.MOMENTUM,
            description=f"MACD histogram with fast={fast}, slow={slow}, signal={signal}",
            required_columns=["close"],
            default_lookback=slow,
            min_lookback=slow,
            max_lookback=slow
        )
        super().__init__(config)
        self.fast = fast
        self.slow = slow
        self.signal = signal
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate MACD histogram."""
        close_prices = data['close']
        
        # Calculate EMAs
        ema_fast = close_prices.ewm(span=self.fast).mean()
        ema_slow = close_prices.ewm(span=self.slow).mean()
        
        # Calculate MACD line
        macd = ema_fast - ema_slow
        
        # Calculate signal line (EMA of MACD)
        signal = macd.ewm(span=self.signal).mean()
        
        # Calculate histogram
        histogram = macd - signal
        
        return histogram

class StochasticGenerator(FeatureGenerator):
    """Generator for Stochastic Oscillator with different base calculations."""
    
    def __init__(self, 
                 k_period: int = 14, 
                 d_period: int = 3,
                 base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_LEVELS,
                 **base_kwargs):
        """
        Initialize Stochastic generator.
        
        Args:
            k_period: %K period
            d_period: %D period
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
            name=f"stochastic_k_{k_period}_d_{d_period}_{base_calculation.value}",
            category=FeatureCategory.MOMENTUM,
            description=f"Stochastic Oscillator with K={k_period}, D={d_period} based on {base_calculation.value}",
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
        super().__init__(config)
        self.k_period = k_period
        self.d_period = d_period
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate Stochastic based on the specified base calculation."""
        if self.base_calculation == BaseCalculationType.PRICE_LEVELS:
            # Traditional Stochastic calculation on price levels
            high = data['high']
            low = data['low']
            close = data['close']
            
            # Calculate lowest low and highest high over K period
            lowest_low = low.rolling(window=self.k_period).min()
            highest_high = high.rolling(window=self.k_period).max()
            
            # Calculate %K
            k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
            
            return k_percent
        else:
            # For other base calculations, calculate Stochastic on the base values
            base_values = self.base_calculator.calculate(data)
            
            # Calculate rolling min/max on base values
            lowest_low = base_values.rolling(window=self.k_period).min()
            highest_high = base_values.rolling(window=self.k_period).max()
            
            # Calculate %K
            k_percent = 100 * ((base_values - lowest_low) / (highest_high - lowest_low))
            
            return k_percent

class WilliamsRGenerator(FeatureGenerator):
    """Generator for Williams %R with different base calculations."""
    
    def __init__(self, 
                 period: int = 14,
                 base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_LEVELS,
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
        super().__init__(config)
        self.period = period
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate Williams %R based on the specified base calculation."""
        if self.base_calculation == BaseCalculationType.PRICE_LEVELS:
            # Traditional Williams %R calculation on price levels
            high = data['high']
            low = data['low']
            close = data['close']
            
            # Calculate highest high and lowest low over period
            highest_high = high.rolling(window=self.period).max()
            lowest_low = low.rolling(window=self.period).min()
            
            # Calculate Williams %R
            williams_r = -100 * ((highest_high - close) / (highest_high - lowest_low))
            
            return williams_r
        else:
            # For other base calculations, calculate Williams %R on the base values
            base_values = self.base_calculator.calculate(data)
            
            # Calculate rolling min/max on base values
            highest_high = base_values.rolling(window=self.period).max()
            lowest_low = base_values.rolling(window=self.period).min()
            
            # Calculate Williams %R
            williams_r = -100 * ((highest_high - base_values) / (highest_high - lowest_low))
            
            return williams_r

class ROCGenerator(FeatureGenerator):
    """Generator for Rate of Change (ROC) with different base calculations."""
    
    def __init__(self, 
                 period: int = 10,
                 base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_LEVELS,
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
        super().__init__(config)
        self.period = period
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate ROC based on the specified base calculation."""
        if self.base_calculation == BaseCalculationType.PRICE_LEVELS:
            # Traditional ROC calculation on price levels
            close_prices = data['close']
            
            # Calculate ROC
            roc = ((close_prices - close_prices.shift(self.period)) / close_prices.shift(self.period)) * 100
            
            return roc
        else:
            # For other base calculations, calculate ROC on the base values
            base_values = self.base_calculator.calculate(data)
            
            # Calculate ROC on base values
            roc = ((base_values - base_values.shift(self.period)) / base_values.shift(self.period)) * 100
            return roc

class MomentumGenerator(FeatureGenerator):
    """Generator for Momentum indicator with different base calculations."""
    
    def __init__(self, 
                 period: int = 10,
                 base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_LEVELS,
                 **base_kwargs):
        """
        Initialize Momentum generator.
        
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
            description=f"Momentum over {period} periods based on {base_calculation.value}",
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
        super().__init__(config)
        self.period = period
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate Momentum based on the specified base calculation."""
        if self.base_calculation == BaseCalculationType.PRICE_LEVELS:
            # Traditional Momentum calculation on price levels
            close_prices = data['close']
            
            # Calculate Momentum
            momentum = close_prices - close_prices.shift(self.period)
            
            return momentum
        else:
            # For other base calculations, calculate Momentum on the base values
            base_values = self.base_calculator.calculate(data)
            
            # Calculate Momentum on base values
            momentum = base_values - base_values.shift(self.period)
            return momentum

# Factory functions for creating momentum generators
def create_momentum_generators(periods: Dict[str, List[int]] = None) -> List[FeatureGenerator]:
    """
    Create a set of momentum feature generators.
    
    Args:
        periods: Dictionary mapping indicator types to lists of periods
        
    Returns:
        List of momentum feature generators
    """
    if periods is None:
        periods = {
            'rsi': [14, 21],
            'macd': [(12, 26, 9)],
            'stochastic': [(14, 3)],
            'williams_r': [14],
            'roc': [10, 20],
            'momentum': [10, 20]
        }
    
    generators = []
    
    # RSI generators
    for period in periods.get('rsi', [14]):
        generators.append(RSIGenerator(period))
    
    # MACD generators
    for fast, slow, signal in periods.get('macd', [(12, 26, 9)]):
        generators.extend([
            MACDGenerator(fast, slow, signal),
            MACDSignalGenerator(fast, slow, signal),
            MACDHistogramGenerator(fast, slow, signal)
        ])
    
    # Stochastic generators
    for k_period, d_period in periods.get('stochastic', [(14, 3)]):
        generators.append(StochasticGenerator(k_period, d_period))
    
    # Williams %R generators
    for period in periods.get('williams_r', [14]):
        generators.append(WilliamsRGenerator(period))
    
    # ROC generators
    for period in periods.get('roc', [10, 20]):
        generators.append(ROCGenerator(period))
    
    # Momentum generators
    for period in periods.get('momentum', [10, 20]):
        generators.append(MomentumGenerator(period))
    
    return generators

def create_default_momentum_generators() -> List[FeatureGenerator]:
    """Create default momentum feature generators."""
    return create_momentum_generators()