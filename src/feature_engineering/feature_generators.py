"""
Enhanced Feature Generators for Lookback Optimization

This module provides comprehensive feature generator functions for all
available technical indicators and features from the feature engineering
pipeline. Each generator function is optimized for hardware acceleration
and includes safe math operations.
"""

import pandas as pd
import numpy as np
from typing import Dict, Callable, Any, Optional, List, Tuple
import logging
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)

# Import hardware optimization tools
try:
    from src.utils.hardware.m1_gpu_utils import M1GPUManager
    from src.utils.hardware.m1_cpu_optimizer import M1CPUOptimizer
    from src.utils.hardware.m1_memory_optimizer import M1MemoryOptimizer
    HARDWARE_OPTIMIZATION_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Hardware optimization tools not available: {e}")
    HARDWARE_OPTIMIZATION_AVAILABLE = False

# Import safe math operations
try:
    from src.utils.math_validation import safe_divide, safe_log, safe_sqrt
    SAFE_MATH_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Safe math operations not available: {e}")
    SAFE_MATH_AVAILABLE = False

# Import feature selection tools
try:
    from src.utils.feature_selection.step08_optimized_methods import (
        fast_correlation_matrix, optimized_mutual_information, 
        vectorized_feature_stability, parallel_feature_importance
    )
    FEATURE_SELECTION_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Feature selection tools not available: {e}")
    FEATURE_SELECTION_AVAILABLE = False

class FeatureGenerators:
    """Enhanced collection of feature generator functions with hardware optimization."""
    
    def __init__(self):
        """Initialize feature generators with hardware optimization."""
        self.logger = logger.getChild('FeatureGenerators')
        
        # Initialize hardware optimization if available
        if HARDWARE_OPTIMIZATION_AVAILABLE:
            self.gpu_manager = M1GPUManager()
            self.cpu_optimizer = M1CPUOptimizer()
            self.memory_optimizer = M1MemoryOptimizer()
            self.logger.info("✅ Hardware optimization initialized")
        else:
            self.gpu_manager = None
            self.cpu_optimizer = None
            self.memory_optimizer = None
            self.logger.info("ℹ️ Hardware optimization not available")
    
    def _safe_divide(self, numerator: float, denominator: float, default: float = 0.0) -> float:
        """Safe division with fallback."""
        if SAFE_MATH_AVAILABLE:
            return safe_divide(numerator, denominator, default)
        else:
            return numerator / denominator if denominator != 0 else default
    
    def _safe_log(self, value: float, default: float = 0.0) -> float:
        """Safe logarithm with fallback."""
        if SAFE_MATH_AVAILABLE:
            return safe_log(value, default)
        else:
            return np.log(value) if value > 0 else default
    
    def _safe_sqrt(self, value: float, default: float = 0.0) -> float:
        """Safe square root with fallback."""
        if SAFE_MATH_AVAILABLE:
            return safe_sqrt(value, default)
        else:
            return np.sqrt(value) if value >= 0 else default

    @staticmethod
    def taker_buy_ratio_generator(data: pd.DataFrame, taker_base_col: str = 'taker_buy_base_asset_volume') -> pd.Series:
        """
        Generate taker buy ratio - percentage of volume from aggressive buyers.

        Args:
            data: DataFrame with volume and taker data
            taker_base_col: Column name for taker base volume

        Returns:
            Taker buy ratio (0-1) as pandas Series
        """
        if taker_base_col not in data.columns:
            return pd.Series([0.5] * len(data), index=data.index, name='taker_buy_ratio')

        total_volume = data['volume']
        taker_volume = data[taker_base_col]

        ratio = taker_volume / total_volume.replace(0, 1)
        ratio = ratio.fillna(0.5).clip(0, 1)

        return pd.Series(ratio, index=data.index, name='taker_buy_ratio')

    @staticmethod
    def market_aggression_generator(data: pd.DataFrame, taker_base_col: str = 'taker_buy_base_asset_volume') -> pd.Series:
        """
        Generate market aggression index - ratio of taker to maker volume.

        Args:
            data: DataFrame with volume and taker data
            taker_base_col: Column name for taker base volume

        Returns:
            Market aggression index as pandas Series
        """
        if taker_base_col not in data.columns:
            return pd.Series([1.0] * len(data), index=data.index, name='market_aggression')

        total_volume = data['volume']
        taker_volume = data[taker_base_col]
        maker_volume = total_volume - taker_volume

        aggression = taker_volume / maker_volume.replace(0, 1)
        aggression = aggression.fillna(1.0).clip(0, 10)  # Cap extreme values

        return pd.Series(aggression, index=data.index, name='market_aggression')

    @staticmethod
    def taker_price_impact_generator(data: pd.DataFrame,
                                   taker_base_col: str = 'taker_buy_base_asset_volume',
                                   taker_quote_col: str = 'taker_buy_quote_asset_volume') -> pd.Series:
        """
        Generate taker price impact - average price paid by aggressive buyers vs market price.

        Args:
            data: DataFrame with price and taker data
            taker_base_col: Column name for taker base volume
            taker_quote_col: Column name for taker quote volume

        Returns:
            Taker price impact as pandas Series
        """
        if taker_base_col not in data.columns or taker_quote_col not in data.columns:
            return pd.Series([0.0] * len(data), index=data.index, name='taker_price_impact')

        taker_avg_price = data[taker_quote_col] / data[taker_base_col].replace(0, 1)
        market_price = data['close']

        impact = (taker_avg_price - market_price) / market_price.replace(0, 1)
        impact = impact.fillna(0.0).clip(-1, 1)  # Cap extreme values

        return pd.Series(impact, index=data.index, name='taker_price_impact')

    @staticmethod
    def order_flow_imbalance_generator(data: pd.DataFrame, taker_base_col: str = 'taker_buy_base_asset_volume') -> pd.Series:
        """
        Generate order flow imbalance - net aggressive buying/selling pressure.

        Args:
            data: DataFrame with volume and taker data
            taker_base_col: Column name for taker base volume

        Returns:
            Order flow imbalance (-1 to 1) as pandas Series
        """
        if taker_base_col not in data.columns:
            return pd.Series([0.0] * len(data), index=data.index, name='order_flow_imbalance')

        total_volume = data['volume']
        taker_volume = data[taker_base_col]
        maker_volume = total_volume - taker_volume

        imbalance = (taker_volume - maker_volume) / total_volume.replace(0, 1)
        imbalance = imbalance.fillna(0.0).clip(-1, 1)

        return pd.Series(imbalance, index=data.index, name='order_flow_imbalance')

    @staticmethod
    def institutional_indicator_generator(data: pd.DataFrame,
                                        taker_base_col: str = 'taker_buy_base_asset_volume',
                                        taker_quote_col: str = 'taker_buy_quote_asset_volume') -> pd.Series:
        """
        Generate institutional vs retail trading indicator.

        High participation rate + stable pricing = institutional activity
        Low participation + volatile pricing = retail activity

        Args:
            data: DataFrame with price and taker data
            taker_base_col: Column name for taker base volume
            taker_quote_col: Column name for taker quote volume

        Returns:
            Institutional indicator (higher = more institutional) as pandas Series
        """
        if taker_base_col not in data.columns or taker_quote_col not in data.columns:
            return pd.Series([0.5] * len(data), index=data.index, name='institutional_indicator')

        # Participation rate
        participation = data[taker_base_col] / data['volume'].replace(0, 1)

        # Price stability (inverse of volatility)
        taker_avg_price = data[taker_quote_col] / data[taker_base_col].replace(0, 1)
        price_stability = 1 / (taker_avg_price.rolling(10).std() + 0.001)

        # Combined indicator
        indicator = participation * price_stability
        indicator = indicator.fillna(0.5).clip(0, 10)  # Cap extreme values

        return pd.Series(indicator, index=data.index, name='institutional_indicator')

    @staticmethod
    def taker_volume_momentum_generator(data: pd.DataFrame,
                                      taker_base_col: str = 'taker_buy_base_asset_volume',
                                      lookback: int = 5) -> pd.Series:
        """
        Generate taker volume momentum - rate of change in aggressive trading volume.

        Args:
            data: DataFrame with taker data
            taker_base_col: Column name for taker base volume
            lookback: Lookback period for momentum calculation

        Returns:
            Taker volume momentum as pandas Series
        """
        if taker_base_col not in data.columns:
            return pd.Series([0.0] * len(data), index=data.index, name=f'taker_momentum_{lookback}')

        momentum = data[taker_base_col].pct_change(lookback)
        momentum = momentum.fillna(0.0).clip(-5, 5)  # Cap extreme values

        return pd.Series(momentum, index=data.index, name=f'taker_momentum_{lookback}')

    @staticmethod
    def rsi_generator(data: pd.DataFrame, lookback: int, price_column: str = 'close') -> pd.Series:
        """
        Generate RSI (Relative Strength Index) indicator.
        
        Args:
            data: DataFrame with price data
            lookback: RSI period
            price_column: Column name for price data
            
        Returns:
            Series with RSI values
        """
        try:
            if price_column not in data.columns:
                raise ValueError(f"Price column '{price_column}' not found in data")
            
            prices = data[price_column]
            delta = prices.diff()
            
            # Separate gains and losses
            gains = delta.where(delta > 0, 0)
            losses = -delta.where(delta < 0, 0)
            
            # Calculate average gains and losses
            avg_gains = gains.rolling(window=lookback).mean()
            avg_losses = losses.rolling(window=lookback).mean()
            
            # Calculate RS and RSI
            rs = avg_gains / avg_losses
            rsi = 100 - (100 / (1 + rs))
            
            return rsi
            
        except Exception as e:
            logger.error(f"Error generating RSI: {e}")
            return pd.Series(index=data.index, dtype=float)
    
    @staticmethod
    def sma_generator(data: pd.DataFrame, lookback: int, price_column: str = 'close') -> pd.Series:
        """
        Generate SMA (Simple Moving Average) indicator.
        
        Args:
            data: DataFrame with price data
            lookback: SMA period
            price_column: Column name for price data
            
        Returns:
            Series with SMA values
        """
        try:
            if price_column not in data.columns:
                raise ValueError(f"Price column '{price_column}' not found in data")
            
            prices = data[price_column]
            sma = prices.rolling(window=lookback).mean()
            
            return sma
            
        except Exception as e:
            logger.error(f"Error generating SMA: {e}")
            return pd.Series(index=data.index, dtype=float)
    
    @staticmethod
    def ema_generator(data: pd.DataFrame, lookback: int, price_column: str = 'close') -> pd.Series:
        """
        Generate EMA (Exponential Moving Average) indicator.
        
        Args:
            data: DataFrame with price data
            lookback: EMA period (span)
            price_column: Column name for price data
            
        Returns:
            Series with EMA values
        """
        try:
            if price_column not in data.columns:
                raise ValueError(f"Price column '{price_column}' not found in data")
            
            prices = data[price_column]
            ema = prices.ewm(span=lookback).mean()
            
            return ema
            
        except Exception as e:
            logger.error(f"Error generating EMA: {e}")
            return pd.Series(index=data.index, dtype=float)
    
    @staticmethod
    def bollinger_bands_generator(data: pd.DataFrame, lookback: int, price_column: str = 'close', 
                                 std_dev: float = 2.0) -> pd.Series:
        """
        Generate Bollinger Bands indicator.
        
        Args:
            data: DataFrame with price data
            lookback: Period for moving average
            price_column: Column name for price data
            std_dev: Standard deviation multiplier
            
        Returns:
            Series with Bollinger Band position (0-1 scale)
        """
        try:
            if price_column not in data.columns:
                raise ValueError(f"Price column '{price_column}' not found in data")
            
            prices = data[price_column]
            sma = prices.rolling(window=lookback).mean()
            std = prices.rolling(window=lookback).std()
            
            upper_band = sma + (std * std_dev)
            lower_band = sma - (std * std_dev)
            
            # Calculate position within bands (0-1 scale)
            bb_position = (prices - lower_band) / (upper_band - lower_band)
            
            return bb_position
            
        except Exception as e:
            logger.error(f"Error generating Bollinger Bands: {e}")
            return pd.Series(index=data.index, dtype=float)
    
    @staticmethod
    def macd_generator(data: pd.DataFrame, lookback: int, price_column: str = 'close',
                      fast_period: int = 12, slow_period: int = 26) -> pd.Series:
        """
        Generate MACD (Moving Average Convergence Divergence) indicator.
        
        Args:
            data: DataFrame with price data
            lookback: Signal line period
            price_column: Column name for price data
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            
        Returns:
            Series with MACD signal line
        """
        try:
            if price_column not in data.columns:
                raise ValueError(f"Price column '{price_column}' not found in data")
            
            prices = data[price_column]
            
            # Calculate EMAs
            ema_fast = prices.ewm(span=fast_period).mean()
            ema_slow = prices.ewm(span=slow_period).mean()
            
            # Calculate MACD line
            macd_line = ema_fast - ema_slow
            
            # Calculate signal line
            signal_line = macd_line.ewm(span=lookback).mean()
            
            return signal_line
            
        except Exception as e:
            logger.error(f"Error generating MACD: {e}")
            return pd.Series(index=data.index, dtype=float)
    
    @staticmethod
    def stochastic_generator(data: pd.DataFrame, lookback: int, k_period: int = 14,
                           d_period: int = 3) -> pd.Series:
        """
        Generate Stochastic Oscillator indicator.
        
        Args:
            data: DataFrame with OHLC data
            lookback: Period for %K calculation
            k_period: %K period
            d_period: %D period
            
        Returns:
            Series with Stochastic %D values
        """
        try:
            required_columns = ['high', 'low', 'close']
            if not all(col in data.columns for col in required_columns):
                raise ValueError(f"Required columns {required_columns} not found in data")
            
            high = data['high']
            low = data['low']
            close = data['close']
            
            # Calculate %K
            lowest_low = low.rolling(window=k_period).min()
            highest_high = high.rolling(window=k_period).max()
            k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
            
            # Calculate %D (smoothed %K)
            d_percent = k_percent.rolling(window=d_period).mean()
            
            return d_percent
            
        except Exception as e:
            logger.error(f"Error generating Stochastic: {e}")
            return pd.Series(index=data.index, dtype=float)
    
    @staticmethod
    def atr_generator(data: pd.DataFrame, lookback: int) -> pd.Series:
        """
        Generate ATR (Average True Range) indicator.
        
        Args:
            data: DataFrame with OHLC data
            lookback: ATR period
            
        Returns:
            Series with ATR values
        """
        try:
            required_columns = ['high', 'low', 'close']
            if not all(col in data.columns for col in required_columns):
                raise ValueError(f"Required columns {required_columns} not found in data")
            
            high = data['high']
            low = data['low']
            close = data['close']
            
            # Calculate True Range
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            # Calculate ATR
            atr = true_range.rolling(window=lookback).mean()
            
            return atr
            
        except Exception as e:
            logger.error(f"Error generating ATR: {e}")
            return pd.Series(index=data.index, dtype=float)
    
    @staticmethod
    def volume_sma_generator(data: pd.DataFrame, lookback: int, volume_column: str = 'volume') -> pd.Series:
        """
        Generate Volume SMA indicator.
        
        Args:
            data: DataFrame with volume data
            lookback: SMA period
            volume_column: Column name for volume data
            
        Returns:
            Series with Volume SMA values
        """
        try:
            if volume_column not in data.columns:
                raise ValueError(f"Volume column '{volume_column}' not found in data")
            
            volume = data[volume_column]
            volume_sma = volume.rolling(window=lookback).mean()
            
            return volume_sma
            
        except Exception as e:
            logger.error(f"Error generating Volume SMA: {e}")
            return pd.Series(index=data.index, dtype=float)
    
    @staticmethod
    def price_momentum_generator(data: pd.DataFrame, lookback: int, price_column: str = 'close') -> pd.Series:
        """
        Generate Price Momentum indicator.
        
        Args:
            data: DataFrame with price data
            lookback: Momentum period
            price_column: Column name for price data
            
        Returns:
            Series with price momentum values
        """
        try:
            if price_column not in data.columns:
                raise ValueError(f"Price column '{price_column}' not found in data")
            
            prices = data[price_column]
            momentum = prices.pct_change(lookback)
            
            return momentum
            
        except Exception as e:
            logger.error(f"Error generating Price Momentum: {e}")
            return pd.Series(index=data.index, dtype=float)
    
    @staticmethod
    def volatility_generator(data: pd.DataFrame, lookback: int, price_column: str = 'close') -> pd.Series:
        """
        Generate Volatility indicator (rolling standard deviation of returns).
        
        Args:
            data: DataFrame with price data
            lookback: Volatility period
            price_column: Column name for price data
            
        Returns:
            Series with volatility values
        """
        try:
            if price_column not in data.columns:
                raise ValueError(f"Price column '{price_column}' not found in data")
            
            prices = data[price_column]
            returns = prices.pct_change()
            volatility = returns.rolling(window=lookback).std()
            
            return volatility
            
        except Exception as e:
            logger.error(f"Error generating Volatility: {e}")
            return pd.Series(index=data.index, dtype=float)

    @staticmethod
    def body_size_generator(data: pd.DataFrame) -> pd.Series:
        """
        Generate Body Size feature (absolute difference between open and close).

        Args:
            data: DataFrame with OHLC data

        Returns:
            Series with body size values
        """
        try:
            required_cols = ['open', 'close']
            if not all(col in data.columns for col in required_cols):
                raise ValueError(f"Required columns {required_cols} not found in data")

            body_size = np.abs(data['close'] - data['open'])
            return pd.Series(body_size, index=data.index, name='body_size')

        except Exception as e:
            logger.error(f"Error generating Body Size: {e}")
            return pd.Series(index=data.index, dtype=float)

    @staticmethod
    def body_size_pct_generator(data: pd.DataFrame) -> pd.Series:
        """
        Generate Body Size Percentage feature (body size relative to open price).

        Args:
            data: DataFrame with OHLC data

        Returns:
            Series with body size percentage values
        """
        try:
            required_cols = ['open', 'close']
            if not all(col in data.columns for col in required_cols):
                raise ValueError(f"Required columns {required_cols} not found in data")

            body_size = np.abs(data['close'] - data['open'])
            body_size_pct = (body_size / data['open']) * 100  # Convert to percentage
            return pd.Series(body_size_pct, index=data.index, name='body_size_pct')

        except Exception as e:
            logger.error(f"Error generating Body Size Percentage: {e}")
            return pd.Series(index=data.index, dtype=float)

    @staticmethod
    def body_to_range_ratio_generator(data: pd.DataFrame) -> pd.Series:
        """
        Generate Body to Range Ratio feature (body size relative to total high-low range).

        Args:
            data: DataFrame with OHLC data

        Returns:
            Series with body to range ratio values
        """
        try:
            required_cols = ['open', 'high', 'low', 'close']
            if not all(col in data.columns for col in required_cols):
                raise ValueError(f"Required columns {required_cols} not found in data")

            body_size = np.abs(data['close'] - data['open'])
            total_range = data['high'] - data['low']
            body_to_range_ratio = body_size / total_range.replace(0, 1)  # Avoid division by zero
            return pd.Series(body_to_range_ratio, index=data.index, name='body_to_range_ratio')

        except Exception as e:
            logger.error(f"Error generating Body to Range Ratio: {e}")
            return pd.Series(index=data.index, dtype=float)

    @staticmethod
    def upper_wick_generator(data: pd.DataFrame) -> pd.Series:
        """
        Generate Upper Wick feature (distance from high to the higher of open/close).

        Args:
            data: DataFrame with OHLC data

        Returns:
            Series with upper wick values
        """
        try:
            required_cols = ['open', 'high', 'close']
            if not all(col in data.columns for col in required_cols):
                raise ValueError(f"Required columns {required_cols} not found in data")

            upper_wick = data['high'] - np.maximum(data['open'], data['close'])
            return pd.Series(upper_wick, index=data.index, name='upper_wick')

        except Exception as e:
            logger.error(f"Error generating Upper Wick: {e}")
            return pd.Series(index=data.index, dtype=float)

    @staticmethod
    def lower_wick_generator(data: pd.DataFrame) -> pd.Series:
        """
        Generate Lower Wick feature (distance from low to the lower of open/close).

        Args:
            data: DataFrame with OHLC data

        Returns:
            Series with lower wick values
        """
        try:
            required_cols = ['open', 'low', 'close']
            if not all(col in data.columns for col in required_cols):
                raise ValueError(f"Required columns {required_cols} not found in data")

            lower_wick = np.minimum(data['open'], data['close']) - data['low']
            return pd.Series(lower_wick, index=data.index, name='lower_wick')

        except Exception as e:
            logger.error(f"Error generating Lower Wick: {e}")
            return pd.Series(index=data.index, dtype=float)

    @staticmethod
    def body_direction_generator(data: pd.DataFrame) -> pd.Series:
        """
        Generate Body Direction feature (sign of price movement: +1 up, -1 down, 0 no change).

        Args:
            data: DataFrame with OHLC data

        Returns:
            Series with body direction values
        """
        try:
            required_cols = ['open', 'close']
            if not all(col in data.columns for col in required_cols):
                raise ValueError(f"Required columns {required_cols} not found in data")

            body_direction = np.sign(data['close'] - data['open'])
            return pd.Series(body_direction, index=data.index, name='body_direction')

        except Exception as e:
            logger.error(f"Error generating Body Direction: {e}")
            return pd.Series(index=data.index, dtype=float)

    @staticmethod
    def body_strength_generator(data: pd.DataFrame) -> pd.Series:
        """
        Generate Body Strength feature (signed body size: positive for up, negative for down).

        Args:
            data: DataFrame with OHLC data

        Returns:
            Series with body strength values
        """
        try:
            required_cols = ['open', 'close']
            if not all(col in data.columns for col in required_cols):
                raise ValueError(f"Required columns {required_cols} not found in data")

            body_size = np.abs(data['close'] - data['open'])
            body_direction = np.sign(data['close'] - data['open'])
            body_strength = body_size * body_direction
            return pd.Series(body_strength, index=data.index, name='body_strength')

        except Exception as e:
            logger.error(f"Error generating Body Strength: {e}")
            return pd.Series(index=data.index, dtype=float)

# Registry of available feature generators
FEATURE_GENERATORS: Dict[str, Callable] = {
    'rsi': FeatureGenerators.rsi_generator,
    'sma': FeatureGenerators.sma_generator,
    'ema': FeatureGenerators.ema_generator,
    'bollinger_bands': FeatureGenerators.bollinger_bands_generator,
    'macd': FeatureGenerators.macd_generator,
    'stochastic': FeatureGenerators.stochastic_generator,
    'atr': FeatureGenerators.atr_generator,
    'volume_sma': FeatureGenerators.volume_sma_generator,
    'price_momentum': FeatureGenerators.price_momentum_generator,
    'volatility': FeatureGenerators.volatility_generator,
    # Candlestick body size features
    'body_size': FeatureGenerators.body_size_generator,
    'body_size_pct': FeatureGenerators.body_size_pct_generator,
    'body_to_range_ratio': FeatureGenerators.body_to_range_ratio_generator,
    'upper_wick': FeatureGenerators.upper_wick_generator,
    'lower_wick': FeatureGenerators.lower_wick_generator,
    'body_direction': FeatureGenerators.body_direction_generator,
    'body_strength': FeatureGenerators.body_strength_generator
}

def get_feature_generator(feature_name: str) -> Optional[Callable]:
    """
    Get a feature generator function by name.
    
    Args:
        feature_name: Name of the feature generator
        
    Returns:
        Feature generator function or None if not found
    """
    return FEATURE_GENERATORS.get(feature_name.lower())

def list_available_generators() -> list:
    """
    List all available feature generators.
    
    Returns:
        List of available feature generator names
    """
    return list(FEATURE_GENERATORS.keys())

def create_feature_generator_config(feature_name: str, **kwargs) -> Dict[str, Any]:
    """
    Create a configuration for a feature generator.
    
    Args:
        feature_name: Name of the feature
        **kwargs: Additional configuration parameters
        
    Returns:
        Configuration dictionary
    """
    generator = get_feature_generator(feature_name)
    if not generator:
        raise ValueError(f"Unknown feature generator: {feature_name}")
    
    config = {
        'generator': generator,
        'feature_name': feature_name,
        **kwargs
    }
    
    return config

# Convenience functions for common feature configurations
def create_rsi_config(**kwargs) -> Dict[str, Any]:
    """Create RSI feature configuration."""
    return create_feature_generator_config('rsi', **kwargs)

def create_sma_config(**kwargs) -> Dict[str, Any]:
    """Create SMA feature configuration."""
    return create_feature_generator_config('sma', **kwargs)

def create_ema_config(**kwargs) -> Dict[str, Any]:
    """Create EMA feature configuration."""
    return create_feature_generator_config('ema', **kwargs)

def create_bollinger_bands_config(**kwargs) -> Dict[str, Any]:
    """Create Bollinger Bands feature configuration."""
    return create_feature_generator_config('bollinger_bands', **kwargs)

def create_macd_config(**kwargs) -> Dict[str, Any]:
    """Create MACD feature configuration."""
    return create_feature_generator_config('macd', **kwargs)