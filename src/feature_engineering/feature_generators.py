"""
Feature Generators for Lookback Optimization

This module provides standardized feature generator functions for various
technical indicators that can be used in feature lookback optimization.
Each generator function takes a DataFrame and lookback period as input
and returns a pandas Series with the calculated indicator.
"""

import pandas as pd
import numpy as np
from typing import Dict, Callable, Any, Optional
import logging

logger = logging.getLogger(__name__)

class FeatureGenerators:
    """Collection of feature generator functions for optimization."""
    
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
    'volatility': FeatureGenerators.volatility_generator
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