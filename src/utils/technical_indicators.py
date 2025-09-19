#!/usr/bin/env python3
"""
Unified Technical Indicators Module

This module consolidates all technical indicator calculations that were previously
duplicated across 45+ files. It provides a single, well-tested implementation
of each indicator with consistent interfaces.

Replaces duplicate implementations from:
- analyst/unified_regime_classifier.py
- components/modular_analyst.py  
- market_analysis/hmm_clustering/enhanced_hmm_clustering.py
- training/steps/market_analysis/hmm_clustering/step03_hmm_regime_discovery.py
- And 40+ other files
"""

import numpy as np
import pandas as pd
from typing import Tuple, Union, Optional
import warnings

# Import math validation utilities if available
try:
    from .math_validation import safe_divide, safe_log, safe_sqrt, safe_nan_to_num
    MATH_VALIDATION_AVAILABLE = True
except ImportError:
    MATH_VALIDATION_AVAILABLE = False


class TechnicalIndicators:
    """
    Unified technical indicators with consistent interfaces and robust error handling.
    
    All methods are static and can be used independently. This class consolidates
    functionality that was previously scattered across dozens of files.
    """
    
    @staticmethod
    def calculate_rsi(prices: Union[pd.Series, np.ndarray], window: int = 14) -> Union[pd.Series, np.ndarray]:
        """
        Calculate Relative Strength Index (RSI).
        
        Consolidates 45+ duplicate implementations into a single, robust version.
        
        Args:
            prices: Price series (typically closing prices)
            window: Lookback window for RSI calculation
            
        Returns:
            RSI values (0-100 scale)
            
        Raises:
            ValueError: If window is invalid or prices are empty
        """
        if window <= 0:
            raise ValueError(f"Window must be positive, got {window}")
        
        if len(prices) < window + 1:
            raise ValueError(f"Need at least {window + 1} prices, got {len(prices)}")
        
        # Convert to pandas Series for consistent handling
        if isinstance(prices, np.ndarray):
            prices = pd.Series(prices)
            return_array = True
        else:
            return_array = False
        
        # Calculate price changes
        delta = prices.diff()
        
        # Separate gains and losses
        gain = delta.where(delta > 0, 0)
        loss = (-delta.where(delta < 0, 0))
        
        # Calculate rolling averages
        avg_gain = gain.rolling(window=window, min_periods=window).mean()
        avg_loss = loss.rolling(window=window, min_periods=window).mean()
        
        # Calculate RS and RSI with safe division
        if MATH_VALIDATION_AVAILABLE:
            rs = safe_divide(avg_gain, avg_loss, default=0.0)
            rsi = np.where((rs >= 0) & np.isfinite(rs), 100 - (100 / (1 + rs)), 50.0)
        else:
            # Fallback calculation
            rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0.0)
            rsi = np.where((rs >= 0) & np.isfinite(rs), 100 - (100 / (1 + rs)), 50.0)
        
        rsi_series = pd.Series(rsi, index=prices.index)
        
        return rsi_series.values if return_array else rsi_series
    
    @staticmethod
    def calculate_macd(
        prices: Union[pd.Series, np.ndarray], 
        fast: int = 12, 
        slow: int = 26, 
        signal: int = 9
    ) -> Tuple[Union[pd.Series, np.ndarray], Union[pd.Series, np.ndarray], Union[pd.Series, np.ndarray]]:
        """
        Calculate MACD (Moving Average Convergence Divergence).
        
        Consolidates 33+ duplicate implementations into a single, robust version.
        
        Args:
            prices: Price series (typically closing prices)
            fast: Fast EMA period
            slow: Slow EMA period  
            signal: Signal line EMA period
            
        Returns:
            Tuple of (macd_line, signal_line, histogram)
            
        Raises:
            ValueError: If parameters are invalid or prices are empty
        """
        if fast >= slow:
            raise ValueError(f"Fast period ({fast}) must be less than slow period ({slow})")
        
        if signal <= 0:
            raise ValueError(f"Signal period must be positive, got {signal}")
        
        if len(prices) < slow + signal:
            raise ValueError(f"Need at least {slow + signal} prices for MACD calculation")
        
        # Convert to pandas Series for consistent handling
        if isinstance(prices, np.ndarray):
            prices = pd.Series(prices)
            return_array = True
        else:
            return_array = False
        
        # Calculate EMAs
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        
        # MACD line
        macd_line = ema_fast - ema_slow
        
        # Signal line
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        
        # Histogram
        histogram = macd_line - signal_line
        
        if return_array:
            return macd_line.values, signal_line.values, histogram.values
        else:
            return macd_line, signal_line, histogram
    
    @staticmethod
    def calculate_bollinger_bands(
        prices: Union[pd.Series, np.ndarray], 
        window: int = 20, 
        num_std: float = 2.0
    ) -> Tuple[Union[pd.Series, np.ndarray], Union[pd.Series, np.ndarray], Union[pd.Series, np.ndarray]]:
        """
        Calculate Bollinger Bands.
        
        Args:
            prices: Price series
            window: Moving average window
            num_std: Number of standard deviations for bands
            
        Returns:
            Tuple of (upper_band, middle_band, lower_band)
        """
        if window <= 0:
            raise ValueError(f"Window must be positive, got {window}")
        
        if len(prices) < window:
            raise ValueError(f"Need at least {window} prices for Bollinger Bands")
        
        # Convert to pandas Series for consistent handling
        if isinstance(prices, np.ndarray):
            prices = pd.Series(prices)
            return_array = True
        else:
            return_array = False
        
        # Calculate rolling statistics
        rolling_mean = prices.rolling(window=window).mean()
        rolling_std = prices.rolling(window=window).std()
        
        # Calculate bands
        upper_band = rolling_mean + (rolling_std * num_std)
        lower_band = rolling_mean - (rolling_std * num_std)
        
        if return_array:
            return upper_band.values, rolling_mean.values, lower_band.values
        else:
            return upper_band, rolling_mean, lower_band
    
    @staticmethod
    def calculate_atr(
        data: pd.DataFrame, 
        window: int = 14
    ) -> Union[pd.Series, np.ndarray]:
        """
        Calculate Average True Range (ATR).
        
        Args:
            data: DataFrame with 'high', 'low', 'close' columns
            window: ATR calculation window
            
        Returns:
            ATR values
        """
        if window <= 0:
            raise ValueError(f"Window must be positive, got {window}")
        
        required_cols = ['high', 'low', 'close']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        high = data['high']
        low = data['low']
        close = data['close']
        
        # Calculate True Range components
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        # True Range is the maximum of the three
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # ATR is the rolling average of True Range
        atr = true_range.rolling(window=window).mean()
        
        return atr
    
    @staticmethod
    def calculate_stochastic(
        data: pd.DataFrame, 
        window: int = 14, 
        smooth_k: int = 3
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate Stochastic Oscillator (%K and %D).
        
        Args:
            data: DataFrame with 'high', 'low', 'close' columns
            window: Lookback window for highest high and lowest low
            smooth_k: Smoothing period for %K
            
        Returns:
            Tuple of (%K, %D)
        """
        if window <= 0:
            raise ValueError(f"Window must be positive, got {window}")
        
        required_cols = ['high', 'low', 'close']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        high = data['high']
        low = data['low']
        close = data['close']
        
        # Calculate highest high and lowest low over window
        lowest_low = low.rolling(window=window).min()
        highest_high = high.rolling(window=window).max()
        
        # Calculate %K with safe division
        range_diff = highest_high - lowest_low
        
        if MATH_VALIDATION_AVAILABLE:
            k_percent = safe_divide(100 * (close - lowest_low), range_diff, default=50.0)
        else:
            k_percent = np.where(
                (range_diff > 1e-10),  # Use small epsilon instead of zero
                100 * ((close - lowest_low) / range_diff),
                50.0
            )
        
        k_percent = pd.Series(k_percent, index=close.index)
        
        # Smooth %K if requested
        if smooth_k > 1:
            k_percent = k_percent.rolling(window=smooth_k).mean()
        
        # %D is the 3-period moving average of %K
        d_percent = k_percent.rolling(window=3).mean()
        
        return k_percent, d_percent


# Convenience functions for backward compatibility
def calculate_rsi(prices: Union[pd.Series, np.ndarray], window: int = 14) -> Union[pd.Series, np.ndarray]:
    """Convenience function for RSI calculation."""
    return TechnicalIndicators.calculate_rsi(prices, window)


def calculate_macd(
    prices: Union[pd.Series, np.ndarray], 
    fast: int = 12, 
    slow: int = 26, 
    signal: int = 9
) -> Tuple[Union[pd.Series, np.ndarray], Union[pd.Series, np.ndarray], Union[pd.Series, np.ndarray]]:
    """Convenience function for MACD calculation."""
    return TechnicalIndicators.calculate_macd(prices, fast, slow, signal)


def calculate_bollinger_bands(
    prices: Union[pd.Series, np.ndarray], 
    window: int = 20, 
    num_std: float = 2.0
) -> Tuple[Union[pd.Series, np.ndarray], Union[pd.Series, np.ndarray], Union[pd.Series, np.ndarray]]:
    """Convenience function for Bollinger Bands calculation."""
    return TechnicalIndicators.calculate_bollinger_bands(prices, window, num_std)


def calculate_atr(data: pd.DataFrame, window: int = 14) -> pd.Series:
    """Convenience function for ATR calculation."""
    return TechnicalIndicators.calculate_atr(data, window)


def calculate_stochastic(
    data: pd.DataFrame, 
    window: int = 14, 
    smooth_k: int = 3
) -> Tuple[pd.Series, pd.Series]:
    """Convenience function for Stochastic calculation."""
    return TechnicalIndicators.calculate_stochastic(data, window, smooth_k)


# Legacy support - warn about deprecated individual implementations
def _warn_deprecated_implementation(func_name: str, file_location: str):
    """Warn about deprecated duplicate implementations."""
    warnings.warn(
        f"Using deprecated {func_name} implementation from {file_location}. "
        f"Please use src.utils.technical_indicators.{func_name} instead.",
        DeprecationWarning,
        stacklevel=3
    )