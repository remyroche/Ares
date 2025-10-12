"""
MACD (Moving Average Convergence Divergence) Calculator

Centralized MACD calculation that all other modules should use.
"""

import numpy as np
import pandas as pd
from typing import Union, Optional, Tuple
import warnings

class MACDCalculator:
    """
    Centralized MACD calculator.
    
    All modules should use this instead of implementing their own MACD calculations.
    """
    
    @staticmethod
    def calculate(prices: Union[pd.Series, np.ndarray], 
                  fast: int = 12, 
                  slow: int = 26, 
                  signal: int = 9) -> Union[Tuple[pd.Series, pd.Series, pd.Series], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Calculate MACD (Moving Average Convergence Divergence).
        
        Args:
            prices: Price series (close prices)
            fast: Fast EMA period (default: 12)
            slow: Slow EMA period (default: 26)
            signal: Signal line EMA period (default: 9)
            
        Returns:
            Tuple of (MACD line, Signal line, Histogram) as Series or arrays
        """
        if isinstance(prices, np.ndarray):
            return MACDCalculator._calculate_numpy(prices, fast, slow, signal)
        else:
            return MACDCalculator._calculate_pandas(prices, fast, slow, signal)
    
    @staticmethod
    def _calculate_pandas(prices: pd.Series, fast: int, slow: int, signal: int) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD using pandas operations."""
        if len(prices) < slow:
            nan_series = pd.Series(np.full(len(prices), np.nan), index=prices.index)
            return nan_series, nan_series, nan_series
        
        # Calculate EMAs
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        
        # Calculate MACD line
        macd_line = ema_fast - ema_slow
        
        # Calculate signal line
        signal_line = macd_line.ewm(span=signal).mean()
        
        # Calculate histogram
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    @staticmethod
    def _calculate_numpy(prices: np.ndarray, fast: int, slow: int, signal: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate MACD using numpy operations."""
        if len(prices) < slow:
            nan_array = np.full(len(prices), np.nan)
            return nan_array, nan_array, nan_array
        
        # Calculate EMAs
        ema_fast = MACDCalculator._calculate_ema(prices, fast)
        ema_slow = MACDCalculator._calculate_ema(prices, slow)
        
        # Calculate MACD line
        macd_line = ema_fast - ema_slow
        
        # Calculate signal line
        signal_line = MACDCalculator._calculate_ema(macd_line, signal)
        
        # Calculate histogram
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    @staticmethod
    def _calculate_ema(prices: np.ndarray, period: int) -> np.ndarray:
        """Calculate Exponential Moving Average using numpy."""
        if len(prices) < period:
            return np.full(len(prices), np.nan)
        
        alpha = 2.0 / (period + 1)
        ema = np.full(len(prices), np.nan)
        ema[period - 1] = np.mean(prices[:period])
        
        for i in range(period, len(prices)):
            ema[i] = alpha * prices[i] + (1 - alpha) * ema[i - 1]
        
        return ema