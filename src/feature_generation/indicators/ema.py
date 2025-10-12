"""
EMA (Exponential Moving Average) Calculator

Centralized EMA calculation that all other modules should use.
"""

import numpy as np
import pandas as pd
from typing import Union, Optional
import warnings

class EMACalculator:
    """
    Centralized EMA calculator.
    
    All modules should use this instead of implementing their own EMA calculations.
    """
    
    @staticmethod
    def calculate(prices: Union[pd.Series, np.ndarray], period: int) -> Union[pd.Series, np.ndarray]:
        """
        Calculate EMA (Exponential Moving Average).
        
        Args:
            prices: Price series
            period: EMA period
            
        Returns:
            EMA values as Series or array
        """
        if isinstance(prices, np.ndarray):
            return EMACalculator._calculate_numpy(prices, period)
        else:
            return EMACalculator._calculate_pandas(prices, period)
    
    @staticmethod
    def _calculate_pandas(prices: pd.Series, period: int) -> pd.Series:
        """Calculate EMA using pandas operations."""
        if len(prices) < period:
            return pd.Series(np.full(len(prices), np.nan), index=prices.index)
        
        return prices.ewm(span=period).mean()
    
    @staticmethod
    def _calculate_numpy(prices: np.ndarray, period: int) -> np.ndarray:
        """Calculate EMA using numpy operations."""
        if len(prices) < period:
            return np.full(len(prices), np.nan)
        
        alpha = 2.0 / (period + 1)
        ema = np.full(len(prices), np.nan)
        ema[period - 1] = np.mean(prices[:period])
        
        for i in range(period, len(prices)):
            ema[i] = alpha * prices[i] + (1 - alpha) * ema[i - 1]
        
        return ema