"""
SMA (Simple Moving Average) Calculator

Centralized SMA calculation that all other modules should use.
"""

import numpy as np
import pandas as pd
from typing import Union, Optional
import warnings

class SMACalculator:
    """
    Centralized SMA calculator.
    
    All modules should use this instead of implementing their own SMA calculations.
    """
    
    @staticmethod
    def calculate(prices: Union[pd.Series, np.ndarray], period: int) -> Union[pd.Series, np.ndarray]:
        """
        Calculate SMA (Simple Moving Average).
        
        Args:
            prices: Price series
            period: SMA period
            
        Returns:
            SMA values as Series or array
        """
        if isinstance(prices, np.ndarray):
            return SMACalculator._calculate_numpy(prices, period)
        else:
            return SMACalculator._calculate_pandas(prices, period)
    
    @staticmethod
    def _calculate_pandas(prices: pd.Series, period: int) -> pd.Series:
        """Calculate SMA using pandas operations."""
        if len(prices) < period:
            return pd.Series(np.full(len(prices), np.nan), index=prices.index)
        
        return prices.rolling(window=period).mean()
    
    @staticmethod
    def _calculate_numpy(prices: np.ndarray, period: int) -> np.ndarray:
        """Calculate SMA using numpy operations."""
        if len(prices) < period:
            return np.full(len(prices), np.nan)
        
        result = np.full(len(prices), np.nan)
        for i in range(period - 1, len(prices)):
            result[i] = np.mean(prices[i - period + 1:i + 1])
        
        return result