"""
RSI (Relative Strength Index) Calculator

Centralized RSI calculation that all other modules should use.
"""

import numpy as np
import pandas as pd
from typing import Union, Optional
import warnings

class RSICalculator:
    """
    Centralized RSI calculator.
    
    All modules should use this instead of implementing their own RSI calculations.
    """
    
    @staticmethod
    def calculate(prices: Union[pd.Series, np.ndarray], period: int = 14) -> Union[pd.Series, np.ndarray]:
        """
        Calculate RSI (Relative Strength Index).
        
        Args:
            prices: Price series (close prices)
            period: RSI period (default: 14)
            
        Returns:
            RSI values as Series or array
        """
        if isinstance(prices, np.ndarray):
            return RSICalculator._calculate_numpy(prices, period)
        else:
            return RSICalculator._calculate_pandas(prices, period)
    
    @staticmethod
    def _calculate_pandas(prices: pd.Series, period: int) -> pd.Series:
        """Calculate RSI using pandas operations."""
        if len(prices) < period + 1:
            return pd.Series(np.full(len(prices), np.nan), index=prices.index)
        
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        rs = avg_gain / avg_loss.replace(0, 1)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    @staticmethod
    def _calculate_numpy(prices: np.ndarray, period: int) -> np.ndarray:
        """Calculate RSI using numpy operations."""
        if len(prices) < period + 1:
            return np.full(len(prices), np.nan)
        
        delta = np.diff(prices, prepend=prices[0])
        gains = np.where(delta > 0, delta, 0)
        losses = np.where(delta < 0, -delta, 0)
        
        # Calculate rolling means
        avg_gains = RSICalculator._rolling_mean(gains, period)
        avg_losses = RSICalculator._rolling_mean(losses, period)
        
        # Calculate RS and RSI
        rs = np.divide(avg_gains, avg_losses, out=np.ones_like(avg_gains), where=avg_losses!=0)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    @staticmethod
    def _rolling_mean(data: np.ndarray, window: int) -> np.ndarray:
        """Calculate rolling mean using numpy."""
        if len(data) < window:
            return np.full(len(data), np.nan)
        
        result = np.full(len(data), np.nan)
        for i in range(window - 1, len(data)):
            result[i] = np.mean(data[i - window + 1:i + 1])
        
        return result