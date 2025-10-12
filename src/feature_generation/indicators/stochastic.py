"""
Stochastic Oscillator Calculator

Centralized Stochastic calculation that all other modules should use.
"""

import numpy as np
import pandas as pd
from typing import Union, Optional, Tuple
import warnings

class StochasticCalculator:
    """
    Centralized Stochastic calculator.
    
    All modules should use this instead of implementing their own Stochastic calculations.
    """
    
    @staticmethod
    def calculate(high: Union[pd.Series, np.ndarray], 
                  low: Union[pd.Series, np.ndarray], 
                  close: Union[pd.Series, np.ndarray], 
                  k_period: int = 14, 
                  d_period: int = 3) -> Union[Tuple[pd.Series, pd.Series], Tuple[np.ndarray, np.ndarray]]:
        """
        Calculate Stochastic Oscillator.
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            k_period: %K period (default: 14)
            d_period: %D period (default: 3)
            
        Returns:
            Tuple of (%K, %D) as Series or arrays
        """
        if isinstance(high, np.ndarray):
            return StochasticCalculator._calculate_numpy(high, low, close, k_period, d_period)
        else:
            return StochasticCalculator._calculate_pandas(high, low, close, k_period, d_period)
    
    @staticmethod
    def _calculate_pandas(high: pd.Series, low: pd.Series, close: pd.Series, k_period: int, d_period: int) -> Tuple[pd.Series, pd.Series]:
        """Calculate Stochastic using pandas operations."""
        if len(close) < k_period:
            nan_series = pd.Series(np.full(len(close), np.nan), index=close.index)
            return nan_series, nan_series
        
        # Calculate %K
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        
        # Calculate %D (smoothed %K)
        d_percent = k_percent.rolling(window=d_period).mean()
        
        return k_percent, d_percent
    
    @staticmethod
    def _calculate_numpy(high: np.ndarray, low: np.ndarray, close: np.ndarray, k_period: int, d_period: int) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate Stochastic using numpy operations."""
        if len(close) < k_period:
            nan_array = np.full(len(close), np.nan)
            return nan_array, nan_array
        
        # Calculate %K
        lowest_low = StochasticCalculator._rolling_min(low, k_period)
        highest_high = StochasticCalculator._rolling_max(high, k_period)
        
        k_percent = np.full(len(close), np.nan)
        for i in range(k_period - 1, len(close)):
            if not (np.isnan(lowest_low[i]) or np.isnan(highest_high[i]) or highest_high[i] == lowest_low[i]):
                k_percent[i] = 100 * ((close[i] - lowest_low[i]) / (highest_high[i] - lowest_low[i]))
        
        # Calculate %D (smoothed %K)
        d_percent = StochasticCalculator._rolling_mean(k_percent, d_period)
        
        return k_percent, d_percent
    
    @staticmethod
    def _rolling_min(data: np.ndarray, window: int) -> np.ndarray:
        """Calculate rolling minimum using numpy."""
        if len(data) < window:
            return np.full(len(data), np.nan)
        
        result = np.full(len(data), np.nan)
        for i in range(window - 1, len(data)):
            result[i] = np.min(data[i - window + 1:i + 1])
        
        return result
    
    @staticmethod
    def _rolling_max(data: np.ndarray, window: int) -> np.ndarray:
        """Calculate rolling maximum using numpy."""
        if len(data) < window:
            return np.full(len(data), np.nan)
        
        result = np.full(len(data), np.nan)
        for i in range(window - 1, len(data)):
            result[i] = np.max(data[i - window + 1:i + 1])
        
        return result
    
    @staticmethod
    def _rolling_mean(data: np.ndarray, window: int) -> np.ndarray:
        """Calculate rolling mean using numpy."""
        if len(data) < window:
            return np.full(len(data), np.nan)
        
        result = np.full(len(data), np.nan)
        for i in range(window - 1, len(data)):
            result[i] = np.mean(data[i - window + 1:i + 1])
        
        return result