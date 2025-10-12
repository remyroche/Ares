"""
Bollinger Bands Calculator

Centralized Bollinger Bands calculation that all other modules should use.
"""

import numpy as np
import pandas as pd
from typing import Union, Optional, Tuple
import warnings

class BollingerBandsCalculator:
    """
    Centralized Bollinger Bands calculator.
    
    All modules should use this instead of implementing their own Bollinger Bands calculations.
    """
    
    @staticmethod
    def calculate(prices: Union[pd.Series, np.ndarray], 
                  period: int = 20, 
                  std_dev: float = 2.0) -> Union[Tuple[pd.Series, pd.Series, pd.Series], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Calculate Bollinger Bands.
        
        Args:
            prices: Price series (close prices)
            period: Moving average period (default: 20)
            std_dev: Standard deviation multiplier (default: 2.0)
            
        Returns:
            Tuple of (Upper Band, Middle Band, Lower Band) as Series or arrays
        """
        if isinstance(prices, np.ndarray):
            return BollingerBandsCalculator._calculate_numpy(prices, period, std_dev)
        else:
            return BollingerBandsCalculator._calculate_pandas(prices, period, std_dev)
    
    @staticmethod
    def _calculate_pandas(prices: pd.Series, period: int, std_dev: float) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands using pandas operations."""
        if len(prices) < period:
            nan_series = pd.Series(np.full(len(prices), np.nan), index=prices.index)
            return nan_series, nan_series, nan_series
        
        # Calculate middle band (SMA)
        middle_band = prices.rolling(window=period).mean()
        
        # Calculate standard deviation
        std = prices.rolling(window=period).std()
        
        # Calculate upper and lower bands
        upper_band = middle_band + (std * std_dev)
        lower_band = middle_band - (std * std_dev)
        
        return upper_band, middle_band, lower_band
    
    @staticmethod
    def _calculate_numpy(prices: np.ndarray, period: int, std_dev: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate Bollinger Bands using numpy operations."""
        if len(prices) < period:
            nan_array = np.full(len(prices), np.nan)
            return nan_array, nan_array, nan_array
        
        # Calculate middle band (SMA)
        middle_band = BollingerBandsCalculator._rolling_mean(prices, period)
        
        # Calculate standard deviation
        std = BollingerBandsCalculator._rolling_std(prices, period)
        
        # Calculate upper and lower bands
        upper_band = middle_band + (std * std_dev)
        lower_band = middle_band - (std * std_dev)
        
        return upper_band, middle_band, lower_band
    
    @staticmethod
    def _rolling_mean(data: np.ndarray, window: int) -> np.ndarray:
        """Calculate rolling mean using numpy."""
        if len(data) < window:
            return np.full(len(data), np.nan)
        
        result = np.full(len(data), np.nan)
        for i in range(window - 1, len(data)):
            result[i] = np.mean(data[i - window + 1:i + 1])
        
        return result
    
    @staticmethod
    def _rolling_std(data: np.ndarray, window: int) -> np.ndarray:
        """Calculate rolling standard deviation using numpy."""
        if len(data) < window:
            return np.full(len(data), np.nan)
        
        result = np.full(len(data), np.nan)
        for i in range(window - 1, len(data)):
            result[i] = np.std(data[i - window + 1:i + 1], ddof=1)
        
        return result