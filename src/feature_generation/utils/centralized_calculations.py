"""
Centralized Calculation Utilities

This module provides centralized implementations of common technical indicators
and mathematical operations to eliminate code duplication across feature generators.

All calculations use VectorBTRollingOptimizer and other centralized utilities
from the feature_generation/ and features_common/ directories.
"""

import numpy as np
import pandas as pd
import logging
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass

# Import centralized utilities
from .vectorbt_rolling_optimizer import get_vectorbt_rolling_optimizer, VectorBTRollingOptimizer
from ..core.feature_generator import FeatureConfig, FeatureCategory
from ...features_common.transforms.vectorbt_scaler import VectorBTScaler, VectorBTBatchScaler
from ...features_common.transforms.base_scaler import BaseScaler

logger = logging.getLogger(__name__)

@dataclass
class CalculationConfig:
    """Configuration for centralized calculations."""
    use_vectorbt: bool = True
    enable_gpu: bool = False
    enable_parallel: bool = True
    memory_efficient: bool = True
    chunk_size: int = 1000

class CentralizedCalculations:
    """
    Centralized calculation utilities using VectorBTRollingOptimizer and other
    centralized utilities to eliminate code duplication.
    """
    
    def __init__(self, config: Optional[CalculationConfig] = None):
        """
        Initialize centralized calculations.
        
        Args:
            config: Configuration for calculations
        """
        self.config = config or CalculationConfig()
        
        # Initialize VectorBT rolling optimizer
        self.rolling_optimizer = get_vectorbt_rolling_optimizer(
            enable_gpu=self.config.enable_gpu,
            enable_parallel=self.config.enable_parallel,
            memory_efficient=self.config.memory_efficient,
            chunk_size=self.config.chunk_size
        )
        
        # Initialize scalers
        self.scaler = VectorBTScaler(method='zscore', enable_gpu=self.config.enable_gpu)
        self.batch_scaler = VectorBTBatchScaler(method='zscore', enable_gpu=self.config.enable_gpu)
        
        logger.info("CentralizedCalculations initialized with VectorBT optimization")
    
    # =============================================================================
    # ROLLING OPERATIONS - Use VectorBTRollingOptimizer
    # =============================================================================
    
    def rolling_mean(self, data: Union[pd.Series, pd.DataFrame], window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Calculate rolling mean using VectorBTRollingOptimizer."""
        return self.rolling_optimizer.rolling_mean(data, window, **kwargs)
    
    def rolling_std(self, data: Union[pd.Series, pd.DataFrame], window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Calculate rolling standard deviation using VectorBTRollingOptimizer."""
        return self.rolling_optimizer.rolling_std(data, window, **kwargs)
    
    def rolling_var(self, data: Union[pd.Series, pd.DataFrame], window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Calculate rolling variance using VectorBTRollingOptimizer."""
        return self.rolling_optimizer.rolling_var(data, window, **kwargs)
    
    def rolling_min(self, data: Union[pd.Series, pd.DataFrame], window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Calculate rolling minimum using VectorBTRollingOptimizer."""
        return self.rolling_optimizer.rolling_min(data, window, **kwargs)
    
    def rolling_max(self, data: Union[pd.Series, pd.DataFrame], window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Calculate rolling maximum using VectorBTRollingOptimizer."""
        return self.rolling_optimizer.rolling_max(data, window, **kwargs)
    
    def rolling_sum(self, data: Union[pd.Series, pd.DataFrame], window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Calculate rolling sum using VectorBTRollingOptimizer."""
        return self.rolling_optimizer.rolling_sum(data, window, **kwargs)
    
    def rolling_quantile(self, data: Union[pd.Series, pd.DataFrame], window: int, q: float = 0.5, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Calculate rolling quantile using VectorBTRollingOptimizer."""
        return self.rolling_optimizer.rolling_quantile(data, window, q=q, **kwargs)
    
    def rolling_median(self, data: Union[pd.Series, pd.DataFrame], window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Calculate rolling median using VectorBTRollingOptimizer."""
        return self.rolling_optimizer.rolling_median(data, window, **kwargs)
    
    def rolling_skew(self, data: Union[pd.Series, pd.DataFrame], window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Calculate rolling skewness using VectorBTRollingOptimizer."""
        return self.rolling_optimizer.rolling_skew(data, window, **kwargs)
    
    def rolling_kurt(self, data: Union[pd.Series, pd.DataFrame], window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Calculate rolling kurtosis using VectorBTRollingOptimizer."""
        return self.rolling_optimizer.rolling_kurt(data, window, **kwargs)
    
    def rolling_corr(self, data: Union[pd.Series, pd.DataFrame], other: Union[pd.Series, pd.DataFrame], 
                    window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Calculate rolling correlation using VectorBTRollingOptimizer."""
        return self.rolling_optimizer.rolling_corr(data, other, window, **kwargs)
    
    def rolling_cov(self, data: Union[pd.Series, pd.DataFrame], other: Union[pd.Series, pd.DataFrame], 
                   window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Calculate rolling covariance using VectorBTRollingOptimizer."""
        return self.rolling_optimizer.rolling_cov(data, other, window, **kwargs)
    
    def rolling_apply(self, data: Union[pd.Series, pd.DataFrame], func: callable, 
                     window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Calculate rolling apply using VectorBTRollingOptimizer."""
        return self.rolling_optimizer.rolling_apply(data, func, window, **kwargs)
    
    # =============================================================================
    # MOVING AVERAGES - Centralized implementations
    # =============================================================================
    
    def sma(self, data: Union[pd.Series, pd.DataFrame], window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Calculate Simple Moving Average using VectorBTRollingOptimizer."""
        return self.rolling_mean(data, window, **kwargs)
    
    def ema(self, data: Union[pd.Series, pd.DataFrame], window: int, alpha: float = None, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Calculate Exponential Moving Average using VectorBTRollingOptimizer."""
        return self.rolling_optimizer.rolling_ewm(data, window, alpha=alpha, **kwargs)
    
    def wma(self, data: Union[pd.Series, pd.DataFrame], window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Calculate Weighted Moving Average."""
        if isinstance(data, pd.DataFrame):
            return data.apply(lambda col: self.wma(col, window, **kwargs), axis=0)
        
        weights = np.arange(1, window + 1)
        weights = weights / weights.sum()
        
        def weighted_mean(x):
            return np.average(x, weights=weights)
        
        return self.rolling_apply(data, weighted_mean, window, **kwargs)
    
    def hma(self, data: Union[pd.Series, pd.DataFrame], window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Calculate Hull Moving Average."""
        if isinstance(data, pd.DataFrame):
            return data.apply(lambda col: self.hma(col, window, **kwargs), axis=0)
        
        # HMA = WMA(2*WMA(n/2) - WMA(n))
        wma_half = self.wma(data, window // 2, **kwargs)
        wma_full = self.wma(data, window, **kwargs)
        hma_input = 2 * wma_half - wma_full
        return self.wma(hma_input, int(np.sqrt(window)), **kwargs)
    
    # =============================================================================
    # TECHNICAL INDICATORS - Centralized implementations
    # =============================================================================
    
    def rsi(self, data: Union[pd.Series, pd.DataFrame], window: int = 14, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Calculate Relative Strength Index using VectorBTRollingOptimizer."""
        if isinstance(data, pd.DataFrame):
            return data.apply(lambda col: self.rsi(col, window, **kwargs), axis=0)
        
        # Calculate price changes
        delta = data.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # Calculate average gain and loss using VectorBT
        avg_gain = self.rolling_mean(gain, window, **kwargs)
        avg_loss = self.rolling_mean(loss, window, **kwargs)
        
        # Calculate RSI
        rs = avg_gain / avg_loss.replace(0, 1)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def macd(self, data: Union[pd.Series, pd.DataFrame], fast: int = 12, slow: int = 26, 
             signal: int = 9, **kwargs) -> Dict[str, Union[pd.Series, pd.DataFrame]]:
        """Calculate MACD using VectorBTRollingOptimizer."""
        if isinstance(data, pd.DataFrame):
            result = {}
            for col in data.columns:
                col_result = self.macd(data[col], fast, slow, signal, **kwargs)
                for key, value in col_result.items():
                    if key not in result:
                        result[key] = pd.DataFrame(index=data.index)
                    result[key][col] = value
            return result
        
        # Calculate EMAs
        ema_fast = self.ema(data, fast, **kwargs)
        ema_slow = self.ema(data, slow, **kwargs)
        
        # Calculate MACD line
        macd_line = ema_fast - ema_slow
        
        # Calculate signal line
        signal_line = self.ema(macd_line, signal, **kwargs)
        
        # Calculate histogram
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }
    
    def bollinger_bands(self, data: Union[pd.Series, pd.DataFrame], window: int = 20, 
                       std_dev: float = 2.0, **kwargs) -> Dict[str, Union[pd.Series, pd.DataFrame]]:
        """Calculate Bollinger Bands using VectorBTRollingOptimizer."""
        if isinstance(data, pd.DataFrame):
            result = {}
            for col in data.columns:
                col_result = self.bollinger_bands(data[col], window, std_dev, **kwargs)
                for key, value in col_result.items():
                    if key not in result:
                        result[key] = pd.DataFrame(index=data.index)
                    result[key][col] = value
            return result
        
        # Calculate middle band (SMA)
        middle = self.sma(data, window, **kwargs)
        
        # Calculate standard deviation
        std = self.rolling_std(data, window, **kwargs)
        
        # Calculate upper and lower bands
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        
        # Calculate width and percent
        width = upper - lower
        percent = (data - lower) / (upper - lower)
        
        return {
            'upper': upper,
            'middle': middle,
            'lower': lower,
            'width': width,
            'percent': percent
        }
    
    def atr(self, high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14, **kwargs) -> pd.Series:
        """Calculate Average True Range using VectorBTRollingOptimizer."""
        # Calculate True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Calculate ATR using VectorBT
        return self.rolling_mean(tr, window, **kwargs)
    
    def stochastic(self, high: pd.Series, low: pd.Series, close: pd.Series, 
                  k_window: int = 14, d_window: int = 3, **kwargs) -> Dict[str, pd.Series]:
        """Calculate Stochastic Oscillator using VectorBTRollingOptimizer."""
        # Calculate %K
        lowest_low = self.rolling_min(low, k_window, **kwargs)
        highest_high = self.rolling_max(high, k_window, **kwargs)
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        
        # Calculate %D (smoothed %K)
        d_percent = self.rolling_mean(k_percent, d_window, **kwargs)
        
        return {
            'k': k_percent,
            'd': d_percent
        }
    
    def williams_r(self, high: pd.Series, low: pd.Series, close: pd.Series, 
                  window: int = 14, **kwargs) -> pd.Series:
        """Calculate Williams %R using VectorBTRollingOptimizer."""
        highest_high = self.rolling_max(high, window, **kwargs)
        lowest_low = self.rolling_min(low, window, **kwargs)
        williams_r = -100 * ((highest_high - close) / (highest_high - lowest_low))
        return williams_r
    
    def cci(self, high: pd.Series, low: pd.Series, close: pd.Series, 
           window: int = 20, **kwargs) -> pd.Series:
        """Calculate Commodity Channel Index using VectorBTRollingOptimizer."""
        # Calculate typical price
        typical_price = (high + low + close) / 3
        
        # Calculate CCI
        sma_tp = self.rolling_mean(typical_price, window, **kwargs)
        mad = self.rolling_mean((typical_price - sma_tp).abs(), window, **kwargs)
        cci = (typical_price - sma_tp) / (0.015 * mad)
        
        return cci
    
    def adx(self, high: pd.Series, low: pd.Series, close: pd.Series, 
           window: int = 14, **kwargs) -> pd.Series:
        """Calculate Average Directional Index using VectorBTRollingOptimizer."""
        # Calculate True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Calculate Directional Movement
        dm_plus = high.diff()
        dm_minus = -low.diff()
        
        dm_plus = np.where((dm_plus > dm_minus) & (dm_plus > 0), dm_plus, 0)
        dm_minus = np.where((dm_minus > dm_plus) & (dm_minus > 0), dm_minus, 0)
        
        # Calculate smoothed values using VectorBT
        atr = self.rolling_mean(tr, window, **kwargs)
        di_plus = 100 * (self.rolling_mean(pd.Series(dm_plus, index=high.index), window, **kwargs) / atr)
        di_minus = 100 * (self.rolling_mean(pd.Series(dm_minus, index=high.index), window, **kwargs) / atr)
        
        # Calculate ADX
        dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
        adx = self.rolling_mean(dx, window, **kwargs)
        
        return adx
    
    # =============================================================================
    # SCALING AND NORMALIZATION - Use VectorBTScaler
    # =============================================================================
    
    def zscore(self, data: Union[pd.Series, pd.DataFrame], **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Calculate z-score normalization using VectorBTScaler."""
        if isinstance(data, pd.DataFrame):
            return self.batch_scaler.fit_transform(data)
        else:
            return self.scaler.fit_transform(data)
    
    def minmax_scale(self, data: Union[pd.Series, pd.DataFrame], **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Calculate min-max scaling using VectorBTScaler."""
        if isinstance(data, pd.DataFrame):
            scaler = VectorBTBatchScaler(method='minmax', enable_gpu=self.config.enable_gpu)
            return scaler.fit_transform(data)
        else:
            scaler = VectorBTScaler(method='minmax', enable_gpu=self.config.enable_gpu)
            return scaler.fit_transform(data)
    
    def robust_scale(self, data: Union[pd.Series, pd.DataFrame], **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Calculate robust scaling using VectorBTScaler."""
        if isinstance(data, pd.DataFrame):
            scaler = VectorBTBatchScaler(method='robust', enable_gpu=self.config.enable_gpu)
            return scaler.fit_transform(data)
        else:
            scaler = VectorBTScaler(method='robust', enable_gpu=self.config.enable_gpu)
            return scaler.fit_transform(data)
    
    # =============================================================================
    # STATISTICAL CALCULATIONS - Use VectorBTRollingOptimizer
    # =============================================================================
    
    def rolling_skewness(self, data: Union[pd.Series, pd.DataFrame], window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Calculate rolling skewness using VectorBTRollingOptimizer."""
        return self.rolling_skew(data, window, **kwargs)
    
    def rolling_kurtosis(self, data: Union[pd.Series, pd.DataFrame], window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Calculate rolling kurtosis using VectorBTRollingOptimizer."""
        return self.rolling_kurt(data, window, **kwargs)
    
    def rolling_momentum(self, data: Union[pd.Series, pd.DataFrame], window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Calculate rolling momentum using VectorBTRollingOptimizer."""
        return data - data.shift(window)
    
    def rolling_roc(self, data: Union[pd.Series, pd.DataFrame], window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Calculate rolling rate of change using VectorBTRollingOptimizer."""
        return data.pct_change(window) * 100
    
    # =============================================================================
    # UTILITY METHODS
    # =============================================================================
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics from VectorBTRollingOptimizer."""
        return self.rolling_optimizer.get_performance_stats()
    
    def reset_performance_stats(self) -> None:
        """Reset performance statistics."""
        self.rolling_optimizer.reset_performance_stats()


# Global instance for easy access
_global_calculations: Optional[CentralizedCalculations] = None

def get_centralized_calculations(config: Optional[CalculationConfig] = None) -> CentralizedCalculations:
    """
    Get the global centralized calculations instance.
    
    Args:
        config: Configuration for calculations
        
    Returns:
        CentralizedCalculations instance
    """
    global _global_calculations
    
    if _global_calculations is None:
        _global_calculations = CentralizedCalculations(config)
    
    return _global_calculations

def reset_global_calculations() -> None:
    """Reset the global centralized calculations instance."""
    global _global_calculations
    _global_calculations = None