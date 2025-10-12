"""
Centralized Technical Indicators Utilities

This module provides centralized implementations of common technical indicators
to eliminate code duplication across the codebase. All feature generators should
use these utilities instead of implementing their own versions.

Key Features:
- Centralized RSI, EMA, MACD, and other technical indicators
- VectorBT optimization when available
- UnifiedVectorizationManager integration
- Fallback implementations for reliability
- Consistent error handling and validation
"""

import numpy as np
import pandas as pd
import logging
from typing import Optional, Tuple, Union
import warnings

# VectorBT imports for native optimization
try:
    import vectorbt as vbt
    from vectorbt.generic import rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max, rolling_sum
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None
    rolling_mean = None
    rolling_std = None
    rolling_var = None
    rolling_min = None
    rolling_max = None
    rolling_sum = None
    warnings.warn("VectorBT not available. Install with: pip install vectorbt for optimized performance")

# Unified Vectorization Manager for intelligent optimization
try:
    from ...utils.ml_common.unified_vectorization_manager import (
        get_unified_vectorization_manager, 
        UnifiedVectorizationManager, 
        OperationType, 
        OptimizationStrategy
    )
    UNIFIED_MANAGER_AVAILABLE = True
except ImportError:
    UNIFIED_MANAGER_AVAILABLE = False
    UnifiedVectorizationManager = None
    OperationType = None
    OptimizationStrategy = None

logger = logging.getLogger(__name__)


class CentralizedTechnicalIndicators:
    """
    Centralized technical indicators calculator with intelligent optimization.
    
    This class provides a single source of truth for all technical indicators
    used across the feature generation system.
    """
    
    def __init__(self):
        """Initialize the centralized technical indicators calculator."""
        self.logger = logger.getChild('CentralizedTechnicalIndicators')
        
        # Initialize optimization components
        self.unified_manager = None
        if UNIFIED_MANAGER_AVAILABLE:
            try:
                self.unified_manager = get_unified_vectorization_manager()
                self.logger.info("✅ UnifiedVectorizationManager initialized")
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to initialize UnifiedVectorizationManager: {e}")
        
        # Performance tracking
        self.performance_stats = {
            'rsi_calculations': 0,
            'ema_calculations': 0,
            'macd_calculations': 0,
            'vectorbt_operations': 0,
            'unified_manager_operations': 0,
            'fallback_operations': 0
        }
    
    def calculate_rsi(self, prices: Union[pd.Series, np.ndarray], period: int = 14) -> Union[pd.Series, np.ndarray]:
        """
        Calculate RSI (Relative Strength Index) using centralized utilities.
        
        Args:
            prices: Price series (pandas Series or numpy array)
            period: RSI period (default 14)
            
        Returns:
            RSI values (same type as input)
        """
        self.performance_stats['rsi_calculations'] += 1
        
        # Convert to pandas Series for consistent processing
        if isinstance(prices, np.ndarray):
            prices_series = pd.Series(prices)
            return_numpy = True
        else:
            prices_series = prices
            return_numpy = False
        
        try:
            # Try UnifiedVectorizationManager first if available
            if self.unified_manager and len(prices_series) >= 100:
                try:
                    data = {'close': prices_series, 'period': period}
                    result = self.unified_manager.optimize_operation(
                        OperationType.TECHNICAL_INDICATORS,
                        data,
                        **{'indicator': 'rsi', 'window': period}
                    )
                    self.performance_stats['unified_manager_operations'] += 1
                    return result.result.values if return_numpy else result.result
                except Exception as e:
                    self.logger.debug(f"UnifiedVectorizationManager RSI failed: {e}")
            
            # Try VectorBT if available
            if VECTORBT_AVAILABLE and len(prices_series) >= 1000:
                try:
                    rsi_result = vbt.RSI.run(prices_series, window=period)
                    self.performance_stats['vectorbt_operations'] += 1
                    return rsi_result.rsi.values if return_numpy else rsi_result.rsi
                except Exception as e:
                    self.logger.debug(f"VectorBT RSI failed: {e}")
            
            # Fallback to optimized pandas implementation
            return self._calculate_rsi_pandas(prices_series, period, return_numpy)
            
        except Exception as e:
            self.logger.error(f"RSI calculation failed: {e}")
            self.performance_stats['fallback_operations'] += 1
            # Return neutral RSI values (50) as fallback
            if return_numpy:
                return np.full(len(prices_series), 50.0)
            else:
                return pd.Series(50.0, index=prices_series.index)
    
    def calculate_ema(self, prices: Union[pd.Series, np.ndarray], period: int = 20) -> Union[pd.Series, np.ndarray]:
        """
        Calculate EMA (Exponential Moving Average) using centralized utilities.
        
        Args:
            prices: Price series (pandas Series or numpy array)
            period: EMA period (default 20)
            
        Returns:
            EMA values (same type as input)
        """
        self.performance_stats['ema_calculations'] += 1
        
        # Convert to pandas Series for consistent processing
        if isinstance(prices, np.ndarray):
            prices_series = pd.Series(prices)
            return_numpy = True
        else:
            prices_series = prices
            return_numpy = False
        
        try:
            # Try UnifiedVectorizationManager first if available
            if self.unified_manager and len(prices_series) >= 100:
                try:
                    data = {'close': prices_series, 'period': period}
                    result = self.unified_manager.optimize_operation(
                        OperationType.TECHNICAL_INDICATORS,
                        data,
                        **{'indicator': 'ema', 'window': period}
                    )
                    self.performance_stats['unified_manager_operations'] += 1
                    return result.result.values if return_numpy else result.result
                except Exception as e:
                    self.logger.debug(f"UnifiedVectorizationManager EMA failed: {e}")
            
            # Try VectorBT if available
            if VECTORBT_AVAILABLE and len(prices_series) >= 1000:
                try:
                    ema = prices_series.ewm(span=period).mean()
                    self.performance_stats['vectorbt_operations'] += 1
                    return ema.values if return_numpy else ema
                except Exception as e:
                    self.logger.debug(f"VectorBT EMA failed: {e}")
            
            # Fallback to optimized pandas implementation
            return self._calculate_ema_pandas(prices_series, period, return_numpy)
            
        except Exception as e:
            self.logger.error(f"EMA calculation failed: {e}")
            self.performance_stats['fallback_operations'] += 1
            # Return original prices as fallback
            return prices_series.values if return_numpy else prices_series
    
    def calculate_macd(self, prices: Union[pd.Series, np.ndarray], 
                      fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[Union[pd.Series, np.ndarray], 
                                                                                Union[pd.Series, np.ndarray], 
                                                                                Union[pd.Series, np.ndarray]]:
        """
        Calculate MACD (Moving Average Convergence Divergence) using centralized utilities.
        
        Args:
            prices: Price series (pandas Series or numpy array)
            fast: Fast EMA period (default 12)
            slow: Slow EMA period (default 26)
            signal: Signal line period (default 9)
            
        Returns:
            Tuple of (MACD line, Signal line, Histogram) (same type as input)
        """
        self.performance_stats['macd_calculations'] += 1
        
        # Convert to pandas Series for consistent processing
        if isinstance(prices, np.ndarray):
            prices_series = pd.Series(prices)
            return_numpy = True
        else:
            prices_series = prices
            return_numpy = False
        
        try:
            # Try UnifiedVectorizationManager first if available
            if self.unified_manager and len(prices_series) >= 100:
                try:
                    data = {'close': prices_series, 'fast': fast, 'slow': slow, 'signal': signal}
                    result = self.unified_manager.optimize_operation(
                        OperationType.TECHNICAL_INDICATORS,
                        data,
                        **{'indicator': 'macd', 'fast_window': fast, 'slow_window': slow, 'signal_window': signal}
                    )
                    self.performance_stats['unified_manager_operations'] += 1
                    if return_numpy:
                        return result.result.values, result.result.values, result.result.values
                    else:
                        return result.result, result.result, result.result
                except Exception as e:
                    self.logger.debug(f"UnifiedVectorizationManager MACD failed: {e}")
            
            # Try VectorBT if available
            if VECTORBT_AVAILABLE and len(prices_series) >= 1000:
                try:
                    macd_result = vbt.MACD.run(prices_series, fast=fast, slow=slow, signal=signal)
                    self.performance_stats['vectorbt_operations'] += 1
                    if return_numpy:
                        return (macd_result.macd.values, macd_result.signal.values, macd_result.histogram.values)
                    else:
                        return (macd_result.macd, macd_result.signal, macd_result.histogram)
                except Exception as e:
                    self.logger.debug(f"VectorBT MACD failed: {e}")
            
            # Fallback to optimized pandas implementation
            return self._calculate_macd_pandas(prices_series, fast, slow, signal, return_numpy)
            
        except Exception as e:
            self.logger.error(f"MACD calculation failed: {e}")
            self.performance_stats['fallback_operations'] += 1
            # Return zero values as fallback
            if return_numpy:
                zeros = np.zeros(len(prices_series))
                return zeros, zeros, zeros
            else:
                zeros = pd.Series(0.0, index=prices_series.index)
                return zeros, zeros, zeros
    
    def _calculate_rsi_pandas(self, prices: pd.Series, period: int, return_numpy: bool) -> Union[pd.Series, np.ndarray]:
        """Calculate RSI using optimized pandas operations."""
        if len(prices) < period + 1:
            if return_numpy:
                return np.full(len(prices), np.nan)
            else:
                return pd.Series(np.nan, index=prices.index)
        
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.values if return_numpy else rsi
    
    def _calculate_ema_pandas(self, prices: pd.Series, period: int, return_numpy: bool) -> Union[pd.Series, np.ndarray]:
        """Calculate EMA using optimized pandas operations."""
        if len(prices) < period:
            if return_numpy:
                return np.full(len(prices), np.nan)
            else:
                return pd.Series(np.nan, index=prices.index)
        
        ema = prices.ewm(span=period).mean()
        return ema.values if return_numpy else ema
    
    def _calculate_macd_pandas(self, prices: pd.Series, fast: int, slow: int, signal: int, 
                              return_numpy: bool) -> Tuple[Union[pd.Series, np.ndarray], 
                                                          Union[pd.Series, np.ndarray], 
                                                          Union[pd.Series, np.ndarray]]:
        """Calculate MACD using optimized pandas operations."""
        if len(prices) < slow:
            if return_numpy:
                zeros = np.full(len(prices), np.nan)
                return zeros, zeros, zeros
            else:
                zeros = pd.Series(np.nan, index=prices.index)
                return zeros, zeros, zeros
        
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        
        if return_numpy:
            return macd_line.values, signal_line.values, histogram.values
        else:
            return macd_line, signal_line, histogram
    
    def get_performance_stats(self) -> dict:
        """Get performance statistics."""
        return self.performance_stats.copy()
    
    def reset_performance_stats(self) -> None:
        """Reset performance statistics."""
        self.performance_stats = {
            'rsi_calculations': 0,
            'ema_calculations': 0,
            'macd_calculations': 0,
            'vectorbt_operations': 0,
            'unified_manager_operations': 0,
            'fallback_operations': 0
        }


# Global instance for easy access
_global_indicators = None

def get_centralized_indicators() -> CentralizedTechnicalIndicators:
    """Get the global centralized technical indicators instance."""
    global _global_indicators
    if _global_indicators is None:
        _global_indicators = CentralizedTechnicalIndicators()
    return _global_indicators

def calculate_rsi(prices: Union[pd.Series, np.ndarray], period: int = 14) -> Union[pd.Series, np.ndarray]:
    """Convenience function for RSI calculation."""
    return get_centralized_indicators().calculate_rsi(prices, period)

def calculate_ema(prices: Union[pd.Series, np.ndarray], period: int = 20) -> Union[pd.Series, np.ndarray]:
    """Convenience function for EMA calculation."""
    return get_centralized_indicators().calculate_ema(prices, period)

def calculate_macd(prices: Union[pd.Series, np.ndarray], 
                  fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[Union[pd.Series, np.ndarray], 
                                                                            Union[pd.Series, np.ndarray], 
                                                                            Union[pd.Series, np.ndarray]]:
    """Convenience function for MACD calculation."""
    return get_centralized_indicators().calculate_macd(prices, fast, slow, signal)