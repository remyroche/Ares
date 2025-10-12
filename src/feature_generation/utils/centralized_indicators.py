"""
Centralized Technical Indicators

This module provides centralized implementations of all technical indicators
to eliminate code duplication across feature generators. All indicators use
VectorBTRollingOptimizer and VectorBTScaler from the centralized utilities.

Key Features:
- Single source of truth for all technical indicators
- Consistent VectorBT optimization across all implementations
- Unified error handling and fallback strategies
- Integration with features_common scalers and transforms
- Memory-efficient batch processing
"""

import numpy as np
import pandas as pd
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import warnings

# VectorBT imports for native optimization
try:
    import vectorbt as vbt
    from vectorbt.generic import rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max, rolling_sum, rolling_apply, rolling_corr, rolling_cov
    from vectorbt.generic import scale, rank, zscore, winsorize, clip, quantile
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
    rolling_apply = None
    rolling_corr = None
    rolling_cov = None
    scale = None
    rank = None
    zscore = None
    winsorize = None
    clip = None
    quantile = None
    warnings.warn("VectorBT not available. Install with: pip install vectorbt for optimized performance")

# Centralized utility imports
from ..utils.vectorbt_rolling_optimizer import get_vectorbt_rolling_optimizer
from ...features_common.transforms.vectorbt_scaler import VectorBTScaler, create_vectorbt_scaler
from ...utils.math_validation import safe_divide, validate_finite, safe_percentage_change

logger = logging.getLogger(__name__)


class CentralizedIndicators:
    """
    Centralized technical indicators using VectorBTRollingOptimizer and VectorBTScaler.
    
    This class provides a single source of truth for all technical indicators,
    ensuring consistency and eliminating code duplication across feature generators.
    """
    
    def __init__(self, enable_gpu: bool = False, enable_parallel: bool = True):
        """
        Initialize centralized indicators.
        
        Args:
            enable_gpu: Whether to enable GPU acceleration
            enable_parallel: Whether to enable parallel processing
        """
        self.enable_gpu = enable_gpu
        self.enable_parallel = enable_parallel
        
        # Initialize centralized utilities
        self.rolling_optimizer = get_vectorbt_rolling_optimizer()
        self.scaler = create_vectorbt_scaler(method='zscore')
        
        # Performance tracking
        self.performance_stats = {
            'vectorbt_operations': 0,
            'pandas_fallbacks': 0,
            'scaling_operations': 0,
            'total_operations': 0
        }
    
    def _should_use_vectorbt(self, data: pd.Series, threshold: int = 1000) -> bool:
        """Determine if VectorBT should be used based on data size."""
        return VECTORBT_AVAILABLE and len(data) >= threshold
    
    def _get_rolling_operation(self, data: pd.Series, operation: str, window: int, **kwargs) -> pd.Series:
        """Get rolling operation using centralized VectorBTRollingOptimizer."""
        try:
            if operation == 'mean':
                return self.rolling_optimizer.rolling_mean(data, window, **kwargs)
            elif operation == 'std':
                return self.rolling_optimizer.rolling_std(data, window, **kwargs)
            elif operation == 'var':
                return self.rolling_optimizer.rolling_var(data, window, **kwargs)
            elif operation == 'min':
                return self.rolling_optimizer.rolling_min(data, window, **kwargs)
            elif operation == 'max':
                return self.rolling_optimizer.rolling_max(data, window, **kwargs)
            elif operation == 'sum':
                return self.rolling_optimizer.rolling_sum(data, window, **kwargs)
            elif operation == 'corr':
                other = kwargs.get('other')
                if other is None:
                    raise ValueError("'other' parameter required for correlation")
                return self.rolling_optimizer.rolling_corr(data, other, window, **kwargs)
            elif operation == 'cov':
                other = kwargs.get('other')
                if other is None:
                    raise ValueError("'other' parameter required for covariance")
                return self.rolling_optimizer.rolling_cov(data, other, window, **kwargs)
            else:
                raise ValueError(f"Unsupported operation: {operation}")
        except Exception as e:
            logger.warning(f"VectorBT rolling operation failed: {e}, using pandas fallback")
            self.performance_stats['pandas_fallbacks'] += 1
            return self._pandas_rolling_operation(data, operation, window, **kwargs)
    
    def _pandas_rolling_operation(self, data: pd.Series, operation: str, window: int, **kwargs) -> pd.Series:
        """Fallback rolling operation using pandas."""
        rolling_obj = data.rolling(window=window, **kwargs)
        
        if operation == 'mean':
            return rolling_obj.mean()
        elif operation == 'std':
            return rolling_obj.std()
        elif operation == 'var':
            return rolling_obj.var()
        elif operation == 'min':
            return rolling_obj.min()
        elif operation == 'max':
            return rolling_obj.max()
        elif operation == 'sum':
            return rolling_obj.sum()
        elif operation == 'corr':
            other = kwargs.get('other')
            return rolling_obj.corr(other)
        elif operation == 'cov':
            other = kwargs.get('other')
            return rolling_obj.cov(other)
        else:
            raise ValueError(f"Unsupported operation: {operation}")
    
    def _normalize_feature(self, data: pd.Series, method: str = 'zscore') -> pd.Series:
        """Normalize feature using centralized VectorBTScaler."""
        try:
            scaler = create_vectorbt_scaler(method=method)
            result = scaler.fit_transform(data)
            self.performance_stats['scaling_operations'] += 1
            return result
        except Exception as e:
            logger.warning(f"VectorBT scaling failed: {e}, using fallback")
            return self._fallback_normalize(data, method)
    
    def _fallback_normalize(self, data: pd.Series, method: str = 'zscore') -> pd.Series:
        """Fallback normalization using pandas/numpy."""
        if method == 'zscore':
            return (data - data.mean()) / data.std()
        elif method == 'minmax':
            return (data - data.min()) / (data.max() - data.min())
        elif method == 'robust':
            median = data.median()
            mad = (data - median).abs().median()
            return (data - median) / mad
        else:
            return data
    
    # ==================== MOVING AVERAGES ====================
    
    def calculate_sma(self, data: pd.Series, window: int) -> pd.Series:
        """Calculate Simple Moving Average using centralized utilities."""
        self.performance_stats['total_operations'] += 1
        return self._get_rolling_operation(data, 'mean', window)
    
    def calculate_ema(self, data: pd.Series, window: int, alpha: Optional[float] = None) -> pd.Series:
        """Calculate Exponential Moving Average using centralized utilities."""
        self.performance_stats['total_operations'] += 1
        
        if alpha is None:
            alpha = 2.0 / (window + 1)
        
        if self._should_use_vectorbt(data):
            try:
                return data.ewm(span=window, alpha=alpha).mean()
            except Exception as e:
                logger.warning(f"VectorBT EMA failed: {e}, using pandas fallback")
                return data.ewm(span=window, alpha=alpha).mean()
        else:
            return data.ewm(span=window, alpha=alpha).mean()
    
    # ==================== MOMENTUM INDICATORS ====================
    
    def calculate_rsi(self, data: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Relative Strength Index using centralized utilities."""
        self.performance_stats['total_operations'] += 1
        
        if len(data) < window + 1:
            return pd.Series(np.nan, index=data.index)
        
        # Calculate price changes
        delta = data.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # Calculate rolling means using centralized utilities
        avg_gain = self._get_rolling_operation(gain, 'mean', window)
        avg_loss = self._get_rolling_operation(loss, 'mean', window)
        
        # Calculate RSI
        rs = safe_divide(avg_gain, avg_loss, default=1.0)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def calculate_macd(self, data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD using centralized utilities."""
        self.performance_stats['total_operations'] += 1
        
        # Calculate EMAs using centralized utilities
        ema_fast = self.calculate_ema(data, fast)
        ema_slow = self.calculate_ema(data, slow)
        
        # MACD line
        macd_line = ema_fast - ema_slow
        
        # Signal line
        signal_line = self.calculate_ema(macd_line, signal)
        
        # Histogram
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    def calculate_stochastic(self, high: pd.Series, low: pd.Series, close: pd.Series, 
                           k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        """Calculate Stochastic Oscillator using centralized utilities."""
        self.performance_stats['total_operations'] += 1
        
        # Calculate rolling min and max using centralized utilities
        lowest_low = self._get_rolling_operation(low, 'min', k_period)
        highest_high = self._get_rolling_operation(high, 'max', k_period)
        
        # Calculate %K
        k_percent = safe_divide(close - lowest_low, highest_high - lowest_low, default=0.0) * 100
        
        # Calculate %D (smoothed %K)
        d_percent = self._get_rolling_operation(k_percent, 'mean', d_period)
        
        return k_percent, d_percent
    
    def calculate_williams_r(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Williams %R using centralized utilities."""
        self.performance_stats['total_operations'] += 1
        
        # Calculate rolling min and max using centralized utilities
        lowest_low = self._get_rolling_operation(low, 'min', period)
        highest_high = self._get_rolling_operation(high, 'max', period)
        
        # Calculate Williams %R
        williams_r = safe_divide(highest_high - close, highest_high - lowest_low, default=0.0) * -100
        
        return williams_r
    
    def calculate_roc(self, data: pd.Series, period: int = 10) -> pd.Series:
        """Calculate Rate of Change using centralized utilities."""
        self.performance_stats['total_operations'] += 1
        
        shifted_data = data.shift(period)
        roc = safe_percentage_change(shifted_data, data)
        
        return roc
    
    def calculate_momentum(self, data: pd.Series, period: int = 10) -> pd.Series:
        """Calculate Momentum Oscillator using centralized utilities."""
        self.performance_stats['total_operations'] += 1
        
        return data - data.shift(period)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return self.performance_stats.copy()
    
    def reset_performance_stats(self):
        """Reset performance statistics."""
        self.performance_stats = {
            'vectorbt_operations': 0,
            'pandas_fallbacks': 0,
            'scaling_operations': 0,
            'total_operations': 0
        }


# Global instance for easy access
_global_indicators = None

def get_centralized_indicators() -> CentralizedIndicators:
    """Get the global centralized indicators instance."""
    global _global_indicators
    if _global_indicators is None:
        _global_indicators = CentralizedIndicators()
    return _global_indicators

def reset_centralized_indicators():
    """Reset the global centralized indicators instance."""
    global _global_indicators
    _global_indicators = None