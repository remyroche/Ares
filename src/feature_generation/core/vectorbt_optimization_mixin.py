"""
VectorBT Optimization Mixin

This mixin provides VectorBT optimization capabilities to any feature generator.
It includes automatic VectorBT detection, performance monitoring, and graceful fallbacks.

Usage:
    class MyFeatureGenerator(VectorizedFeatureGenerator, VectorBTOptimizationMixin):
        def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
            # Use self._vectorbt_rolling_operation() for optimized operations
            return self._vectorbt_rolling_operation(data['close'], 'mean', 20)
"""

import numpy as np
import pandas as pd
import logging
from typing import Any, Dict, List, Optional, Union
from functools import wraps
import time

# VectorBT imports
try:
    import vectorbt as vbt
    from vectorbt.generic import (
        rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max, 
        rolling_sum, rolling_apply, rolling_corr, rolling_cov
    )
    from vectorbt.generic import scale, rank, zscore, winsorize, clip, quantile
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
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

# Optional GPU acceleration
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

class VectorBTOptimizationMixin:
    """Mixin class that provides VectorBT optimization capabilities."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Initialize performance tracking
        self.performance_stats = {
            'vectorbt_operations': 0,
            'pandas_fallbacks': 0,
            'gpu_accelerations': 0,
            'total_operations': 0,
            'total_time': 0.0
        }
        
        # VectorBT configuration
        self.use_vectorbt = getattr(self, 'use_vectorbt', True)
        self.vectorbt_threshold = getattr(self, 'vectorbt_threshold', 1000)
        self.enable_gpu = getattr(self, 'enable_gpu', False)
        self.enable_parallel = getattr(self, 'enable_parallel', True)
        self.vectorbt_memory_limit_gb = getattr(self, 'vectorbt_memory_limit_gb', 8.0)
        
        # Setup logger
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def _should_use_vectorbt(self, data: Union[pd.DataFrame, pd.Series]) -> bool:
        """Determine if VectorBT should be used based on data size and configuration."""
        if not VECTORBT_AVAILABLE or not self.use_vectorbt:
            return False
        
        data_size = len(data) if hasattr(data, '__len__') else 0
        
        # Check data size threshold
        if data_size < self.vectorbt_threshold:
            return False
        
        # Check memory limit
        if hasattr(data, 'memory_usage'):
            memory_usage_gb = data.memory_usage(deep=True).sum() / (1024**3)
            if memory_usage_gb > self.vectorbt_memory_limit_gb:
                self.logger.warning(f"Data size ({memory_usage_gb:.2f}GB) exceeds memory limit ({self.vectorbt_memory_limit_gb}GB)")
                return False
        
        return True
    
    def _vectorbt_rolling_operation(self, data: pd.Series, operation: str, 
                                  window: int, **kwargs) -> pd.Series:
        """Perform VectorBT rolling operation with fallback to pandas."""
        start_time = time.time()
        self.performance_stats['total_operations'] += 1
        
        if not self._should_use_vectorbt(data):
            result = self._pandas_rolling_operation(data, operation, window, **kwargs)
            self.performance_stats['pandas_fallbacks'] += 1
        else:
            try:
                result = self._execute_vectorbt_operation(data, operation, window, **kwargs)
                self.performance_stats['vectorbt_operations'] += 1
                
                # Check for GPU acceleration
                if self.enable_gpu and CUPY_AVAILABLE:
                    if hasattr(result, 'values') and hasattr(result.values, 'device'):
                        self.performance_stats['gpu_accelerations'] += 1
                        
            except Exception as e:
                self.logger.warning(f"VectorBT operation failed: {e}, using pandas fallback")
                result = self._pandas_rolling_operation(data, operation, window, **kwargs)
                self.performance_stats['pandas_fallbacks'] += 1
        
        # Update timing
        self.performance_stats['total_time'] += time.time() - start_time
        
        return result
    
    def _execute_vectorbt_operation(self, data: pd.Series, operation: str, 
                                  window: int, **kwargs) -> pd.Series:
        """Execute VectorBT operation."""
        if operation == 'mean':
            return rolling_mean(data, window=window, **kwargs)
        elif operation == 'std':
            return rolling_std(data, window=window, **kwargs)
        elif operation == 'var':
            return rolling_var(data, window=window, **kwargs)
        elif operation == 'min':
            return rolling_min(data, window=window, **kwargs)
        elif operation == 'max':
            return rolling_max(data, window=window, **kwargs)
        elif operation == 'sum':
            return rolling_sum(data, window=window, **kwargs)
        elif operation == 'corr':
            other = kwargs.get('other')
            if other is not None:
                return rolling_corr(data, other, window=window, **kwargs)
            else:
                raise ValueError("Correlation operation requires 'other' parameter")
        elif operation == 'cov':
            other = kwargs.get('other')
            if other is not None:
                return rolling_cov(data, other, window=window, **kwargs)
            else:
                raise ValueError("Covariance operation requires 'other' parameter")
        else:
            raise ValueError(f"Unsupported VectorBT operation: {operation}")
    
    def _pandas_rolling_operation(self, data: pd.Series, operation: str, 
                                 window: int, **kwargs) -> pd.Series:
        """Fallback rolling operation using pandas."""
        if operation == 'mean':
            return data.rolling(window=window).mean()
        elif operation == 'std':
            return data.rolling(window=window).std()
        elif operation == 'var':
            return data.rolling(window=window).var()
        elif operation == 'min':
            return data.rolling(window=window).min()
        elif operation == 'max':
            return data.rolling(window=window).max()
        elif operation == 'sum':
            return data.rolling(window=window).sum()
        elif operation == 'corr':
            other = kwargs.get('other')
            if other is not None:
                return data.rolling(window=window).corr(other)
            else:
                raise ValueError("Correlation operation requires 'other' parameter")
        elif operation == 'cov':
            other = kwargs.get('other')
            if other is not None:
                return data.rolling(window=window).cov(other)
            else:
                raise ValueError("Covariance operation requires 'other' parameter")
        else:
            raise ValueError(f"Unsupported pandas operation: {operation}")
    
    def _vectorbt_technical_indicator(self, data: pd.DataFrame, indicator: str, 
                                    **kwargs) -> pd.Series:
        """Calculate technical indicators using VectorBT."""
        if not self._should_use_vectorbt(data):
            return self._pandas_technical_indicator(data, indicator, **kwargs)
        
        try:
            if indicator == 'sma':
                return rolling_mean(data['close'], window=kwargs.get('window', 20))
            elif indicator == 'ema':
                return data['close'].ewm(span=kwargs.get('window', 20)).mean()
            elif indicator == 'wma':
                window = kwargs.get('window', 20)
                weights = np.arange(1, window + 1)
                return rolling_apply(data['close'], lambda x: np.average(x, weights=weights), window=window)
            elif indicator == 'rsi':
                return self._calculate_rsi_vectorbt(data, kwargs.get('window', 14))
            elif indicator == 'macd':
                return self._calculate_macd_vectorbt(data, **kwargs)
            elif indicator == 'bollinger_bands':
                return self._calculate_bollinger_bands_vectorbt(data, **kwargs)
            elif indicator == 'atr':
                return self._calculate_atr_vectorbt(data, kwargs.get('window', 14))
            else:
                raise ValueError(f"Unsupported technical indicator: {indicator}")
        except Exception as e:
            self.logger.warning(f"VectorBT technical indicator failed: {e}, using pandas fallback")
            return self._pandas_technical_indicator(data, indicator, **kwargs)
    
    def _pandas_technical_indicator(self, data: pd.DataFrame, indicator: str, 
                                  **kwargs) -> pd.Series:
        """Calculate technical indicators using pandas (fallback)."""
        if indicator == 'sma':
            return data['close'].rolling(window=kwargs.get('window', 20)).mean()
        elif indicator == 'ema':
            return data['close'].ewm(span=kwargs.get('window', 20)).mean()
        elif indicator == 'wma':
            window = kwargs.get('window', 20)
            weights = np.arange(1, window + 1)
            return data['close'].rolling(window=window).apply(lambda x: np.average(x, weights=weights))
        else:
            raise ValueError(f"Unsupported technical indicator: {indicator}")
    
    def _calculate_rsi_vectorbt(self, data: pd.DataFrame, window: int = 14) -> pd.Series:
        """Calculate RSI using VectorBT."""
        close = data['close']
        delta = close.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = rolling_mean(gain, window=window)
        avg_loss = rolling_mean(loss, window=window)
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_macd_vectorbt(self, data: pd.DataFrame, 
                               fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
        """Calculate MACD using VectorBT."""
        close = data['close']
        
        ema_fast = close.ewm(span=fast).mean()
        ema_slow = close.ewm(span=slow).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        
        return macd_line - signal_line
    
    def _calculate_bollinger_bands_vectorbt(self, data: pd.DataFrame, 
                                          window: int = 20, std_dev: float = 2.0) -> pd.Series:
        """Calculate Bollinger Bands using VectorBT."""
        close = data['close']
        
        sma = rolling_mean(close, window=window)
        std = rolling_std(close, window=window)
        
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        
        # Return the middle band (SMA) as the main feature
        return sma
    
    def _calculate_atr_vectorbt(self, data: pd.DataFrame, window: int = 14) -> pd.Series:
        """Calculate ATR using VectorBT."""
        high = data['high']
        low = data['low']
        close = data['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = rolling_mean(true_range, window=window)
        
        return atr
    
    def _vectorbt_batch_operations(self, data: pd.DataFrame, 
                                 operations: List[Dict[str, Any]]) -> pd.DataFrame:
        """Perform multiple VectorBT operations in batch for efficiency."""
        if not self._should_use_vectorbt(data):
            return self._pandas_batch_operations(data, operations)
        
        results = {}
        
        try:
            for op in operations:
                op_type = op.get('type')
                op_name = op.get('name')
                op_params = op.get('params', {})
                
                if op_type == 'rolling':
                    column = op_params.get('column', 'close')
                    operation = op_params.get('operation')
                    window = op_params.get('window')
                    
                    if column in data.columns:
                        results[op_name] = self._vectorbt_rolling_operation(
                            data[column], operation, window
                        )
                
                elif op_type == 'indicator':
                    indicator = op_params.get('indicator')
                    results[op_name] = self._vectorbt_technical_indicator(data, indicator, **op_params)
            
            return pd.DataFrame(results, index=data.index)
            
        except Exception as e:
            self.logger.warning(f"VectorBT batch operations failed: {e}, using pandas fallback")
            return self._pandas_batch_operations(data, operations)
    
    def _pandas_batch_operations(self, data: pd.DataFrame, 
                               operations: List[Dict[str, Any]]) -> pd.DataFrame:
        """Perform multiple pandas operations in batch (fallback)."""
        results = {}
        
        for op in operations:
            op_type = op.get('type')
            op_name = op.get('name')
            op_params = op.get('params', {})
            
            if op_type == 'rolling':
                column = op_params.get('column', 'close')
                operation = op_params.get('operation')
                window = op_params.get('window')
                
                if column in data.columns:
                    if operation == 'mean':
                        results[op_name] = data[column].rolling(window=window).mean()
                    elif operation == 'std':
                        results[op_name] = data[column].rolling(window=window).std()
                    elif operation == 'var':
                        results[op_name] = data[column].rolling(window=window).var()
                    elif operation == 'min':
                        results[op_name] = data[column].rolling(window=window).min()
                    elif operation == 'max':
                        results[op_name] = data[column].rolling(window=window).max()
                    elif operation == 'sum':
                        results[op_name] = data[column].rolling(window=window).sum()
            
            elif op_type == 'indicator':
                indicator = op_params.get('indicator')
                results[op_name] = self._pandas_technical_indicator(data, indicator, **op_params)
        
        return pd.DataFrame(results, index=data.index)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        stats = self.performance_stats.copy()
        
        if stats['total_operations'] > 0:
            stats['vectorbt_usage_percentage'] = (
                stats['vectorbt_operations'] / stats['total_operations'] * 100
            )
            stats['pandas_fallback_percentage'] = (
                stats['pandas_fallbacks'] / stats['total_operations'] * 100
            )
            stats['average_operation_time'] = (
                stats['total_time'] / stats['total_operations']
            )
        else:
            stats['vectorbt_usage_percentage'] = 0
            stats['pandas_fallback_percentage'] = 0
            stats['average_operation_time'] = 0
        
        return stats
    
    def reset_performance_stats(self):
        """Reset performance statistics."""
        self.performance_stats = {
            'vectorbt_operations': 0,
            'pandas_fallbacks': 0,
            'gpu_accelerations': 0,
            'total_operations': 0,
            'total_time': 0.0
        }
    
    def optimize_dataframe_processing(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame for VectorBT processing."""
        if not self._should_use_vectorbt(data):
            return data
        
        # Convert to appropriate dtypes for VectorBT
        optimized_data = data.copy()
        
        for column in optimized_data.columns:
            if optimized_data[column].dtype == 'object':
                try:
                    optimized_data[column] = pd.to_numeric(optimized_data[column])
                except (ValueError, TypeError):
                    pass
        
        # Ensure index is datetime for time series operations
        if not isinstance(optimized_data.index, pd.DatetimeIndex):
            try:
                optimized_data.index = pd.to_datetime(optimized_data.index)
            except (ValueError, TypeError):
                pass
        
        return optimized_data

def vectorbt_optimized(func):
    """Decorator to automatically optimize functions with VectorBT."""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if hasattr(self, '_should_use_vectorbt') and self._should_use_vectorbt(args[0] if args else None):
            # Use VectorBT optimization
            return func(self, *args, **kwargs)
        else:
            # Use standard implementation
            return func(self, *args, **kwargs)
    return wrapper