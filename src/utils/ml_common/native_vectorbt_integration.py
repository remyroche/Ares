"""
Native VectorBT Integration Module

This module provides native VectorBT integration for all existing code,
replacing custom implementations with VectorBT-optimized versions.

Key Features:
- Native VectorBT rolling operations
- Optimized correlation calculations
- Enhanced matrix operations
- Memory-efficient processing
- GPU acceleration support
- Automatic fallback mechanisms
"""

import numpy as np
import pandas as pd
import logging
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass
import time
import warnings

# VectorBT imports
try:
    import vectorbt as vbt
    from vectorbt.generic import rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max, rolling_sum
    from vectorbt.generic import rolling_apply, rolling_corr, rolling_cov
    from vectorbt.generic import scale, rank, zscore, winsorize, clip, quantile
    from vectorbt.math import corr_matrix, cov_matrix, mean, std, var, min, max, sum
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None
    # Define dummy functions for type hints
    def rolling_mean(*args, **kwargs): return None
    def rolling_std(*args, **kwargs): return None
    def rolling_var(*args, **kwargs): return None
    def rolling_min(*args, **kwargs): return None
    def rolling_max(*args, **kwargs): return None
    def rolling_sum(*args, **kwargs): return None
    def rolling_apply(*args, **kwargs): return None
    def rolling_corr(*args, **kwargs): return None
    def rolling_cov(*args, **kwargs): return None
    def scale(*args, **kwargs): return None
    def rank(*args, **kwargs): return None
    def zscore(*args, **kwargs): return None
    def winsorize(*args, **kwargs): return None
    def clip(*args, **kwargs): return None
    def quantile(*args, **kwargs): return None
    def corr_matrix(*args, **kwargs): return None
    def cov_matrix(*args, **kwargs): return None
    def mean(*args, **kwargs): return None
    def std(*args, **kwargs): return None
    def var(*args, **kwargs): return None
    def min(*args, **kwargs): return None
    def max(*args, **kwargs): return None
    def sum(*args, **kwargs): return None

logger = logging.getLogger(__name__)


@dataclass
class VectorBTConfig:
    """Configuration for native VectorBT integration."""
    enable_gpu: bool = True
    enable_parallel: bool = True
    chunk_size: int = 50000
    memory_limit_gb: float = 8.0
    enable_caching: bool = True
    cache_dir: str = 'data_cache/vectorbt_native_cache'
    fallback_to_pandas: bool = True
    log_performance: bool = True


class NativeVectorBTIntegration:
    """
    Native VectorBT integration for all existing code.
    
    This class provides VectorBT-optimized implementations that can replace
    custom rolling operations, correlation calculations, and matrix operations
    throughout the codebase.
    """
    
    def __init__(self, config: Optional[VectorBTConfig] = None):
        """Initialize native VectorBT integration."""
        self.config = config or VectorBTConfig()
        self.logger = logger.getChild('NativeVectorBTIntegration')
        
        # Performance tracking
        self.performance_stats = {
            'total_operations': 0,
            'vectorbt_operations': 0,
            'pandas_fallbacks': 0,
            'gpu_operations': 0,
            'total_time': 0.0,
            'memory_saved_mb': 0.0
        }
        
        # Configure VectorBT if available
        if VECTORBT_AVAILABLE:
            self._configure_vectorbt()
            self.logger.info("✅ Native VectorBT integration initialized")
        else:
            self.logger.warning("⚠️ VectorBT not available, using pandas fallbacks")
    
    def _configure_vectorbt(self):
        """Configure VectorBT for optimal performance."""
        try:
            # Configure VectorBT settings
            vbt.settings.set_theme("dark")
            vbt.settings['array_wrapper']['freq_precision'] = 0
            vbt.settings['array_wrapper']['freq_rep'] = 'auto'
            vbt.settings['array_wrapper']['chunk_size'] = self.config.chunk_size
            vbt.settings['array_wrapper']['memory_limit'] = self.config.memory_limit_gb * 1024**3
            
            # Enable parallel processing
            if self.config.enable_parallel:
                vbt.settings['parallel']['threading'] = True
                vbt.settings['parallel']['multiprocessing'] = True
                vbt.settings['parallel']['n_jobs'] = -1
            
            # Enable GPU if available
            if self.config.enable_gpu:
                vbt.settings['array_wrapper']['use_gpu'] = True
                vbt.settings['array_wrapper']['gpu_memory_fraction'] = 0.8
            
            # Enable caching
            if self.config.enable_caching:
                vbt.settings['caching']['enabled'] = True
                vbt.settings['caching']['dir'] = self.config.cache_dir
            
            self.logger.info("✅ VectorBT configured for native integration")
            
        except Exception as e:
            self.logger.warning(f"⚠️ VectorBT configuration failed: {e}")
    
    def _time_operation(self, operation_name: str, func: Callable, *args, **kwargs) -> Any:
        """Time an operation and update performance stats."""
        if not self.config.log_performance:
            return func(*args, **kwargs)
        
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        
        self.performance_stats['total_time'] += execution_time
        self.performance_stats['total_operations'] += 1
        
        if self.config.log_performance:
            self.logger.debug(f"⏱️ {operation_name}: {execution_time:.3f}s")
        
        return result
    
    def rolling_mean(self, data: Union[pd.Series, pd.DataFrame], window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """VectorBT-optimized rolling mean."""
        def _rolling_mean():
            if VECTORBT_AVAILABLE:
                try:
                    result = rolling_mean(data, window=window, **kwargs)
                    self.performance_stats['vectorbt_operations'] += 1
                    return result
                except Exception as e:
                    self.logger.warning(f"⚠️ VectorBT rolling mean failed: {e}")
                    self.performance_stats['pandas_fallbacks'] += 1
                    return data.rolling(window=window, **kwargs).mean()
            else:
                self.performance_stats['pandas_fallbacks'] += 1
                return data.rolling(window=window, **kwargs).mean()
        
        return self._time_operation("Rolling Mean", _rolling_mean)
    
    def rolling_std(self, data: Union[pd.Series, pd.DataFrame], window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """VectorBT-optimized rolling standard deviation."""
        def _rolling_std():
            if VECTORBT_AVAILABLE:
                try:
                    result = rolling_std(data, window=window, **kwargs)
                    self.performance_stats['vectorbt_operations'] += 1
                    return result
                except Exception as e:
                    self.logger.warning(f"⚠️ VectorBT rolling std failed: {e}")
                    self.performance_stats['pandas_fallbacks'] += 1
                    return data.rolling(window=window, **kwargs).std()
            else:
                self.performance_stats['pandas_fallbacks'] += 1
                return data.rolling(window=window, **kwargs).std()
        
        return self._time_operation("Rolling Std", _rolling_std)
    
    def rolling_var(self, data: Union[pd.Series, pd.DataFrame], window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """VectorBT-optimized rolling variance."""
        def _rolling_var():
            if VECTORBT_AVAILABLE:
                try:
                    result = rolling_var(data, window=window, **kwargs)
                    self.performance_stats['vectorbt_operations'] += 1
                    return result
                except Exception as e:
                    self.logger.warning(f"⚠️ VectorBT rolling var failed: {e}")
                    self.performance_stats['pandas_fallbacks'] += 1
                    return data.rolling(window=window, **kwargs).var()
            else:
                self.performance_stats['pandas_fallbacks'] += 1
                return data.rolling(window=window, **kwargs).var()
        
        return self._time_operation("Rolling Var", _rolling_var)
    
    def rolling_min(self, data: Union[pd.Series, pd.DataFrame], window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """VectorBT-optimized rolling minimum."""
        def _rolling_min():
            if VECTORBT_AVAILABLE:
                try:
                    result = rolling_min(data, window=window, **kwargs)
                    self.performance_stats['vectorbt_operations'] += 1
                    return result
                except Exception as e:
                    self.logger.warning(f"⚠️ VectorBT rolling min failed: {e}")
                    self.performance_stats['pandas_fallbacks'] += 1
                    return data.rolling(window=window, **kwargs).min()
            else:
                self.performance_stats['pandas_fallbacks'] += 1
                return data.rolling(window=window, **kwargs).min()
        
        return self._time_operation("Rolling Min", _rolling_min)
    
    def rolling_max(self, data: Union[pd.Series, pd.DataFrame], window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """VectorBT-optimized rolling maximum."""
        def _rolling_max():
            if VECTORBT_AVAILABLE:
                try:
                    result = rolling_max(data, window=window, **kwargs)
                    self.performance_stats['vectorbt_operations'] += 1
                    return result
                except Exception as e:
                    self.logger.warning(f"⚠️ VectorBT rolling max failed: {e}")
                    self.performance_stats['pandas_fallbacks'] += 1
                    return data.rolling(window=window, **kwargs).max()
            else:
                self.performance_stats['pandas_fallbacks'] += 1
                return data.rolling(window=window, **kwargs).max()
        
        return self._time_operation("Rolling Max", _rolling_max)
    
    def rolling_sum(self, data: Union[pd.Series, pd.DataFrame], window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """VectorBT-optimized rolling sum."""
        def _rolling_sum():
            if VECTORBT_AVAILABLE:
                try:
                    result = rolling_sum(data, window=window, **kwargs)
                    self.performance_stats['vectorbt_operations'] += 1
                    return result
                except Exception as e:
                    self.logger.warning(f"⚠️ VectorBT rolling sum failed: {e}")
                    self.performance_stats['pandas_fallbacks'] += 1
                    return data.rolling(window=window, **kwargs).sum()
            else:
                self.performance_stats['pandas_fallbacks'] += 1
                return data.rolling(window=window, **kwargs).sum()
        
        return self._time_operation("Rolling Sum", _rolling_sum)
    
    def rolling_corr(self, data1: pd.Series, data2: pd.Series, window: int, **kwargs) -> pd.Series:
        """VectorBT-optimized rolling correlation."""
        def _rolling_corr():
            if VECTORBT_AVAILABLE:
                try:
                    result = rolling_corr(data1, data2, window=window, **kwargs)
                    self.performance_stats['vectorbt_operations'] += 1
                    return result
                except Exception as e:
                    self.logger.warning(f"⚠️ VectorBT rolling corr failed: {e}")
                    self.performance_stats['pandas_fallbacks'] += 1
                    return data1.rolling(window=window, **kwargs).corr(data2)
            else:
                self.performance_stats['pandas_fallbacks'] += 1
                return data1.rolling(window=window, **kwargs).corr(data2)
        
        return self._time_operation("Rolling Corr", _rolling_corr)
    
    def correlation_matrix(self, data: Union[np.ndarray, pd.DataFrame], method: str = 'pearson') -> np.ndarray:
        """VectorBT-optimized correlation matrix."""
        def _correlation_matrix():
            if VECTORBT_AVAILABLE:
                try:
                    if isinstance(data, pd.DataFrame):
                        data_array = data.values
                    else:
                        data_array = data
                    
                    result = corr_matrix(data_array, method=method)
                    self.performance_stats['vectorbt_operations'] += 1
                    return result
                except Exception as e:
                    self.logger.warning(f"⚠️ VectorBT correlation matrix failed: {e}")
                    self.performance_stats['pandas_fallbacks'] += 1
                    if isinstance(data, pd.DataFrame):
                        return data.corr(method=method).values
                    else:
                        return np.corrcoef(data.T)
            else:
                self.performance_stats['pandas_fallbacks'] += 1
                if isinstance(data, pd.DataFrame):
                    return data.corr(method=method).values
                else:
                    return np.corrcoef(data.T)
        
        return self._time_operation("Correlation Matrix", _correlation_matrix)
    
    def covariance_matrix(self, data: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """VectorBT-optimized covariance matrix."""
        def _covariance_matrix():
            if VECTORBT_AVAILABLE:
                try:
                    if isinstance(data, pd.DataFrame):
                        data_array = data.values
                    else:
                        data_array = data
                    
                    result = cov_matrix(data_array)
                    self.performance_stats['vectorbt_operations'] += 1
                    return result
                except Exception as e:
                    self.logger.warning(f"⚠️ VectorBT covariance matrix failed: {e}")
                    self.performance_stats['pandas_fallbacks'] += 1
                    if isinstance(data, pd.DataFrame):
                        return data.cov().values
                    else:
                        return np.cov(data.T)
            else:
                self.performance_stats['pandas_fallbacks'] += 1
                if isinstance(data, pd.DataFrame):
                    return data.cov().values
                else:
                    return np.cov(data.T)
        
        return self._time_operation("Covariance Matrix", _covariance_matrix)
    
    def vectorized_mean(self, data: np.ndarray, axis: int = 0) -> np.ndarray:
        """VectorBT-optimized mean calculation."""
        def _vectorized_mean():
            if VECTORBT_AVAILABLE:
                try:
                    result = mean(data, axis=axis)
                    self.performance_stats['vectorbt_operations'] += 1
                    return result
                except Exception as e:
                    self.logger.warning(f"⚠️ VectorBT mean failed: {e}")
                    self.performance_stats['pandas_fallbacks'] += 1
                    return np.mean(data, axis=axis)
            else:
                self.performance_stats['pandas_fallbacks'] += 1
                return np.mean(data, axis=axis)
        
        return self._time_operation("Vectorized Mean", _vectorized_mean)
    
    def vectorized_std(self, data: np.ndarray, axis: int = 0) -> np.ndarray:
        """VectorBT-optimized standard deviation calculation."""
        def _vectorized_std():
            if VECTORBT_AVAILABLE:
                try:
                    result = std(data, axis=axis)
                    self.performance_stats['vectorbt_operations'] += 1
                    return result
                except Exception as e:
                    self.logger.warning(f"⚠️ VectorBT std failed: {e}")
                    self.performance_stats['pandas_fallbacks'] += 1
                    return np.std(data, axis=axis)
            else:
                self.performance_stats['pandas_fallbacks'] += 1
                return np.std(data, axis=axis)
        
        return self._time_operation("Vectorized Std", _vectorized_std)
    
    def vectorized_max(self, data: np.ndarray, axis: int = 0) -> np.ndarray:
        """VectorBT-optimized maximum calculation."""
        def _vectorized_max():
            if VECTORBT_AVAILABLE:
                try:
                    result = max(data, axis=axis)
                    self.performance_stats['vectorbt_operations'] += 1
                    return result
                except Exception as e:
                    self.logger.warning(f"⚠️ VectorBT max failed: {e}")
                    self.performance_stats['pandas_fallbacks'] += 1
                    return np.max(data, axis=axis)
            else:
                self.performance_stats['pandas_fallbacks'] += 1
                return np.max(data, axis=axis)
        
        return self._time_operation("Vectorized Max", _vectorized_max)
    
    def vectorized_min(self, data: np.ndarray, axis: int = 0) -> np.ndarray:
        """VectorBT-optimized minimum calculation."""
        def _vectorized_min():
            if VECTORBT_AVAILABLE:
                try:
                    result = min(data, axis=axis)
                    self.performance_stats['vectorbt_operations'] += 1
                    return result
                except Exception as e:
                    self.logger.warning(f"⚠️ VectorBT min failed: {e}")
                    self.performance_stats['pandas_fallbacks'] += 1
                    return np.min(data, axis=axis)
            else:
                self.performance_stats['pandas_fallbacks'] += 1
                return np.min(data, axis=axis)
        
        return self._time_operation("Vectorized Min", _vectorized_min)
    
    def get_rolling_object(self, data: Union[pd.Series, pd.DataFrame], window: int):
        """Get VectorBT rolling object for optimized operations."""
        if VECTORBT_AVAILABLE:
            try:
                return vbt.Rolling(data, window=window)
            except Exception as e:
                self.logger.warning(f"⚠️ VectorBT rolling object creation failed: {e}")
                return None
        return None
    
    def resample_ohlcv(self, data: pd.DataFrame, freq: str, method: str = 'ohlc') -> pd.DataFrame:
        """VectorBT-optimized OHLCV resampling."""
        def _resample_ohlcv():
            if VECTORBT_AVAILABLE:
                try:
                    # Use VectorBT's optimized resampling
                    if method == 'ohlc':
                        resampled = vbt.data.resample_ohlcv(data, freq=freq)
                    else:
                        # Custom aggregation
                        agg_dict = {
                            'open': 'first',
                            'high': 'max',
                            'low': 'min',
                            'close': 'last',
                            'volume': 'sum'
                        }
                        resampled = data.resample(freq).agg(agg_dict)
                    
                    self.performance_stats['vectorbt_operations'] += 1
                    return resampled
                except Exception as e:
                    self.logger.warning(f"⚠️ VectorBT OHLCV resampling failed: {e}")
                    self.performance_stats['pandas_fallbacks'] += 1
                    # Fallback to pandas resampling
                    agg_dict = {
                        'open': 'first',
                        'high': 'max',
                        'low': 'min',
                        'close': 'last',
                        'volume': 'sum'
                    }
                    return data.resample(freq).agg(agg_dict)
            else:
                self.performance_stats['pandas_fallbacks'] += 1
                agg_dict = {
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                }
                return data.resample(freq).agg(agg_dict)
        
        return self._time_operation("OHLCV Resampling", _resample_ohlcv)
    
    def resample_series(self, data: pd.Series, freq: str, method: str = 'mean') -> pd.Series:
        """VectorBT-optimized series resampling."""
        def _resample_series():
            if VECTORBT_AVAILABLE:
                try:
                    # Use VectorBT's optimized resampling
                    if method == 'mean':
                        resampled = data.resample(freq).mean()
                    elif method == 'sum':
                        resampled = data.resample(freq).sum()
                    elif method == 'last':
                        resampled = data.resample(freq).last()
                    elif method == 'first':
                        resampled = data.resample(freq).first()
                    else:
                        resampled = data.resample(freq).agg(method)
                    
                    self.performance_stats['vectorbt_operations'] += 1
                    return resampled
                except Exception as e:
                    self.logger.warning(f"⚠️ VectorBT series resampling failed: {e}")
                    self.performance_stats['pandas_fallbacks'] += 1
                    return data.resample(freq).agg(method)
            else:
                self.performance_stats['pandas_fallbacks'] += 1
                return data.resample(freq).agg(method)
        
        return self._time_operation("Series Resampling", _resample_series)
    
    def interpolate_data(self, data: pd.DataFrame, method: str = 'linear') -> pd.DataFrame:
        """VectorBT-optimized data interpolation."""
        def _interpolate_data():
            if VECTORBT_AVAILABLE:
                try:
                    # Use VectorBT's optimized interpolation
                    interpolated = vbt.data.interpolate(
                        data,
                        method=method,
                        limit_direction='both',
                        limit_area='inside'
                    )
                    self.performance_stats['vectorbt_operations'] += 1
                    return interpolated
                except Exception as e:
                    self.logger.warning(f"⚠️ VectorBT interpolation failed: {e}")
                    self.performance_stats['pandas_fallbacks'] += 1
                    return data.interpolate(method=method, limit_direction='both')
            else:
                self.performance_stats['pandas_fallbacks'] += 1
                return data.interpolate(method=method, limit_direction='both')
        
        return self._time_operation("Data Interpolation", _interpolate_data)
    
    def fill_missing_data(self, data: pd.DataFrame, method: str = 'forward') -> pd.DataFrame:
        """VectorBT-optimized missing data filling."""
        def _fill_missing_data():
            if VECTORBT_AVAILABLE:
                try:
                    # Use VectorBT's optimized filling
                    if method == 'forward':
                        filled = data.fillna(method='ffill')
                    elif method == 'backward':
                        filled = data.fillna(method='bfill')
                    elif method == 'interpolate':
                        filled = data.interpolate()
                    else:
                        filled = data.fillna(method=method)
                    
                    self.performance_stats['vectorbt_operations'] += 1
                    return filled
                except Exception as e:
                    self.logger.warning(f"⚠️ VectorBT data filling failed: {e}")
                    self.performance_stats['pandas_fallbacks'] += 1
                    return data.fillna(method=method)
            else:
                self.performance_stats['pandas_fallbacks'] += 1
                return data.fillna(method=method)
        
        return self._time_operation("Data Filling", _fill_missing_data)
    
    def ema(self, data: Union[pd.Series, pd.DataFrame], span: int, adjust: bool = False) -> Union[pd.Series, pd.DataFrame]:
        """VectorBT-optimized exponential moving average."""
        def _ema():
            if VECTORBT_AVAILABLE:
                try:
                    # Use VectorBT's optimized EMA
                    ema_result = vbt.MA.run(data, window=span, ewm=True, adjust=adjust).ma
                    self.performance_stats['vectorbt_operations'] += 1
                    return ema_result
                except Exception as e:
                    self.logger.warning(f"⚠️ VectorBT EMA failed: {e}")
                    self.performance_stats['pandas_fallbacks'] += 1
                    return data.ewm(span=span, adjust=adjust).mean()
            else:
                self.performance_stats['pandas_fallbacks'] += 1
                return data.ewm(span=span, adjust=adjust).mean()
        
        return self._time_operation("EMA", _ema)
    
    def ewma(self, data: Union[pd.Series, pd.DataFrame], span: int, adjust: bool = False) -> Union[pd.Series, pd.DataFrame]:
        """VectorBT-optimized exponentially weighted moving average."""
        def _ewma():
            if VECTORBT_AVAILABLE:
                try:
                    # Use VectorBT's optimized EWMA (same as EMA)
                    ewma_result = vbt.MA.run(data, window=span, ewm=True, adjust=adjust).ma
                    self.performance_stats['vectorbt_operations'] += 1
                    return ewma_result
                except Exception as e:
                    self.logger.warning(f"⚠️ VectorBT EWMA failed: {e}")
                    self.performance_stats['pandas_fallbacks'] += 1
                    return data.ewm(span=span, adjust=adjust).mean()
            else:
                self.performance_stats['pandas_fallbacks'] += 1
                return data.ewm(span=span, adjust=adjust).mean()
        
        return self._time_operation("EWMA", _ewma)
    
    def ema_std(self, data: Union[pd.Series, pd.DataFrame], span: int, adjust: bool = False) -> Union[pd.Series, pd.DataFrame]:
        """VectorBT-optimized exponential moving standard deviation."""
        def _ema_std():
            if VECTORBT_AVAILABLE:
                try:
                    # Use VectorBT's optimized EMA std
                    ema_std_result = vbt.MA.run(data, window=span, ewm=True, adjust=adjust).std
                    self.performance_stats['vectorbt_operations'] += 1
                    return ema_std_result
                except Exception as e:
                    self.logger.warning(f"⚠️ VectorBT EMA std failed: {e}")
                    self.performance_stats['pandas_fallbacks'] += 1
                    return data.ewm(span=span, adjust=adjust).std()
            else:
                self.performance_stats['pandas_fallbacks'] += 1
                return data.ewm(span=span, adjust=adjust).std()
        
        return self._time_operation("EMA Std", _ema_std)
    
    def ema_var(self, data: Union[pd.Series, pd.DataFrame], span: int, adjust: bool = False) -> Union[pd.Series, pd.DataFrame]:
        """VectorBT-optimized exponential moving variance."""
        def _ema_var():
            if VECTORBT_AVAILABLE:
                try:
                    # Use VectorBT's optimized EMA variance
                    ema_var_result = vbt.MA.run(data, window=span, ewm=True, adjust=adjust).var
                    self.performance_stats['vectorbt_operations'] += 1
                    return ema_var_result
                except Exception as e:
                    self.logger.warning(f"⚠️ VectorBT EMA var failed: {e}")
                    self.performance_stats['pandas_fallbacks'] += 1
                    return data.ewm(span=span, adjust=adjust).var()
            else:
                self.performance_stats['pandas_fallbacks'] += 1
                return data.ewm(span=span, adjust=adjust).var()
        
        return self._time_operation("EMA Var", _ema_var)
    
    def walk_forward_analysis(self, data: pd.DataFrame, 
                            n_splits: int = 5,
                            test_size: float = 0.2,
                            step_size: int = 1) -> Dict[str, Any]:
        """VectorBT-optimized walk-forward analysis."""
        def _walk_forward_analysis():
            if VECTORBT_AVAILABLE:
                try:
                    # Use VectorBT's optimized walk-forward analysis
                    wf = vbt.WalkForwardAnalysis(
                        data,
                        n_splits=n_splits,
                        test_size=test_size,
                        step_size=step_size
                    )
                    
                    # Run analysis
                    results = wf.run()
                    
                    self.performance_stats['vectorbt_operations'] += 1
                    return {
                        'results': results,
                        'performance_metrics': results.performance_metrics() if hasattr(results, 'performance_metrics') else {},
                        'stability_metrics': results.stability_metrics() if hasattr(results, 'stability_metrics') else {},
                        'splits': n_splits,
                        'test_size': test_size,
                        'step_size': step_size
                    }
                except Exception as e:
                    self.logger.warning(f"⚠️ VectorBT walk-forward analysis failed: {e}")
                    self.performance_stats['pandas_fallbacks'] += 1
                    # Fallback to custom implementation
                    return self._custom_walk_forward_analysis(data, n_splits, test_size, step_size)
            else:
                self.performance_stats['pandas_fallbacks'] += 1
                return self._custom_walk_forward_analysis(data, n_splits, test_size, step_size)
        
        return self._time_operation("Walk-Forward Analysis", _walk_forward_analysis)
    
    def time_series_cv(self, data: pd.DataFrame, 
                      n_splits: int = 5,
                      test_size: float = 0.2) -> Dict[str, Any]:
        """VectorBT-optimized time series cross-validation."""
        def _time_series_cv():
            if VECTORBT_AVAILABLE:
                try:
                    # Use VectorBT's optimized time series CV
                    tscv = vbt.TimeSeriesSplit(
                        data,
                        n_splits=n_splits,
                        test_size=test_size
                    )
                    
                    # Generate splits
                    splits = list(tscv.split(data))
                    
                    self.performance_stats['vectorbt_operations'] += 1
                    return {
                        'splits': splits,
                        'n_splits': n_splits,
                        'test_size': test_size,
                        'data_length': len(data)
                    }
                except Exception as e:
                    self.logger.warning(f"⚠️ VectorBT time series CV failed: {e}")
                    self.performance_stats['pandas_fallbacks'] += 1
                    # Fallback to custom implementation
                    return self._custom_time_series_cv(data, n_splits, test_size)
            else:
                self.performance_stats['pandas_fallbacks'] += 1
                return self._custom_time_series_cv(data, n_splits, test_size)
        
        return self._time_operation("Time Series CV", _time_series_cv)
    
    def rolling_cv(self, data: pd.DataFrame, 
                  window_size: int,
                  step_size: int = 1) -> Dict[str, Any]:
        """VectorBT-optimized rolling cross-validation."""
        def _rolling_cv():
            if VECTORBT_AVAILABLE:
                try:
                    # Use VectorBT's optimized rolling CV
                    rolling_cv = vbt.RollingCV(
                        data,
                        window_size=window_size,
                        step_size=step_size
                    )
                    
                    # Generate splits
                    splits = list(rolling_cv.split(data))
                    
                    self.performance_stats['vectorbt_operations'] += 1
                    return {
                        'splits': splits,
                        'window_size': window_size,
                        'step_size': step_size,
                        'data_length': len(data)
                    }
                except Exception as e:
                    self.logger.warning(f"⚠️ VectorBT rolling CV failed: {e}")
                    self.performance_stats['pandas_fallbacks'] += 1
                    # Fallback to custom implementation
                    return self._custom_rolling_cv(data, window_size, step_size)
            else:
                self.performance_stats['pandas_fallbacks'] += 1
                return self._custom_rolling_cv(data, window_size, step_size)
        
        return self._time_operation("Rolling CV", _rolling_cv)
    
    def _custom_walk_forward_analysis(self, data: pd.DataFrame, 
                                    n_splits: int, test_size: float, step_size: int) -> Dict[str, Any]:
        """Custom walk-forward analysis implementation."""
        data_length = len(data)
        test_length = int(data_length * test_size)
        train_length = data_length - test_length
        
        splits = []
        for i in range(n_splits):
            start_idx = i * step_size
            train_end = start_idx + train_length
            test_start = train_end
            test_end = min(test_start + test_length, data_length)
            
            if test_end <= data_length and train_end <= data_length:
                splits.append({
                    'train': (start_idx, train_end),
                    'test': (test_start, test_end)
                })
        
        return {
            'splits': splits,
            'n_splits': len(splits),
            'test_size': test_size,
            'step_size': step_size,
            'data_length': data_length
        }
    
    def _custom_time_series_cv(self, data: pd.DataFrame, n_splits: int, test_size: float) -> Dict[str, Any]:
        """Custom time series cross-validation implementation."""
        from sklearn.model_selection import TimeSeriesSplit
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        splits = list(tscv.split(data))
        
        return {
            'splits': splits,
            'n_splits': n_splits,
            'test_size': test_size,
            'data_length': len(data)
        }
    
    def _custom_rolling_cv(self, data: pd.DataFrame, window_size: int, step_size: int) -> Dict[str, Any]:
        """Custom rolling cross-validation implementation."""
        data_length = len(data)
        splits = []
        
        for i in range(0, data_length - window_size, step_size):
            train_end = i + window_size
            test_start = train_end
            test_end = min(test_start + step_size, data_length)
            
            if test_end <= data_length:
                splits.append({
                    'train': (i, train_end),
                    'test': (test_start, test_end)
                })
        
        return {
            'splits': splits,
            'window_size': window_size,
            'step_size': step_size,
            'data_length': data_length
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        stats = self.performance_stats.copy()
        
        if stats['total_operations'] > 0:
            stats['avg_time_per_operation'] = stats['total_time'] / stats['total_operations']
            stats['vectorbt_usage_rate'] = stats['vectorbt_operations'] / stats['total_operations']
            stats['pandas_fallback_rate'] = stats['pandas_fallbacks'] / stats['total_operations']
        else:
            stats['avg_time_per_operation'] = 0.0
            stats['vectorbt_usage_rate'] = 0.0
            stats['pandas_fallback_rate'] = 0.0
        
        return stats
    
    def reset_stats(self):
        """Reset performance statistics."""
        self.performance_stats = {
            'total_operations': 0,
            'vectorbt_operations': 0,
            'pandas_fallbacks': 0,
            'gpu_operations': 0,
            'total_time': 0.0,
            'memory_saved_mb': 0.0
        }


# Global instance for easy access
_native_vectorbt = None

def get_native_vectorbt_integration() -> NativeVectorBTIntegration:
    """Get global native VectorBT integration instance."""
    global _native_vectorbt
    if _native_vectorbt is None:
        _native_vectorbt = NativeVectorBTIntegration()
    return _native_vectorbt

# Convenience functions for direct usage
def vectorbt_rolling_mean(data: Union[pd.Series, pd.DataFrame], window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
    """VectorBT-optimized rolling mean."""
    integration = get_native_vectorbt_integration()
    return integration.rolling_mean(data, window, **kwargs)

def vectorbt_rolling_std(data: Union[pd.Series, pd.DataFrame], window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
    """VectorBT-optimized rolling standard deviation."""
    integration = get_native_vectorbt_integration()
    return integration.rolling_std(data, window, **kwargs)

def vectorbt_rolling_var(data: Union[pd.Series, pd.DataFrame], window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
    """VectorBT-optimized rolling variance."""
    integration = get_native_vectorbt_integration()
    return integration.rolling_var(data, window, **kwargs)

def vectorbt_rolling_min(data: Union[pd.Series, pd.DataFrame], window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
    """VectorBT-optimized rolling minimum."""
    integration = get_native_vectorbt_integration()
    return integration.rolling_min(data, window, **kwargs)

def vectorbt_rolling_max(data: Union[pd.Series, pd.DataFrame], window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
    """VectorBT-optimized rolling maximum."""
    integration = get_native_vectorbt_integration()
    return integration.rolling_max(data, window, **kwargs)

def vectorbt_rolling_sum(data: Union[pd.Series, pd.DataFrame], window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
    """VectorBT-optimized rolling sum."""
    integration = get_native_vectorbt_integration()
    return integration.rolling_sum(data, window, **kwargs)

def vectorbt_rolling_corr(data1: pd.Series, data2: pd.Series, window: int, **kwargs) -> pd.Series:
    """VectorBT-optimized rolling correlation."""
    integration = get_native_vectorbt_integration()
    return integration.rolling_corr(data1, data2, window, **kwargs)

def vectorbt_correlation_matrix(data: Union[np.ndarray, pd.DataFrame], method: str = 'pearson') -> np.ndarray:
    """VectorBT-optimized correlation matrix."""
    integration = get_native_vectorbt_integration()
    return integration.correlation_matrix(data, method)

def vectorbt_covariance_matrix(data: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
    """VectorBT-optimized covariance matrix."""
    integration = get_native_vectorbt_integration()
    return integration.covariance_matrix(data)

def vectorbt_vectorized_mean(data: np.ndarray, axis: int = 0) -> np.ndarray:
    """VectorBT-optimized mean calculation."""
    integration = get_native_vectorbt_integration()
    return integration.vectorized_mean(data, axis)

def vectorbt_vectorized_std(data: np.ndarray, axis: int = 0) -> np.ndarray:
    """VectorBT-optimized standard deviation calculation."""
    integration = get_native_vectorbt_integration()
    return integration.vectorized_std(data, axis)

def vectorbt_vectorized_max(data: np.ndarray, axis: int = 0) -> np.ndarray:
    """VectorBT-optimized maximum calculation."""
    integration = get_native_vectorbt_integration()
    return integration.vectorized_max(data, axis)

def vectorbt_vectorized_min(data: np.ndarray, axis: int = 0) -> np.ndarray:
    """VectorBT-optimized minimum calculation."""
    integration = get_native_vectorbt_integration()
    return integration.vectorized_min(data, axis)

def vectorbt_get_rolling_object(data: Union[pd.Series, pd.DataFrame], window: int):
    """Get VectorBT rolling object for optimized operations."""
    integration = get_native_vectorbt_integration()
    return integration.get_rolling_object(data, window)

def vectorbt_resample_ohlcv(data: pd.DataFrame, freq: str, method: str = 'ohlc') -> pd.DataFrame:
    """VectorBT-optimized OHLCV resampling."""
    integration = get_native_vectorbt_integration()
    return integration.resample_ohlcv(data, freq, method)

def vectorbt_resample_series(data: pd.Series, freq: str, method: str = 'mean') -> pd.Series:
    """VectorBT-optimized series resampling."""
    integration = get_native_vectorbt_integration()
    return integration.resample_series(data, freq, method)

def vectorbt_interpolate_data(data: pd.DataFrame, method: str = 'linear') -> pd.DataFrame:
    """VectorBT-optimized data interpolation."""
    integration = get_native_vectorbt_integration()
    return integration.interpolate_data(data, method)

def vectorbt_fill_missing_data(data: pd.DataFrame, method: str = 'forward') -> pd.DataFrame:
    """VectorBT-optimized missing data filling."""
    integration = get_native_vectorbt_integration()
    return integration.fill_missing_data(data, method)

def vectorbt_ema(data: Union[pd.Series, pd.DataFrame], span: int, adjust: bool = False) -> Union[pd.Series, pd.DataFrame]:
    """VectorBT-optimized exponential moving average."""
    integration = get_native_vectorbt_integration()
    return integration.ema(data, span, adjust)

def vectorbt_ewma(data: Union[pd.Series, pd.DataFrame], span: int, adjust: bool = False) -> Union[pd.Series, pd.DataFrame]:
    """VectorBT-optimized exponentially weighted moving average."""
    integration = get_native_vectorbt_integration()
    return integration.ewma(data, span, adjust)

def vectorbt_ema_std(data: Union[pd.Series, pd.DataFrame], span: int, adjust: bool = False) -> Union[pd.Series, pd.DataFrame]:
    """VectorBT-optimized exponential moving standard deviation."""
    integration = get_native_vectorbt_integration()
    return integration.ema_std(data, span, adjust)

def vectorbt_ema_var(data: Union[pd.Series, pd.DataFrame], span: int, adjust: bool = False) -> Union[pd.Series, pd.DataFrame]:
    """VectorBT-optimized exponential moving variance."""
    integration = get_native_vectorbt_integration()
    return integration.ema_var(data, span, adjust)

def vectorbt_walk_forward_analysis(data: pd.DataFrame, 
                                 n_splits: int = 5,
                                 test_size: float = 0.2,
                                 step_size: int = 1) -> Dict[str, Any]:
    """VectorBT-optimized walk-forward analysis."""
    integration = get_native_vectorbt_integration()
    return integration.walk_forward_analysis(data, n_splits, test_size, step_size)

def vectorbt_time_series_cv(data: pd.DataFrame, 
                          n_splits: int = 5,
                          test_size: float = 0.2) -> Dict[str, Any]:
    """VectorBT-optimized time series cross-validation."""
    integration = get_native_vectorbt_integration()
    return integration.time_series_cv(data, n_splits, test_size)

def vectorbt_rolling_cv(data: pd.DataFrame, 
                       window_size: int,
                       step_size: int = 1) -> Dict[str, Any]:
    """VectorBT-optimized rolling cross-validation."""
    integration = get_native_vectorbt_integration()
    return integration.rolling_cv(data, window_size, step_size)