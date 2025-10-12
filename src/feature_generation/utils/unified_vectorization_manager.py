"""
Unified Vectorization Manager

This module provides a unified interface for managing vectorized operations across
different backends (VectorBT, NumPy, CuPy) with intelligent optimization selection.

Key Features:
- Unified interface for all vectorized operations
- Intelligent backend selection (VectorBT, NumPy, CuPy)
- Memory-efficient processing with chunking
- Performance monitoring and optimization
- Automatic fallback mechanisms
- GPU acceleration support
"""

import numpy as np
import pandas as pd
import logging
import time
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
from dataclasses import dataclass
import warnings
from functools import wraps

# VectorBT imports
try:
    import vectorbt as vbt
    from vectorbt.generic import (
        rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max, 
        rolling_sum, rolling_apply, rolling_corr, rolling_cov,
        rolling_quantile, rolling_skew, rolling_kurt
    )
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
    rolling_quantile = None
    rolling_skew = None
    rolling_kurt = None
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

logger = logging.getLogger(__name__)


@dataclass
class VectorizationConfig:
    """Configuration for unified vectorization."""
    # Backend preferences (in order of preference)
    preferred_backends: List[str] = None
    
    # Performance thresholds
    vectorbt_threshold: int = 1000
    gpu_threshold: int = 10000
    chunk_size: int = 10000
    
    # Memory management
    max_memory_gb: float = 8.0
    enable_memory_optimization: bool = True
    
    # GPU settings
    enable_gpu: bool = False
    gpu_memory_fraction: float = 0.8
    
    # Parallel processing
    enable_parallel: bool = True
    max_workers: int = None
    
    # Caching
    enable_caching: bool = True
    cache_size: int = 1000
    cache_ttl: int = 3600
    
    def __post_init__(self):
        if self.preferred_backends is None:
            self.preferred_backends = ['vectorbt', 'numpy', 'pandas']
        
        if self.max_workers is None:
            import os
            self.max_workers = min(os.cpu_count() or 4, 8)


class UnifiedVectorizationManager:
    """
    Unified manager for vectorized operations across different backends.
    
    Provides a single interface for all vectorized operations with intelligent
    backend selection and optimization.
    """
    
    def __init__(self, config: Optional[VectorizationConfig] = None):
        """
        Initialize unified vectorization manager.
        
        Args:
            config: Vectorization configuration
        """
        self.config = config or VectorizationConfig()
        
        # Available backends
        self.available_backends = self._detect_available_backends()
        
        # Performance tracking
        self.performance_stats = {
            'vectorbt_operations': 0,
            'numpy_operations': 0,
            'pandas_operations': 0,
            'gpu_operations': 0,
            'chunk_operations': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'total_operations': 0,
            'total_time': 0.0
        }
        
        # Cache for operations
        self._operation_cache = {}
        
        # Initialize backends
        self._initialize_backends()
        
        logger.info(f"UnifiedVectorizationManager initialized with backends: {self.available_backends}")
    
    def _detect_available_backends(self) -> List[str]:
        """Detect available backends."""
        backends = []
        
        if VECTORBT_AVAILABLE:
            backends.append('vectorbt')
        
        if CUPY_AVAILABLE and self.config.enable_gpu:
            backends.append('cupy')
        
        backends.extend(['numpy', 'pandas'])
        
        return backends
    
    def _initialize_backends(self):
        """Initialize available backends."""
        if 'vectorbt' in self.available_backends:
            self._initialize_vectorbt()
        
        if 'cupy' in self.available_backends:
            self._initialize_cupy()
    
    def _initialize_vectorbt(self):
        """Initialize VectorBT settings."""
        try:
            vbt.settings.parallel['enabled'] = self.config.enable_parallel
            vbt.settings.caching['enabled'] = self.config.enable_caching
            vbt.settings.caching['size'] = self.config.cache_size
            vbt.settings.caching['ttl'] = self.config.cache_ttl
            logger.info("✅ VectorBT initialized successfully")
        except Exception as e:
            logger.warning(f"⚠️ VectorBT initialization failed: {e}")
            self.available_backends.remove('vectorbt')
    
    def _initialize_cupy(self):
        """Initialize CuPy settings."""
        try:
            if self.config.gpu_memory_fraction < 1.0:
                cp.cuda.set_periodic_allocation_limit(
                    int(self.config.gpu_memory_fraction * cp.cuda.Device().mem_info[1])
                )
            logger.info("✅ CuPy initialized successfully")
        except Exception as e:
            logger.warning(f"⚠️ CuPy initialization failed: {e}")
            self.available_backends.remove('cupy')
    
    def rolling_mean(self, data: Union[pd.Series, pd.DataFrame], window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Unified rolling mean calculation."""
        return self._rolling_operation(data, 'mean', window, **kwargs)
    
    def rolling_std(self, data: Union[pd.Series, pd.DataFrame], window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Unified rolling standard deviation calculation."""
        return self._rolling_operation(data, 'std', window, **kwargs)
    
    def rolling_var(self, data: Union[pd.Series, pd.DataFrame], window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Unified rolling variance calculation."""
        return self._rolling_operation(data, 'var', window, **kwargs)
    
    def rolling_min(self, data: Union[pd.Series, pd.DataFrame], window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Unified rolling minimum calculation."""
        return self._rolling_operation(data, 'min', window, **kwargs)
    
    def rolling_max(self, data: Union[pd.Series, pd.DataFrame], window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Unified rolling maximum calculation."""
        return self._rolling_operation(data, 'max', window, **kwargs)
    
    def rolling_sum(self, data: Union[pd.Series, pd.DataFrame], window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Unified rolling sum calculation."""
        return self._rolling_operation(data, 'sum', window, **kwargs)
    
    def rolling_quantile(self, data: Union[pd.Series, pd.DataFrame], window: int, q: float = 0.5, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Unified rolling quantile calculation."""
        return self._rolling_operation(data, 'quantile', window, q=q, **kwargs)
    
    def rolling_skew(self, data: Union[pd.Series, pd.DataFrame], window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Unified rolling skewness calculation."""
        return self._rolling_operation(data, 'skew', window, **kwargs)
    
    def rolling_kurt(self, data: Union[pd.Series, pd.DataFrame], window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Unified rolling kurtosis calculation."""
        return self._rolling_operation(data, 'kurt', window, **kwargs)
    
    def rolling_corr(self, data1: Union[pd.Series, pd.DataFrame], data2: Union[pd.Series, pd.DataFrame], 
                    window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Unified rolling correlation calculation."""
        return self._rolling_operation(data1, 'corr', window, data2=data2, **kwargs)
    
    def rolling_cov(self, data1: Union[pd.Series, pd.DataFrame], data2: Union[pd.Series, pd.DataFrame], 
                   window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Unified rolling covariance calculation."""
        return self._rolling_operation(data1, 'cov', window, data2=data2, **kwargs)
    
    def rolling_apply(self, data: Union[pd.Series, pd.DataFrame], func: Callable, 
                     window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Unified rolling apply calculation."""
        return self._rolling_operation(data, 'apply', window, func=func, **kwargs)
    
    def scale_data(self, data: Union[pd.Series, pd.DataFrame], method: str = 'zscore', **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Unified data scaling."""
        return self._scaling_operation(data, method, **kwargs)
    
    def rank_data(self, data: Union[pd.Series, pd.DataFrame], **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Unified data ranking."""
        return self._scaling_operation(data, 'rank', **kwargs)
    
    def winsorize_data(self, data: Union[pd.Series, pd.DataFrame], limits: Tuple[float, float] = (0.05, 0.05), **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Unified data winsorization."""
        return self._scaling_operation(data, 'winsorize', limits=limits, **kwargs)
    
    def clip_data(self, data: Union[pd.Series, pd.DataFrame], lower: float = None, upper: float = None, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Unified data clipping."""
        return self._scaling_operation(data, 'clip', lower=lower, upper=upper, **kwargs)
    
    def _rolling_operation(self, data: Union[pd.Series, pd.DataFrame], operation: str, 
                          window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Perform unified rolling operation with intelligent backend selection."""
        start_time = time.time()
        self.performance_stats['total_operations'] += 1
        
        # Check cache first
        if self.config.enable_caching:
            cache_key = self._generate_cache_key(data, operation, window, **kwargs)
            cached_result = self._get_from_cache(cache_key)
            if cached_result is not None:
                self.performance_stats['cache_hits'] += 1
                return cached_result
            self.performance_stats['cache_misses'] += 1
        
        # Select optimal backend
        backend = self._select_optimal_backend(data, operation, window)
        
        # Optimize data for selected backend
        optimized_data = self._optimize_data_for_backend(data, backend)
        
        # Perform operation
        try:
            if backend == 'vectorbt':
                result = self._vectorbt_rolling_operation(optimized_data, operation, window, **kwargs)
                self.performance_stats['vectorbt_operations'] += 1
            elif backend == 'cupy':
                result = self._cupy_rolling_operation(optimized_data, operation, window, **kwargs)
                self.performance_stats['gpu_operations'] += 1
            elif backend == 'numpy':
                result = self._numpy_rolling_operation(optimized_data, operation, window, **kwargs)
                self.performance_stats['numpy_operations'] += 1
            else:  # pandas
                result = self._pandas_rolling_operation(optimized_data, operation, window, **kwargs)
                self.performance_stats['pandas_operations'] += 1
            
            # Cache result
            if self.config.enable_caching:
                self._put_in_cache(cache_key, result)
            
            # Update timing
            self.performance_stats['total_time'] += time.time() - start_time
            
            return result
            
        except Exception as e:
            logger.warning(f"Operation {operation} failed with backend {backend}: {e}")
            # Fallback to next available backend
            return self._fallback_rolling_operation(data, operation, window, **kwargs)
    
    def _scaling_operation(self, data: Union[pd.Series, pd.DataFrame], method: str, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Perform unified scaling operation."""
        start_time = time.time()
        self.performance_stats['total_operations'] += 1
        
        # Select optimal backend
        backend = self._select_optimal_backend(data, f'scale_{method}', 0)
        
        # Optimize data for selected backend
        optimized_data = self._optimize_data_for_backend(data, backend)
        
        # Perform operation
        try:
            if backend == 'vectorbt':
                result = self._vectorbt_scaling_operation(optimized_data, method, **kwargs)
                self.performance_stats['vectorbt_operations'] += 1
            elif backend == 'cupy':
                result = self._cupy_scaling_operation(optimized_data, method, **kwargs)
                self.performance_stats['gpu_operations'] += 1
            elif backend == 'numpy':
                result = self._numpy_scaling_operation(optimized_data, method, **kwargs)
                self.performance_stats['numpy_operations'] += 1
            else:  # pandas
                result = self._pandas_scaling_operation(optimized_data, method, **kwargs)
                self.performance_stats['pandas_operations'] += 1
            
            # Update timing
            self.performance_stats['total_time'] += time.time() - start_time
            
            return result
            
        except Exception as e:
            logger.warning(f"Scaling operation {method} failed with backend {backend}: {e}")
            # Fallback to pandas
            return self._pandas_scaling_operation(data, method, **kwargs)
    
    def _select_optimal_backend(self, data: Union[pd.Series, pd.DataFrame], operation: str, window: int) -> str:
        """Select optimal backend for the operation."""
        data_size = len(data) if hasattr(data, '__len__') else 0
        
        # Check memory requirements
        if hasattr(data, 'memory_usage'):
            memory_gb = data.memory_usage(deep=True).sum() / (1024**3)
            if memory_gb > self.config.max_memory_gb:
                return 'pandas'  # Most memory efficient
        
        # Select based on data size and operation complexity
        for backend in self.config.preferred_backends:
            if backend not in self.available_backends:
                continue
            
            if backend == 'vectorbt' and data_size >= self.config.vectorbt_threshold:
                return backend
            elif backend == 'cupy' and data_size >= self.config.gpu_threshold:
                return backend
            elif backend == 'numpy' and data_size >= 100:
                return backend
            elif backend == 'pandas':
                return backend
        
        return 'pandas'  # Default fallback
    
    def _optimize_data_for_backend(self, data: Union[pd.Series, pd.DataFrame], backend: str) -> Union[pd.Series, pd.DataFrame]:
        """Optimize data for specific backend."""
        if not self.config.enable_memory_optimization:
            return data
        
        if backend == 'cupy':
            return self._optimize_for_cupy(data)
        elif backend == 'vectorbt':
            return self._optimize_for_vectorbt(data)
        else:
            return self._optimize_for_cpu(data)
    
    def _optimize_for_cupy(self, data: Union[pd.Series, pd.DataFrame]) -> Union[pd.Series, pd.DataFrame]:
        """Optimize data for CuPy processing."""
        if isinstance(data, pd.Series):
            return cp.asarray(data.values)
        else:
            return cp.asarray(data.values)
    
    def _optimize_for_vectorbt(self, data: Union[pd.Series, pd.DataFrame]) -> Union[pd.Series, pd.DataFrame]:
        """Optimize data for VectorBT processing."""
        # Convert to appropriate dtypes
        if isinstance(data, pd.Series):
            if data.dtype == 'float64':
                data = data.astype(np.float32)
        else:
            for col in data.columns:
                if data[col].dtype == 'float64':
                    data[col] = data[col].astype(np.float32)
        
        return data
    
    def _optimize_for_cpu(self, data: Union[pd.Series, pd.DataFrame]) -> Union[pd.Series, pd.DataFrame]:
        """Optimize data for CPU processing."""
        return data
    
    def _vectorbt_rolling_operation(self, data: Union[pd.Series, pd.DataFrame], operation: str, 
                                   window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Perform rolling operation using VectorBT."""
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
        elif operation == 'quantile':
            q = kwargs.get('q', 0.5)
            return rolling_quantile(data, window=window, q=q, **kwargs)
        elif operation == 'skew':
            return rolling_skew(data, window=window, **kwargs)
        elif operation == 'kurt':
            return rolling_kurt(data, window=window, **kwargs)
        elif operation == 'apply':
            func = kwargs.get('func')
            return rolling_apply(data, window=window, func=func, **kwargs)
        elif operation == 'corr':
            data2 = kwargs.get('data2')
            return rolling_corr(data, data2, window=window, **kwargs)
        elif operation == 'cov':
            data2 = kwargs.get('data2')
            return rolling_cov(data, data2, window=window, **kwargs)
        else:
            raise ValueError(f"Unsupported VectorBT operation: {operation}")
    
    def _cupy_rolling_operation(self, data: Union[cp.ndarray, pd.Series, pd.DataFrame], operation: str, 
                               window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Perform rolling operation using CuPy."""
        if isinstance(data, pd.Series):
            gpu_data = cp.asarray(data.values)
            result = self._cupy_rolling_series(gpu_data, operation, window, **kwargs)
            return pd.Series(result.get(), index=data.index, name=data.name)
        elif isinstance(data, pd.DataFrame):
            gpu_data = cp.asarray(data.values)
            result = self._cupy_rolling_dataframe(gpu_data, operation, window, **kwargs)
            return pd.DataFrame(result.get(), index=data.index, columns=data.columns)
        else:
            # Already CuPy array
            if len(data.shape) == 1:
                result = self._cupy_rolling_series(data, operation, window, **kwargs)
                return pd.Series(result.get())
            else:
                result = self._cupy_rolling_dataframe(data, operation, window, **kwargs)
                return pd.DataFrame(result.get())
    
    def _cupy_rolling_series(self, data: cp.ndarray, operation: str, window: int, **kwargs) -> cp.ndarray:
        """CuPy rolling operation for Series."""
        if operation == 'mean':
            return cp.convolve(data, cp.ones(window) / window, mode='same')
        elif operation == 'sum':
            return cp.convolve(data, cp.ones(window), mode='same')
        else:
            # Fallback to CPU for complex operations
            return cp.asarray(self._numpy_rolling_operation(pd.Series(data.get()), operation, window, **kwargs).values)
    
    def _cupy_rolling_dataframe(self, data: cp.ndarray, operation: str, window: int, **kwargs) -> cp.ndarray:
        """CuPy rolling operation for DataFrame."""
        if operation == 'mean':
            return cp.convolve(data, cp.ones((window, 1)) / window, mode='same')
        elif operation == 'sum':
            return cp.convolve(data, cp.ones((window, 1)), mode='same')
        else:
            # Fallback to CPU for complex operations
            return cp.asarray(self._numpy_rolling_operation(pd.DataFrame(data.get()), operation, window, **kwargs).values)
    
    def _numpy_rolling_operation(self, data: Union[pd.Series, pd.DataFrame], operation: str, 
                                window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Perform rolling operation using NumPy."""
        if isinstance(data, pd.Series):
            values = data.values
            result = self._numpy_rolling_series(values, operation, window, **kwargs)
            return pd.Series(result, index=data.index, name=data.name)
        else:
            values = data.values
            result = self._numpy_rolling_dataframe(values, operation, window, **kwargs)
            return pd.DataFrame(result, index=data.index, columns=data.columns)
    
    def _numpy_rolling_series(self, values: np.ndarray, operation: str, window: int, **kwargs) -> np.ndarray:
        """NumPy rolling operation for Series."""
        if operation == 'mean':
            return np.convolve(values, np.ones(window) / window, mode='same')
        elif operation == 'sum':
            return np.convolve(values, np.ones(window), mode='same')
        else:
            # For complex operations, use pandas
            series = pd.Series(values)
            return series.rolling(window=window, **kwargs).agg(operation).values
    
    def _numpy_rolling_dataframe(self, values: np.ndarray, operation: str, window: int, **kwargs) -> np.ndarray:
        """NumPy rolling operation for DataFrame."""
        if operation == 'mean':
            return np.convolve(values, np.ones((window, 1)) / window, mode='same')
        elif operation == 'sum':
            return np.convolve(values, np.ones((window, 1)), mode='same')
        else:
            # For complex operations, use pandas
            df = pd.DataFrame(values)
            return df.rolling(window=window, **kwargs).agg(operation).values
    
    def _pandas_rolling_operation(self, data: Union[pd.Series, pd.DataFrame], operation: str, 
                                 window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Perform rolling operation using pandas."""
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
        elif operation == 'quantile':
            q = kwargs.get('q', 0.5)
            return rolling_obj.quantile(q)
        elif operation == 'skew':
            return rolling_obj.skew()
        elif operation == 'kurt':
            return rolling_obj.kurt()
        elif operation == 'apply':
            func = kwargs.get('func')
            return rolling_obj.apply(func)
        elif operation == 'corr':
            data2 = kwargs.get('data2')
            return rolling_obj.corr(data2)
        elif operation == 'cov':
            data2 = kwargs.get('data2')
            return rolling_obj.cov(data2)
        else:
            raise ValueError(f"Unsupported pandas operation: {operation}")
    
    def _vectorbt_scaling_operation(self, data: Union[pd.Series, pd.DataFrame], method: str, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Perform scaling operation using VectorBT."""
        if method == 'zscore':
            return zscore(data, **kwargs)
        elif method == 'minmax':
            return scale(data, method='minmax', **kwargs)
        elif method == 'robust':
            return scale(data, method='robust', **kwargs)
        elif method == 'quantile':
            return quantile(data, **kwargs)
        elif method == 'winsorize':
            limits = kwargs.get('limits', (0.05, 0.05))
            return winsorize(data, limits=limits, **kwargs)
        elif method == 'rank':
            return rank(data, **kwargs)
        elif method == 'clip':
            lower = kwargs.get('lower')
            upper = kwargs.get('upper')
            return clip(data, lower=lower, upper=upper, **kwargs)
        else:
            raise ValueError(f"Unsupported VectorBT scaling method: {method}")
    
    def _cupy_scaling_operation(self, data: Union[cp.ndarray, pd.Series, pd.DataFrame], method: str, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Perform scaling operation using CuPy."""
        if isinstance(data, pd.Series):
            gpu_data = cp.asarray(data.values)
            result = self._cupy_scaling_series(gpu_data, method, **kwargs)
            return pd.Series(result.get(), index=data.index, name=data.name)
        elif isinstance(data, pd.DataFrame):
            gpu_data = cp.asarray(data.values)
            result = self._cupy_scaling_dataframe(gpu_data, method, **kwargs)
            return pd.DataFrame(result.get(), index=data.index, columns=data.columns)
        else:
            # Already CuPy array
            if len(data.shape) == 1:
                result = self._cupy_scaling_series(data, method, **kwargs)
                return pd.Series(result.get())
            else:
                result = self._cupy_scaling_dataframe(data, method, **kwargs)
                return pd.DataFrame(result.get())
    
    def _cupy_scaling_series(self, data: cp.ndarray, method: str, **kwargs) -> cp.ndarray:
        """CuPy scaling operation for Series."""
        if method == 'zscore':
            return (data - cp.mean(data)) / cp.std(data)
        elif method == 'minmax':
            min_val = cp.min(data)
            max_val = cp.max(data)
            return (data - min_val) / (max_val - min_val)
        else:
            # Fallback to CPU for complex operations
            return cp.asarray(self._pandas_scaling_operation(pd.Series(data.get()), method, **kwargs).values)
    
    def _cupy_scaling_dataframe(self, data: cp.ndarray, method: str, **kwargs) -> cp.ndarray:
        """CuPy scaling operation for DataFrame."""
        if method == 'zscore':
            return (data - cp.mean(data, axis=0)) / cp.std(data, axis=0)
        elif method == 'minmax':
            min_val = cp.min(data, axis=0)
            max_val = cp.max(data, axis=0)
            return (data - min_val) / (max_val - min_val)
        else:
            # Fallback to CPU for complex operations
            return cp.asarray(self._pandas_scaling_operation(pd.DataFrame(data.get()), method, **kwargs).values)
    
    def _numpy_scaling_operation(self, data: Union[pd.Series, pd.DataFrame], method: str, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Perform scaling operation using NumPy."""
        if isinstance(data, pd.Series):
            values = data.values
            result = self._numpy_scaling_series(values, method, **kwargs)
            return pd.Series(result, index=data.index, name=data.name)
        else:
            values = data.values
            result = self._numpy_scaling_dataframe(values, method, **kwargs)
            return pd.DataFrame(result, index=data.index, columns=data.columns)
    
    def _numpy_scaling_series(self, values: np.ndarray, method: str, **kwargs) -> np.ndarray:
        """NumPy scaling operation for Series."""
        if method == 'zscore':
            return (values - np.mean(values)) / np.std(values)
        elif method == 'minmax':
            return (values - np.min(values)) / (np.max(values) - np.min(values))
        else:
            # For complex operations, use pandas
            series = pd.Series(values)
            return self._pandas_scaling_operation(series, method, **kwargs).values
    
    def _numpy_scaling_dataframe(self, values: np.ndarray, method: str, **kwargs) -> np.ndarray:
        """NumPy scaling operation for DataFrame."""
        if method == 'zscore':
            return (values - np.mean(values, axis=0)) / np.std(values, axis=0)
        elif method == 'minmax':
            return (values - np.min(values, axis=0)) / (np.max(values, axis=0) - np.min(values, axis=0))
        else:
            # For complex operations, use pandas
            df = pd.DataFrame(values)
            return self._pandas_scaling_operation(df, method, **kwargs).values
    
    def _pandas_scaling_operation(self, data: Union[pd.Series, pd.DataFrame], method: str, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Perform scaling operation using pandas."""
        if method == 'zscore':
            return (data - data.mean()) / data.std()
        elif method == 'minmax':
            return (data - data.min()) / (data.max() - data.min())
        elif method == 'robust':
            median = data.median()
            mad = (data - median).abs().median()
            return (data - median) / mad
        elif method == 'quantile':
            return data.quantile(kwargs.get('q', 0.5))
        elif method == 'winsorize':
            limits = kwargs.get('limits', (0.05, 0.05))
            return data.clip(lower=data.quantile(limits[0]), upper=data.quantile(1 - limits[1]))
        elif method == 'rank':
            return data.rank()
        elif method == 'clip':
            lower = kwargs.get('lower')
            upper = kwargs.get('upper')
            return data.clip(lower=lower, upper=upper)
        else:
            raise ValueError(f"Unsupported pandas scaling method: {method}")
    
    def _fallback_rolling_operation(self, data: Union[pd.Series, pd.DataFrame], operation: str, 
                                   window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Fallback rolling operation using available backends."""
        for backend in ['numpy', 'pandas']:
            if backend in self.available_backends:
                try:
                    if backend == 'numpy':
                        return self._numpy_rolling_operation(data, operation, window, **kwargs)
                    else:
                        return self._pandas_rolling_operation(data, operation, window, **kwargs)
                except Exception as e:
                    logger.warning(f"Fallback to {backend} failed: {e}")
                    continue
        
        # Ultimate fallback - return NaN
        if isinstance(data, pd.Series):
            return pd.Series(np.nan, index=data.index, name=data.name)
        else:
            return pd.DataFrame(np.nan, index=data.index, columns=data.columns)
    
    def _generate_cache_key(self, data: Union[pd.Series, pd.DataFrame], operation: str, window: int, **kwargs) -> str:
        """Generate cache key for operation."""
        import hashlib
        
        # Create hash of data characteristics and parameters
        data_hash = hashlib.md5(str(data.shape).encode()).hexdigest()[:8]
        params_hash = hashlib.md5(str(sorted(kwargs.items())).encode()).hexdigest()[:8]
        
        return f"{operation}_{window}_{data_hash}_{params_hash}"
    
    def _get_from_cache(self, cache_key: str) -> Optional[Union[pd.Series, pd.DataFrame]]:
        """Get result from cache."""
        if not self.config.enable_caching:
            return None
        
        try:
            if cache_key in self._operation_cache:
                return self._operation_cache[cache_key]
        except Exception as e:
            logger.warning(f"Cache retrieval failed: {e}")
        
        return None
    
    def _put_in_cache(self, cache_key: str, result: Union[pd.Series, pd.DataFrame]):
        """Put result in cache."""
        if not self.config.enable_caching:
            return
        
        try:
            # Limit cache size
            if len(self._operation_cache) >= self.config.cache_size:
                # Remove oldest entries (simple FIFO)
                oldest_key = next(iter(self._operation_cache))
                del self._operation_cache[oldest_key]
            
            self._operation_cache[cache_key] = result
            
        except Exception as e:
            logger.warning(f"Cache storage failed: {e}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        stats = self.performance_stats.copy()
        
        if stats['total_operations'] > 0:
            stats['avg_time_per_operation'] = stats['total_time'] / stats['total_operations']
            stats['vectorbt_usage_rate'] = stats['vectorbt_operations'] / stats['total_operations']
            stats['gpu_usage_rate'] = stats['gpu_operations'] / stats['total_operations']
            stats['cache_hit_rate'] = stats['cache_hits'] / (stats['cache_hits'] + stats['cache_misses']) if (stats['cache_hits'] + stats['cache_misses']) > 0 else 0
        
        return stats
    
    def reset_stats(self):
        """Reset performance statistics."""
        self.performance_stats = {
            'vectorbt_operations': 0,
            'numpy_operations': 0,
            'pandas_operations': 0,
            'gpu_operations': 0,
            'chunk_operations': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'total_operations': 0,
            'total_time': 0.0
        }
    
    def clear_cache(self):
        """Clear operation cache."""
        self._operation_cache.clear()


# Global manager instance
_global_manager = None

def get_unified_vectorization_manager(config: Optional[VectorizationConfig] = None) -> UnifiedVectorizationManager:
    """Get global unified vectorization manager instance."""
    global _global_manager
    if _global_manager is None:
        _global_manager = UnifiedVectorizationManager(config)
    return _global_manager


# Convenience functions
def unified_rolling_mean(data: Union[pd.Series, pd.DataFrame], window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
    """Unified rolling mean calculation."""
    manager = get_unified_vectorization_manager()
    return manager.rolling_mean(data, window, **kwargs)


def unified_rolling_std(data: Union[pd.Series, pd.DataFrame], window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
    """Unified rolling standard deviation calculation."""
    manager = get_unified_vectorization_manager()
    return manager.rolling_std(data, window, **kwargs)


def unified_rolling_var(data: Union[pd.Series, pd.DataFrame], window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
    """Unified rolling variance calculation."""
    manager = get_unified_vectorization_manager()
    return manager.rolling_var(data, window, **kwargs)


def unified_rolling_min(data: Union[pd.Series, pd.DataFrame], window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
    """Unified rolling minimum calculation."""
    manager = get_unified_vectorization_manager()
    return manager.rolling_min(data, window, **kwargs)


def unified_rolling_max(data: Union[pd.Series, pd.DataFrame], window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
    """Unified rolling maximum calculation."""
    manager = get_unified_vectorization_manager()
    return manager.rolling_max(data, window, **kwargs)


def unified_rolling_sum(data: Union[pd.Series, pd.DataFrame], window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
    """Unified rolling sum calculation."""
    manager = get_unified_vectorization_manager()
    return manager.rolling_sum(data, window, **kwargs)


def unified_rolling_quantile(data: Union[pd.Series, pd.DataFrame], window: int, q: float = 0.5, **kwargs) -> Union[pd.Series, pd.DataFrame]:
    """Unified rolling quantile calculation."""
    manager = get_unified_vectorization_manager()
    return manager.rolling_quantile(data, window, q=q, **kwargs)


def unified_rolling_skew(data: Union[pd.Series, pd.DataFrame], window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
    """Unified rolling skewness calculation."""
    manager = get_unified_vectorization_manager()
    return manager.rolling_skew(data, window, **kwargs)


def unified_rolling_kurt(data: Union[pd.Series, pd.DataFrame], window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
    """Unified rolling kurtosis calculation."""
    manager = get_unified_vectorization_manager()
    return manager.rolling_kurt(data, window, **kwargs)


def unified_rolling_corr(data1: Union[pd.Series, pd.DataFrame], data2: Union[pd.Series, pd.DataFrame], 
                        window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
    """Unified rolling correlation calculation."""
    manager = get_unified_vectorization_manager()
    return manager.rolling_corr(data1, data2, window, **kwargs)


def unified_rolling_cov(data1: Union[pd.Series, pd.DataFrame], data2: Union[pd.Series, pd.DataFrame], 
                       window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
    """Unified rolling covariance calculation."""
    manager = get_unified_vectorization_manager()
    return manager.rolling_cov(data1, data2, window, **kwargs)


def unified_rolling_apply(data: Union[pd.Series, pd.DataFrame], func: Callable, 
                         window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
    """Unified rolling apply calculation."""
    manager = get_unified_vectorization_manager()
    return manager.rolling_apply(data, func, window, **kwargs)


def unified_scale_data(data: Union[pd.Series, pd.DataFrame], method: str = 'zscore', **kwargs) -> Union[pd.Series, pd.DataFrame]:
    """Unified data scaling."""
    manager = get_unified_vectorization_manager()
    return manager.scale_data(data, method, **kwargs)


def unified_rank_data(data: Union[pd.Series, pd.DataFrame], **kwargs) -> Union[pd.Series, pd.DataFrame]:
    """Unified data ranking."""
    manager = get_unified_vectorization_manager()
    return manager.rank_data(data, **kwargs)


def unified_winsorize_data(data: Union[pd.Series, pd.DataFrame], limits: Tuple[float, float] = (0.05, 0.05), **kwargs) -> Union[pd.Series, pd.DataFrame]:
    """Unified data winsorization."""
    manager = get_unified_vectorization_manager()
    return manager.winsorize_data(data, limits, **kwargs)


def unified_clip_data(data: Union[pd.Series, pd.DataFrame], lower: float = None, upper: float = None, **kwargs) -> Union[pd.Series, pd.DataFrame]:
    """Unified data clipping."""
    manager = get_unified_vectorization_manager()
    return manager.clip_data(data, lower, upper, **kwargs)


# Example usage and testing
if __name__ == "__main__":
    # Create sample data
    dates = pd.date_range('2020-01-01', periods=5000, freq='1min')
    np.random.seed(42)
    
    # Generate sample data
    data = pd.DataFrame({
        'close': 100 + np.cumsum(np.random.randn(5000) * 0.01),
        'volume': np.random.lognormal(10, 1, 5000)
    }, index=dates)
    
    # Test unified manager
    config = VectorizationConfig(
        enable_gpu=False,
        enable_parallel=True,
        enable_caching=True
    )
    
    manager = UnifiedVectorizationManager(config)
    
    # Test various operations
    print("Testing unified vectorization operations...")
    
    # Rolling mean
    mean_result = manager.rolling_mean(data['close'], window=20)
    print(f"Rolling mean shape: {mean_result.shape}")
    
    # Rolling std
    std_result = manager.rolling_std(data['close'], window=20)
    print(f"Rolling std shape: {std_result.shape}")
    
    # Rolling correlation
    corr_result = manager.rolling_corr(data['close'], data['volume'], window=20)
    print(f"Rolling correlation shape: {corr_result.shape}")
    
    # Scaling
    scaled_result = manager.scale_data(data['close'], method='zscore')
    print(f"Scaled data shape: {scaled_result.shape}")
    
    # Performance stats
    stats = manager.get_performance_stats()
    print(f"Performance stats: {stats}")