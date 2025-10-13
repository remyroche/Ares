"""
VectorBT Rolling Operations Optimizer

This module provides optimized rolling operations using VectorBT's high-performance
functions, with intelligent fallbacks and performance monitoring.

Key Features:
- VectorBT native rolling operations (mean, std, var, min, max, sum, etc.)
- Intelligent fallback to pandas/numpy when VectorBT unavailable
- Performance monitoring and statistics
- Memory-efficient chunked processing
- GPU acceleration support
"""

import numpy as np
import pandas as pd
import logging
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
import warnings
from functools import wraps
import time

# Enhanced logging with tprint
try:
    from src.utils.tprint import (
        tprint, tprint_debug, tprint_info, tprint_warning, tprint_error, 
        tprint_success, tprint_performance, tprint_timer
    )
    TPRINT_AVAILABLE = True
except ImportError:
    TPRINT_AVAILABLE = False
    # Fallback functions for when tprint is not available
    def tprint(*args, **kwargs): print("TPRINT:", *args, **kwargs)
    def tprint_debug(*args, **kwargs): print("DEBUG:", *args, **kwargs)
    def tprint_info(*args, **kwargs): print("INFO:", *args, **kwargs)
    def tprint_warning(*args, **kwargs): print("WARNING:", *args, **kwargs)
    def tprint_error(*args, **kwargs): print("ERROR:", *args, **kwargs)
    def tprint_success(*args, **kwargs): print("SUCCESS:", *args, **kwargs)
    def tprint_performance(*args, **kwargs): print("PERF:", *args, **kwargs)
    def tprint_timer(*args, **kwargs): print("TIMER:", *args, **kwargs)

# VectorBT imports for optimization
try:
    import vectorbt as vbt
    from vectorbt.generic import (
        rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max, 
        rolling_sum, rolling_apply, rolling_corr, rolling_cov,
        rolling_quantile, rolling_skew, rolling_kurt
    )
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
    warnings.warn("VectorBT not available. Install with: pip install vectorbt for optimized performance")

# Optional GPU acceleration
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

logger = logging.getLogger(__name__)

# Enhanced error handling with fast failing
class VectorBTOptimizationError(Exception):
    """Custom exception for VectorBT optimization errors with detailed context."""
    def __init__(self, message: str, operation: str = None, data_shape: tuple = None, 
                 window: int = None, strategy: str = None, original_error: Exception = None):
        self.operation = operation
        self.data_shape = data_shape
        self.window = window
        self.strategy = strategy
        self.original_error = original_error
        
        # Build detailed error message
        context_parts = []
        if operation:
            context_parts.append(f"Operation: {operation}")
        if data_shape:
            context_parts.append(f"Data shape: {data_shape}")
        if window:
            context_parts.append(f"Window: {window}")
        if strategy:
            context_parts.append(f"Strategy: {strategy}")
        
        context_str = ", ".join(context_parts)
        full_message = f"{message}"
        if context_str:
            full_message += f" (Context: {context_str})"
        if original_error:
            full_message += f" (Original: {str(original_error)})"
            
        super().__init__(full_message)

class VectorBTValidationError(Exception):
    """Custom exception for VectorBT validation errors."""
    def __init__(self, message: str, validation_type: str = None, value: Any = None):
        self.validation_type = validation_type
        self.value = value
        full_message = f"{message}"
        if validation_type:
            full_message += f" (Validation: {validation_type})"
        if value is not None:
            full_message += f" (Value: {value})"
        super().__init__(full_message)


class VectorBTRollingOptimizer:
    """
    Optimized rolling operations using VectorBT with intelligent fallbacks.
    
    Provides high-performance rolling calculations with automatic optimization
    selection based on data size and available hardware.
    """
    
    def __init__(self, enable_gpu: bool = False, enable_parallel: bool = True, 
                 memory_efficient: bool = True, chunk_size: int = 1000, 
                 fast_fail: bool = True, enable_logging: bool = True):
        """
        Initialize VectorBT rolling optimizer with enhanced optimization and logging.
        
        Args:
            enable_gpu: Enable GPU acceleration if available
            enable_parallel: Enable parallel processing
            memory_efficient: Enable memory optimization
            chunk_size: Size of data chunks for processing
            fast_fail: Enable fast failing instead of silent fallbacks
            enable_logging: Enable comprehensive logging with tprint
        """
        tprint_info("🚀 Initializing VectorBTRollingOptimizer with enhanced logging and fast failing")
        
        # Validate input parameters
        self._validate_init_parameters(enable_gpu, enable_parallel, memory_efficient, chunk_size)
        
        self.enable_gpu = enable_gpu and CUPY_AVAILABLE
        self.enable_parallel = enable_parallel and VECTORBT_AVAILABLE
        self.memory_efficient = memory_efficient
        self.chunk_size = chunk_size
        self.use_vectorbt = VECTORBT_AVAILABLE
        self.fast_fail = fast_fail
        self.enable_logging = enable_logging
        
        # Enhanced performance tracking with error tracking
        self.performance_stats = {
            'vectorbt_operations': 0,
            'pandas_fallbacks': 0,
            'numpy_fallbacks': 0,
            'gpu_operations': 0,
            'memory_optimizations': 0,
            'chunk_operations': 0,
            'parallel_operations': 0,
            'total_operations': 0,
            'total_time': 0.0,
            'errors': 0,
            'fast_failures': 0,
            'validation_errors': 0
        }
        
        # Configure VectorBT settings with error handling
        try:
            if self.use_vectorbt:
                tprint_info("🔧 Configuring VectorBT settings")
                vbt.settings.parallel['enabled'] = self.enable_parallel
                if self.enable_gpu:
                    vbt.settings.array_wrapper['freq'] = '1min'
                tprint_success("✅ VectorBT settings configured successfully")
            else:
                tprint_warning("⚠️ VectorBT not available, using fallback methods")
        except Exception as e:
            error_msg = f"Failed to configure VectorBT settings: {e}"
            tprint_error(error_msg)
            if self.fast_fail:
                raise VectorBTOptimizationError(error_msg, strategy="initialization", original_error=e)
            else:
                tprint_warning("⚠️ Continuing with fallback configuration")
        
        tprint_success(f"✅ VectorBTRollingOptimizer initialized: VectorBT={self.use_vectorbt}, GPU={self.enable_gpu}, Memory={self.memory_efficient}, FastFail={self.fast_fail}")
        logger.info(f"VectorBTRollingOptimizer initialized: VectorBT={self.use_vectorbt}, GPU={self.enable_gpu}, Memory={self.memory_efficient}")
    
    def rolling_mean(self, data: Union[pd.Series, pd.DataFrame], window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Optimized rolling mean calculation with enhanced logging and validation."""
        tprint_debug(f"🔄 Starting rolling mean calculation: window={window}, data_shape={data.shape if hasattr(data, 'shape') else 'unknown'}")
        
        # Validate inputs
        self._validate_rolling_inputs(data, window, 'mean')
        
        try:
            result = self._rolling_operation(data, 'mean', window, **kwargs)
            tprint_success(f"✅ Rolling mean completed successfully: result_shape={result.shape if hasattr(result, 'shape') else 'unknown'}")
            return result
        except Exception as e:
            error_msg = f"Rolling mean calculation failed"
            tprint_error(f"❌ {error_msg}: {e}")
            self.performance_stats['errors'] += 1
            if self.fast_fail:
                raise VectorBTOptimizationError(error_msg, operation='mean', data_shape=data.shape if hasattr(data, 'shape') else None, window=window, original_error=e)
            else:
                tprint_warning("⚠️ Fast fail disabled, attempting fallback")
                return self._fallback_rolling_mean(data, window, **kwargs)
    
    def rolling_std(self, data: Union[pd.Series, pd.DataFrame], window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Optimized rolling standard deviation calculation with enhanced logging and validation."""
        tprint_debug(f"🔄 Starting rolling std calculation: window={window}, data_shape={data.shape if hasattr(data, 'shape') else 'unknown'}")
        self._validate_rolling_inputs(data, window, 'std')
        try:
            result = self._rolling_operation(data, 'std', window, **kwargs)
            tprint_success(f"✅ Rolling std completed successfully: result_shape={result.shape if hasattr(result, 'shape') else 'unknown'}")
            return result
        except Exception as e:
            error_msg = f"Rolling std calculation failed"
            tprint_error(f"❌ {error_msg}: {e}")
            self.performance_stats['errors'] += 1
            if self.fast_fail:
                raise VectorBTOptimizationError(error_msg, operation='std', data_shape=data.shape if hasattr(data, 'shape') else None, window=window, original_error=e)
            else:
                tprint_warning("⚠️ Fast fail disabled, attempting fallback")
                return self._fallback_rolling_std(data, window, **kwargs)
    
    def rolling_var(self, data: Union[pd.Series, pd.DataFrame], window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Optimized rolling variance calculation with enhanced logging and validation."""
        tprint_debug(f"🔄 Starting rolling var calculation: window={window}, data_shape={data.shape if hasattr(data, 'shape') else 'unknown'}")
        self._validate_rolling_inputs(data, window, 'var')
        try:
            result = self._rolling_operation(data, 'var', window, **kwargs)
            tprint_success(f"✅ Rolling var completed successfully: result_shape={result.shape if hasattr(result, 'shape') else 'unknown'}")
            return result
        except Exception as e:
            error_msg = f"Rolling var calculation failed"
            tprint_error(f"❌ {error_msg}: {e}")
            self.performance_stats['errors'] += 1
            if self.fast_fail:
                raise VectorBTOptimizationError(error_msg, operation='var', data_shape=data.shape if hasattr(data, 'shape') else None, window=window, original_error=e)
            else:
                tprint_warning("⚠️ Fast fail disabled, attempting fallback")
                return self._fallback_rolling_var(data, window, **kwargs)
    
    def rolling_min(self, data: Union[pd.Series, pd.DataFrame], window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Optimized rolling minimum calculation with enhanced logging and validation."""
        tprint_debug(f"🔄 Starting rolling min calculation: window={window}, data_shape={data.shape if hasattr(data, 'shape') else 'unknown'}")
        self._validate_rolling_inputs(data, window, 'min')
        try:
            result = self._rolling_operation(data, 'min', window, **kwargs)
            tprint_success(f"✅ Rolling min completed successfully: result_shape={result.shape if hasattr(result, 'shape') else 'unknown'}")
            return result
        except Exception as e:
            error_msg = f"Rolling min calculation failed"
            tprint_error(f"❌ {error_msg}: {e}")
            self.performance_stats['errors'] += 1
            if self.fast_fail:
                raise VectorBTOptimizationError(error_msg, operation='min', data_shape=data.shape if hasattr(data, 'shape') else None, window=window, original_error=e)
            else:
                tprint_warning("⚠️ Fast fail disabled, attempting fallback")
                return self._fallback_rolling_min(data, window, **kwargs)
    
    def rolling_max(self, data: Union[pd.Series, pd.DataFrame], window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Optimized rolling maximum calculation with enhanced logging and validation."""
        tprint_debug(f"🔄 Starting rolling max calculation: window={window}, data_shape={data.shape if hasattr(data, 'shape') else 'unknown'}")
        self._validate_rolling_inputs(data, window, 'max')
        try:
            result = self._rolling_operation(data, 'max', window, **kwargs)
            tprint_success(f"✅ Rolling max completed successfully: result_shape={result.shape if hasattr(result, 'shape') else 'unknown'}")
            return result
        except Exception as e:
            error_msg = f"Rolling max calculation failed"
            tprint_error(f"❌ {error_msg}: {e}")
            self.performance_stats['errors'] += 1
            if self.fast_fail:
                raise VectorBTOptimizationError(error_msg, operation='max', data_shape=data.shape if hasattr(data, 'shape') else None, window=window, original_error=e)
            else:
                tprint_warning("⚠️ Fast fail disabled, attempting fallback")
                return self._fallback_rolling_max(data, window, **kwargs)
    
    def rolling_sum(self, data: Union[pd.Series, pd.DataFrame], window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Optimized rolling sum calculation with enhanced logging and validation."""
        tprint_debug(f"🔄 Starting rolling sum calculation: window={window}, data_shape={data.shape if hasattr(data, 'shape') else 'unknown'}")
        self._validate_rolling_inputs(data, window, 'sum')
        try:
            result = self._rolling_operation(data, 'sum', window, **kwargs)
            tprint_success(f"✅ Rolling sum completed successfully: result_shape={result.shape if hasattr(result, 'shape') else 'unknown'}")
            return result
        except Exception as e:
            error_msg = f"Rolling sum calculation failed"
            tprint_error(f"❌ {error_msg}: {e}")
            self.performance_stats['errors'] += 1
            if self.fast_fail:
                raise VectorBTOptimizationError(error_msg, operation='sum', data_shape=data.shape if hasattr(data, 'shape') else None, window=window, original_error=e)
            else:
                tprint_warning("⚠️ Fast fail disabled, attempting fallback")
                return self._fallback_rolling_sum(data, window, **kwargs)
    
    def rolling_quantile(self, data: Union[pd.Series, pd.DataFrame], window: int, q: float = 0.5, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Optimized rolling quantile calculation."""
        return self._rolling_operation(data, 'quantile', window, q=q, **kwargs)
    
    def rolling_skew(self, data: Union[pd.Series, pd.DataFrame], window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Optimized rolling skewness calculation."""
        return self._rolling_operation(data, 'skew', window, **kwargs)
    
    def rolling_kurt(self, data: Union[pd.Series, pd.DataFrame], window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Optimized rolling kurtosis calculation."""
        return self._rolling_operation(data, 'kurt', window, **kwargs)
    
    def rolling_corr(self, data: Union[pd.Series, pd.DataFrame], other: Union[pd.Series, pd.DataFrame], 
                    window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Optimized rolling correlation calculation."""
        return self._rolling_operation(data, 'corr', window, other=other, **kwargs)
    
    def rolling_cov(self, data: Union[pd.Series, pd.DataFrame], other: Union[pd.Series, pd.DataFrame], 
                   window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Optimized rolling covariance calculation."""
        return self._rolling_operation(data, 'cov', window, other=other, **kwargs)
    
    def rolling_apply(self, data: Union[pd.Series, pd.DataFrame], func: callable, 
                     window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Optimized rolling apply calculation."""
        return self._rolling_operation(data, 'apply', window, func=func, **kwargs)
    
    def rolling_median(self, data: Union[pd.Series, pd.DataFrame], window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Optimized rolling median calculation."""
        return self.rolling_quantile(data, window, q=0.5, **kwargs)
    
    def rolling_percentile(self, data: Union[pd.Series, pd.DataFrame], window: int, 
                          percentile: float, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Optimized rolling percentile calculation."""
        return self.rolling_quantile(data, window, q=percentile/100, **kwargs)
    
    def rolling_rank(self, data: Union[pd.Series, pd.DataFrame], window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Optimized rolling rank calculation."""
        return self._rolling_operation(data, 'rank', window, **kwargs)
    
    def rolling_ewm(self, data: Union[pd.Series, pd.DataFrame], window: int, 
                   alpha: float = None, span: float = None, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Optimized exponentially weighted moving average."""
        if alpha is not None:
            return data.ewm(alpha=alpha, **kwargs).mean()
        elif span is not None:
            return data.ewm(span=span, **kwargs).mean()
        else:
            return data.ewm(span=window, **kwargs).mean()
    
    def rolling_ewm_std(self, data: Union[pd.Series, pd.DataFrame], window: int, 
                       alpha: float = None, span: float = None, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Optimized exponentially weighted moving standard deviation."""
        if alpha is not None:
            return data.ewm(alpha=alpha, **kwargs).std()
        elif span is not None:
            return data.ewm(span=span, **kwargs).std()
        else:
            return data.ewm(span=window, **kwargs).std()
    
    def rolling_ewm_var(self, data: Union[pd.Series, pd.DataFrame], window: int, 
                       alpha: float = None, span: float = None, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Optimized exponentially weighted moving variance."""
        if alpha is not None:
            return data.ewm(alpha=alpha, **kwargs).var()
        elif span is not None:
            return data.ewm(span=span, **kwargs).var()
        else:
            return data.ewm(span=window, **kwargs).var()
    
    def rolling_correlation_matrix(self, data: pd.DataFrame, window: int, **kwargs) -> pd.DataFrame:
        """Optimized rolling correlation matrix calculation."""
        if not self.use_vectorbt:
            return self._fallback_rolling_correlation_matrix(data, window, **kwargs)
        
        try:
            result = rolling_corr(data, window=window, **kwargs)
            self.performance_stats['vectorbt_operations'] += 1
            return result
        except Exception as e:
            logger.warning(f"VectorBT rolling correlation matrix failed: {e}, using fallback")
            return self._fallback_rolling_correlation_matrix(data, window, **kwargs)
    
    def rolling_covariance_matrix(self, data: pd.DataFrame, window: int, **kwargs) -> pd.DataFrame:
        """Optimized rolling covariance matrix calculation."""
        if not self.use_vectorbt:
            return self._fallback_rolling_covariance_matrix(data, window, **kwargs)
        
        try:
            result = rolling_cov(data, window=window, **kwargs)
            self.performance_stats['vectorbt_operations'] += 1
            return result
        except Exception as e:
            logger.warning(f"VectorBT rolling covariance matrix failed: {e}, using fallback")
            return self._fallback_rolling_covariance_matrix(data, window, **kwargs)
    
    def rolling_quantile(self, data: Union[pd.Series, pd.DataFrame], window: int, q: float = 0.5, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Optimized rolling quantile calculation."""
        return self._rolling_operation(data, 'quantile', window, q=q, **kwargs)
    
    def rolling_skew(self, data: Union[pd.Series, pd.DataFrame], window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Optimized rolling skewness calculation."""
        return self._rolling_operation(data, 'skew', window, **kwargs)
    
    def rolling_kurt(self, data: Union[pd.Series, pd.DataFrame], window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Optimized rolling kurtosis calculation."""
        return self._rolling_operation(data, 'kurt', window, **kwargs)
    
    def rolling_apply(self, data: Union[pd.Series, pd.DataFrame], window: int, func: Callable, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Optimized rolling apply calculation."""
        return self._rolling_operation(data, 'apply', window, func=func, **kwargs)
    
    def rolling_corr(self, data1: Union[pd.Series, pd.DataFrame], data2: Union[pd.Series, pd.DataFrame], 
                    window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Optimized rolling correlation calculation."""
        return self._rolling_operation(data1, 'corr', window, data2=data2, **kwargs)
    
    def rolling_cov(self, data1: Union[pd.Series, pd.DataFrame], data2: Union[pd.Series, pd.DataFrame], 
                   window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Optimized rolling covariance calculation."""
        return self._rolling_operation(data1, 'cov', window, data2=data2, **kwargs)
    
    def _rolling_operation(self, data: Union[pd.Series, pd.DataFrame], operation: str, 
                          window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """
        Perform optimized rolling operation with intelligent method selection, memory optimization, and comprehensive logging.
        
        Args:
            data: Input data (Series or DataFrame)
            operation: Operation to perform ('mean', 'std', 'var', 'min', 'max', 'sum', 'quantile', 'skew', 'kurt', 'apply', 'corr', 'cov')
            window: Rolling window size
            **kwargs: Additional parameters for the operation
            
        Returns:
            Result of the rolling operation
        """
        start_time = time.time()
        self.performance_stats['total_operations'] += 1
        
        tprint_debug(f"🔄 Starting rolling operation: {operation}, window={window}, data_shape={data.shape if hasattr(data, 'shape') else 'unknown'}")
        
        # Validate inputs before processing
        self._validate_rolling_inputs(data, window, operation)
        
        # Optimize data for processing
        if self.memory_efficient:
            tprint_debug("🧠 Optimizing data types for memory efficiency")
            try:
                data = self._optimize_data_types(data)
                tprint_success("✅ Data type optimization completed")
            except Exception as e:
                error_msg = f"Data type optimization failed: {e}"
                tprint_warning(f"⚠️ {error_msg}")
                if self.fast_fail:
                    raise VectorBTOptimizationError(error_msg, operation=operation, original_error=e)
        
        try:
            # Check if data is large enough for chunked processing
            if len(data) > self.chunk_size and self.memory_efficient:
                tprint_info(f"📦 Using chunked processing: data_size={len(data)}, chunk_size={self.chunk_size}")
                result = self._chunked_rolling_operation(data, operation, window, **kwargs)
                self.performance_stats['chunk_operations'] += 1
                tprint_success("✅ Chunked processing completed")
            else:
                # Determine optimal processing method
                strategy = self._select_processing_strategy(data, window, operation)
                tprint_debug(f"🎯 Selected processing strategy: {strategy}")
                
                if strategy == 'vectorbt':
                    result = self._vectorbt_rolling_operation(data, operation, window, **kwargs)
                    self.performance_stats['vectorbt_operations'] += 1
                    tprint_success("✅ VectorBT processing completed")
                elif strategy == 'gpu':
                    result = self._gpu_rolling_operation(data, operation, window, **kwargs)
                    self.performance_stats['gpu_operations'] += 1
                    tprint_success("✅ GPU processing completed")
                else:
                    result = self._pandas_rolling_operation(data, operation, window, **kwargs)
                    self.performance_stats['pandas_fallbacks'] += 1
                    tprint_success("✅ Pandas processing completed")
            
            # Update timing and validate result
            execution_time = time.time() - start_time
            self.performance_stats['total_time'] += execution_time
            
            # Validate result
            self._validate_rolling_result(result, operation, window)
            
            tprint_performance(f"Rolling {operation}", execution_time)
            return result
            
        except Exception as e:
            error_msg = f"Rolling operation {operation} failed"
            tprint_error(f"❌ {error_msg}: {e}")
            self.performance_stats['errors'] += 1
            
            if self.fast_fail:
                self.performance_stats['fast_failures'] += 1
                raise VectorBTOptimizationError(error_msg, operation=operation, data_shape=data.shape if hasattr(data, 'shape') else None, window=window, original_error=e)
            else:
                tprint_warning("⚠️ Fast fail disabled, attempting numpy fallback")
                try:
                    result = self._numpy_rolling_operation(data, operation, window, **kwargs)
                    self.performance_stats['numpy_fallbacks'] += 1
                    tprint_success("✅ Numpy fallback completed")
                    return result
                except Exception as fallback_error:
                    error_msg = f"All rolling operation methods failed for {operation}"
                    tprint_error(f"❌ {error_msg}: {fallback_error}")
                    raise VectorBTOptimizationError(error_msg, operation=operation, data_shape=data.shape if hasattr(data, 'shape') else None, window=window, original_error=fallback_error)
    
    def _chunked_rolling_operation(self, data: Union[pd.Series, pd.DataFrame], operation: str, 
                                 window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Process large data in chunks for memory efficiency."""
        if isinstance(data, pd.Series):
            return self._chunked_series_operation(data, operation, window, **kwargs)
        else:
            return self._chunked_dataframe_operation(data, operation, window, **kwargs)
    
    def _chunked_series_operation(self, data: pd.Series, operation: str, 
                                window: int, **kwargs) -> pd.Series:
        """Process Series in chunks for memory efficiency."""
        results = []
        chunk_size = self.chunk_size
        
        for i in range(0, len(data), chunk_size):
            chunk = data.iloc[i:i + chunk_size + window - 1]  # Include overlap for rolling window
            
            if self._should_use_vectorbt(chunk, window):
                chunk_result = self._vectorbt_rolling_operation(chunk, operation, window, **kwargs)
                self.performance_stats['vectorbt_operations'] += 1
            elif self._should_use_gpu(chunk, window):
                chunk_result = self._gpu_rolling_operation(chunk, operation, window, **kwargs)
                self.performance_stats['gpu_operations'] += 1
            else:
                chunk_result = self._pandas_rolling_operation(chunk, operation, window, **kwargs)
                self.performance_stats['pandas_fallbacks'] += 1
            
            # Remove overlap from result (except for first chunk)
            if i == 0:
                results.append(chunk_result)
            else:
                results.append(chunk_result.iloc[window-1:])
        
        return pd.concat(results, ignore_index=False)
    
    def _chunked_dataframe_operation(self, data: pd.DataFrame, operation: str, 
                                   window: int, **kwargs) -> pd.DataFrame:
        """Process DataFrame in chunks for memory efficiency."""
        results = []
        chunk_size = self.chunk_size
        
        for i in range(0, len(data), chunk_size):
            chunk = data.iloc[i:i + chunk_size + window - 1]  # Include overlap for rolling window
            
            if self._should_use_vectorbt(chunk, window):
                chunk_result = self._vectorbt_rolling_operation(chunk, operation, window, **kwargs)
                self.performance_stats['vectorbt_operations'] += 1
            elif self._should_use_gpu(chunk, window):
                chunk_result = self._gpu_rolling_operation(chunk, operation, window, **kwargs)
                self.performance_stats['gpu_operations'] += 1
            else:
                chunk_result = self._pandas_rolling_operation(chunk, operation, window, **kwargs)
                self.performance_stats['pandas_fallbacks'] += 1
            
            # Remove overlap from result (except for first chunk)
            if i == 0:
                results.append(chunk_result)
            else:
                results.append(chunk_result.iloc[window-1:])
        
        return pd.concat(results, ignore_index=False)
    
    def _optimize_data_types(self, data: Union[pd.Series, pd.DataFrame]) -> Union[pd.Series, pd.DataFrame]:
        """Optimize data types for memory efficiency."""
        if self.memory_efficient:
            if isinstance(data, pd.Series):
                if data.dtype == 'float64':
                    if (data.min() >= np.finfo(np.float32).min and 
                        data.max() <= np.finfo(np.float32).max):
                        data = data.astype(np.float32)
                        self.performance_stats['memory_optimizations'] += 1
            elif isinstance(data, pd.DataFrame):
                optimized_data = data.copy()
                for column in optimized_data.columns:
                    if optimized_data[column].dtype == 'float64':
                        if (optimized_data[column].min() >= np.finfo(np.float32).min and 
                            optimized_data[column].max() <= np.finfo(np.float32).max):
                            optimized_data[column] = optimized_data[column].astype(np.float32)
                            self.performance_stats['memory_optimizations'] += 1
                return optimized_data
        return data
    
    def _should_use_vectorbt(self, data: Union[pd.Series, pd.DataFrame], window: int) -> bool:
        """Determine if VectorBT should be used for this operation."""
        if not self.use_vectorbt:
            return False
        
        # Use VectorBT for larger datasets or when parallel processing is beneficial
        data_size = len(data) if hasattr(data, '__len__') else 0
        return data_size > 1000 or (self.enable_parallel and data_size > 100)
    
    def _should_use_gpu(self, data: Union[pd.Series, pd.DataFrame], window: int) -> bool:
        """Determine if GPU acceleration should be used."""
        if not self.enable_gpu or not CUPY_AVAILABLE:
            return False
        
        # Use GPU for very large datasets
        data_size = len(data) if hasattr(data, '__len__') else 0
        return data_size > 10000
    
    def _vectorbt_rolling_operation(self, data: Union[pd.Series, pd.DataFrame], operation: str, 
                                   window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Perform rolling operation using VectorBT."""
        try:
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
        except Exception as e:
            logger.warning(f"VectorBT {operation} failed: {e}")
            raise
    
    def _gpu_rolling_operation(self, data: Union[pd.Series, pd.DataFrame], operation: str, 
                              window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Perform rolling operation using GPU acceleration."""
        try:
            # Convert to CuPy arrays
            if isinstance(data, pd.Series):
                gpu_data = cp.asarray(data.values)
                result = self._gpu_rolling_series(gpu_data, operation, window, **kwargs)
                return pd.Series(result, index=data.index, name=data.name)
            else:
                gpu_data = cp.asarray(data.values)
                result = self._gpu_rolling_dataframe(gpu_data, operation, window, **kwargs)
                return pd.DataFrame(result, index=data.index, columns=data.columns)
        except Exception as e:
            logger.warning(f"GPU {operation} failed: {e}")
            raise
    
    def _gpu_rolling_series(self, data, operation: str, window: int, **kwargs):
        """GPU rolling operation for Series."""
        if operation == 'mean':
            return cp.convolve(data, cp.ones(window) / window, mode='same')
        elif operation == 'sum':
            return cp.convolve(data, cp.ones(window), mode='same')
        else:
            # Fallback to CPU for complex operations
            return self._numpy_rolling_operation(pd.Series(data.get()), operation, window, **kwargs).values
    
    def _gpu_rolling_dataframe(self, data, operation: str, window: int, **kwargs):
        """GPU rolling operation for DataFrame."""
        if operation == 'mean':
            return cp.convolve(data, cp.ones((window, 1)) / window, mode='same')
        elif operation == 'sum':
            return cp.convolve(data, cp.ones((window, 1)), mode='same')
        else:
            # Fallback to CPU for complex operations
            return self._numpy_rolling_operation(pd.DataFrame(data.get()), operation, window, **kwargs).values
    
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
    
    def _numpy_rolling_operation(self, data: Union[pd.Series, pd.DataFrame], operation: str, 
                                window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Perform rolling operation using numpy (fallback)."""
        if isinstance(data, pd.Series):
            values = data.values
            result = self._numpy_rolling_series(values, operation, window, **kwargs)
            return pd.Series(result, index=data.index, name=data.name)
        else:
            values = data.values
            result = self._numpy_rolling_dataframe(values, operation, window, **kwargs)
            return pd.DataFrame(result, index=data.index, columns=data.columns)
    
    def _numpy_rolling_series(self, values: np.ndarray, operation: str, window: int, **kwargs) -> np.ndarray:
        """Numpy rolling operation for Series."""
        if operation == 'mean':
            return np.convolve(values, np.ones(window) / window, mode='same')
        elif operation == 'sum':
            return np.convolve(values, np.ones(window), mode='same')
        else:
            # For complex operations, use pandas
            series = pd.Series(values)
            return series.rolling(window=window, **kwargs).agg(operation).values
    
    def _numpy_rolling_dataframe(self, values: np.ndarray, operation: str, window: int, **kwargs) -> np.ndarray:
        """Numpy rolling operation for DataFrame."""
        if operation == 'mean':
            return np.convolve(values, np.ones((window, 1)) / window, mode='same')
        elif operation == 'sum':
            return np.convolve(values, np.ones((window, 1)), mode='same')
        else:
            # For complex operations, use pandas
            df = pd.DataFrame(values)
            return df.rolling(window=window, **kwargs).agg(operation).values
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        stats = self.performance_stats.copy()
        if stats['total_operations'] > 0:
            stats['avg_time_per_operation'] = stats['total_time'] / stats['total_operations']
            stats['vectorbt_usage_rate'] = stats['vectorbt_operations'] / stats['total_operations']
            stats['gpu_usage_rate'] = stats['gpu_operations'] / stats['total_operations']
        return stats
    
    def cleanup(self) -> None:
        """Clean up resources and perform memory management."""
        tprint("🧹 Cleaning up VectorBT rolling optimizer resources")
        
        try:
            # Clear any caches or temporary data
            if hasattr(self, '_operation_cache'):
                self._operation_cache.clear()
                tprint("✅ Operation cache cleared")
            
            # Reset performance stats
            self.performance_stats = {
                'vectorbt_operations': 0,
                'pandas_fallbacks': 0,
                'numpy_fallbacks': 0,
                'gpu_operations': 0,
                'memory_optimizations': 0,
                'chunk_operations': 0,
                'parallel_operations': 0,
                'total_operations': 0,
                'total_time': 0.0,
                'errors': 0,
                'fast_failures': 0,
                'validation_errors': 0
            }
            tprint("✅ Performance stats reset")
            
            # Force garbage collection
            import gc
            gc.collect()
            tprint("✅ Garbage collection completed")
            
        except Exception as e:
            tprint_error(f"❌ ERROR: VectorBT rolling optimizer cleanup failed: {e}")
            raise RuntimeError(f"VectorBT rolling optimizer cleanup failed: {e}")
        
        tprint("✅ VectorBT rolling optimizer cleanup completed")
    
    def __enter__(self) -> 'VectorBTRollingOptimizer':
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit with cleanup."""
        self.cleanup()
    
    def _validate_init_parameters(self, enable_gpu: bool, enable_parallel: bool, 
                                 memory_efficient: bool, chunk_size: int):
        """Validate initialization parameters with detailed error reporting."""
        tprint_debug("🔍 Validating initialization parameters")
        
        if not isinstance(enable_gpu, bool):
            raise VectorBTValidationError("enable_gpu must be a boolean", "type_check", enable_gpu)
        
        if not isinstance(enable_parallel, bool):
            raise VectorBTValidationError("enable_parallel must be a boolean", "type_check", enable_parallel)
        
        if not isinstance(memory_efficient, bool):
            raise VectorBTValidationError("memory_efficient must be a boolean", "type_check", memory_efficient)
        
        if not isinstance(chunk_size, int) or chunk_size <= 0:
            raise VectorBTValidationError("chunk_size must be a positive integer", "range_check", chunk_size)
        
        if chunk_size > 1000000:  # 1M rows
            tprint_warning(f"⚠️ Large chunk_size detected: {chunk_size}, this may cause memory issues")
        
        tprint_success("✅ Initialization parameters validated successfully")
    
    def _validate_rolling_inputs(self, data: Union[pd.Series, pd.DataFrame], 
                                window: int, operation: str):
        """Validate rolling operation inputs with comprehensive checks."""
        tprint_debug(f"🔍 Validating rolling inputs for {operation}")
        
        # Check data type
        if not isinstance(data, (pd.Series, pd.DataFrame)):
            raise VectorBTValidationError("Data must be a pandas Series or DataFrame", "type_check", type(data))
        
        # Check data is not empty
        if len(data) == 0:
            raise VectorBTValidationError("Data cannot be empty", "empty_check", len(data))
        
        # Check window size
        if not isinstance(window, int) or window <= 0:
            raise VectorBTValidationError("Window must be a positive integer", "range_check", window)
        
        if window > len(data):
            raise VectorBTValidationError(f"Window size ({window}) cannot be larger than data length ({len(data)})", "range_check", window)
        
        # Check for NaN values in critical columns
        if isinstance(data, pd.DataFrame):
            nan_counts = data.isnull().sum()
            if nan_counts.any():
                tprint_warning(f"⚠️ NaN values detected in data: {nan_counts[nan_counts > 0].to_dict()}")
        
        # Check data types for numeric operations
        if operation in ['mean', 'std', 'var', 'sum', 'quantile', 'skew', 'kurt']:
            if isinstance(data, pd.Series):
                if not pd.api.types.is_numeric_dtype(data):
                    raise VectorBTValidationError("Data must be numeric for this operation", "dtype_check", data.dtype)
            else:  # DataFrame
                non_numeric_cols = data.select_dtypes(exclude=[np.number]).columns
                if len(non_numeric_cols) > 0:
                    raise VectorBTValidationError(f"All columns must be numeric for {operation}, found: {list(non_numeric_cols)}", "dtype_check", list(non_numeric_cols))
        
        tprint_success(f"✅ Rolling inputs validated for {operation}")
    
    def _validate_rolling_result(self, result: Union[pd.Series, pd.DataFrame], 
                                operation: str, window: int):
        """Validate rolling operation result."""
        tprint_debug(f"🔍 Validating rolling result for {operation}")
        
        if result is None:
            raise VectorBTValidationError("Result cannot be None", "null_check", result)
        
        # Check result type matches input type
        if not isinstance(result, (pd.Series, pd.DataFrame)):
            raise VectorBTValidationError("Result must be a pandas Series or DataFrame", "type_check", type(result))
        
        # Check for infinite values
        if isinstance(result, pd.Series):
            if np.isinf(result).any():
                tprint_warning(f"⚠️ Infinite values detected in {operation} result")
        else:  # DataFrame
            inf_counts = np.isinf(result).sum()
            if inf_counts.any():
                tprint_warning(f"⚠️ Infinite values detected in {operation} result: {inf_counts[inf_counts > 0].to_dict()}")
        
        tprint_success(f"✅ Rolling result validated for {operation}")
    
    def _select_processing_strategy(self, data: Union[pd.Series, pd.DataFrame], 
                                   window: int, operation: str) -> str:
        """Select optimal processing strategy with detailed logging."""
        tprint_debug(f"🎯 Selecting processing strategy for {operation}")
        
        # VectorBT strategy
        if self._should_use_vectorbt(data, window):
            tprint_debug("✅ Selected VectorBT strategy")
            return 'vectorbt'
        
        # GPU strategy
        if self._should_use_gpu(data, window):
            tprint_debug("✅ Selected GPU strategy")
            return 'gpu'
        
        # Pandas fallback
        tprint_debug("✅ Selected Pandas strategy (fallback)")
        return 'pandas'
    
    def _fallback_rolling_mean(self, data: Union[pd.Series, pd.DataFrame], 
                              window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Fallback rolling mean with error handling."""
        tprint_warning("⚠️ Using fallback rolling mean implementation")
        try:
            return data.rolling(window=window, **kwargs).mean()
        except Exception as e:
            error_msg = f"Fallback rolling mean failed: {e}"
            tprint_error(f"❌ {error_msg}")
            raise VectorBTOptimizationError(error_msg, operation='mean', original_error=e)
    
    def _fallback_rolling_std(self, data: Union[pd.Series, pd.DataFrame], 
                             window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Fallback rolling std with error handling."""
        tprint_warning("⚠️ Using fallback rolling std implementation")
        try:
            return data.rolling(window=window, **kwargs).std()
        except Exception as e:
            error_msg = f"Fallback rolling std failed: {e}"
            tprint_error(f"❌ {error_msg}")
            raise VectorBTOptimizationError(error_msg, operation='std', original_error=e)
    
    def _fallback_rolling_var(self, data: Union[pd.Series, pd.DataFrame], 
                             window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Fallback rolling var with error handling."""
        tprint_warning("⚠️ Using fallback rolling var implementation")
        try:
            return data.rolling(window=window, **kwargs).var()
        except Exception as e:
            error_msg = f"Fallback rolling var failed: {e}"
            tprint_error(f"❌ {error_msg}")
            raise VectorBTOptimizationError(error_msg, operation='var', original_error=e)
    
    def _fallback_rolling_min(self, data: Union[pd.Series, pd.DataFrame], 
                             window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Fallback rolling min with error handling."""
        tprint_warning("⚠️ Using fallback rolling min implementation")
        try:
            return data.rolling(window=window, **kwargs).min()
        except Exception as e:
            error_msg = f"Fallback rolling min failed: {e}"
            tprint_error(f"❌ {error_msg}")
            raise VectorBTOptimizationError(error_msg, operation='min', original_error=e)
    
    def _fallback_rolling_max(self, data: Union[pd.Series, pd.DataFrame], 
                             window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Fallback rolling max with error handling."""
        tprint_warning("⚠️ Using fallback rolling max implementation")
        try:
            return data.rolling(window=window, **kwargs).max()
        except Exception as e:
            error_msg = f"Fallback rolling max failed: {e}"
            tprint_error(f"❌ {error_msg}")
            raise VectorBTOptimizationError(error_msg, operation='max', original_error=e)
    
    def _fallback_rolling_sum(self, data: Union[pd.Series, pd.DataFrame], 
                             window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Fallback rolling sum with error handling."""
        tprint_warning("⚠️ Using fallback rolling sum implementation")
        try:
            return data.rolling(window=window, **kwargs).sum()
        except Exception as e:
            error_msg = f"Fallback rolling sum failed: {e}"
            tprint_error(f"❌ {error_msg}")
            raise VectorBTOptimizationError(error_msg, operation='sum', original_error=e)

    def reset_stats(self):
        """Reset performance statistics."""
        tprint_info("🔄 Resetting performance statistics")
        self.performance_stats = {
            'vectorbt_operations': 0,
            'pandas_fallbacks': 0,
            'numpy_fallbacks': 0,
            'gpu_operations': 0,
            'memory_optimizations': 0,
            'chunk_operations': 0,
            'parallel_operations': 0,
            'total_operations': 0,
            'total_time': 0.0,
            'errors': 0,
            'fast_failures': 0,
            'validation_errors': 0
        }
        tprint_success("✅ Performance statistics reset")


# Global optimizer instance
_global_optimizer = None

def get_vectorbt_rolling_optimizer(enable_gpu: bool = False, enable_parallel: bool = True, 
                                 memory_efficient: bool = True, chunk_size: int = 1000,
                                 fast_fail: bool = True, enable_logging: bool = True) -> VectorBTRollingOptimizer:
    """Get global VectorBT rolling optimizer instance."""
    global _global_optimizer
    if _global_optimizer is None:
        _global_optimizer = VectorBTRollingOptimizer(
            enable_gpu=enable_gpu, 
            enable_parallel=enable_parallel,
            memory_efficient=memory_efficient,
            chunk_size=chunk_size,
            fast_fail=fast_fail,
            enable_logging=enable_logging
        )
    return _global_optimizer


def optimized_rolling_mean(data: Union[pd.Series, pd.DataFrame], window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
    """Optimized rolling mean using VectorBT."""
    optimizer = get_vectorbt_rolling_optimizer()
    return optimizer.rolling_mean(data, window, **kwargs)


def optimized_rolling_std(data: Union[pd.Series, pd.DataFrame], window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
    """Optimized rolling standard deviation using VectorBT."""
    optimizer = get_vectorbt_rolling_optimizer()
    return optimizer.rolling_std(data, window, **kwargs)


def optimized_rolling_var(data: Union[pd.Series, pd.DataFrame], window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
    """Optimized rolling variance using VectorBT."""
    optimizer = get_vectorbt_rolling_optimizer()
    return optimizer.rolling_var(data, window, **kwargs)


def optimized_rolling_min(data: Union[pd.Series, pd.DataFrame], window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
    """Optimized rolling minimum using VectorBT."""
    optimizer = get_vectorbt_rolling_optimizer()
    return optimizer.rolling_min(data, window, **kwargs)


def optimized_rolling_max(data: Union[pd.Series, pd.DataFrame], window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
    """Optimized rolling maximum using VectorBT."""
    optimizer = get_vectorbt_rolling_optimizer()
    return optimizer.rolling_max(data, window, **kwargs)


def optimized_rolling_sum(data: Union[pd.Series, pd.DataFrame], window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
    """Optimized rolling sum using VectorBT."""
    optimizer = get_vectorbt_rolling_optimizer()
    return optimizer.rolling_sum(data, window, **kwargs)


def optimized_rolling_quantile(data: Union[pd.Series, pd.DataFrame], window: int, q: float = 0.5, **kwargs) -> Union[pd.Series, pd.DataFrame]:
    """Optimized rolling quantile using VectorBT."""
    optimizer = get_vectorbt_rolling_optimizer()
    return optimizer.rolling_quantile(data, window, q=q, **kwargs)


def optimized_rolling_apply(data: Union[pd.Series, pd.DataFrame], window: int, func: Callable, **kwargs) -> Union[pd.Series, pd.DataFrame]:
    """Optimized rolling apply using VectorBT."""
    optimizer = get_vectorbt_rolling_optimizer()
    return optimizer.rolling_apply(data, window, func, **kwargs)


def optimized_rolling_corr(data1: Union[pd.Series, pd.DataFrame], data2: Union[pd.Series, pd.DataFrame], 
                          window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
    """Optimized rolling correlation using VectorBT."""
    optimizer = get_vectorbt_rolling_optimizer()
    return optimizer.rolling_corr(data1, data2, window, **kwargs)


def optimized_rolling_cov(data1: Union[pd.Series, pd.DataFrame], data2: Union[pd.Series, pd.DataFrame], 
                         window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
    """Optimized rolling covariance using VectorBT."""
    optimizer = get_vectorbt_rolling_optimizer()
    return optimizer.rolling_cov(data1, data2, window, **kwargs)


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
    
    # Test optimizer
    optimizer = VectorBTRollingOptimizer(enable_gpu=False, enable_parallel=True)
    
    # Test various operations
    print("Testing VectorBT rolling operations...")
    
    # Rolling mean
    mean_result = optimizer.rolling_mean(data['close'], window=20)
    print(f"Rolling mean shape: {mean_result.shape}")
    
    # Rolling std
    std_result = optimizer.rolling_std(data['close'], window=20)
    print(f"Rolling std shape: {std_result.shape}")
    
    # Rolling correlation
    corr_result = optimizer.rolling_corr(data['close'], data['volume'], window=20)
    print(f"Rolling correlation shape: {corr_result.shape}")
    
    # Performance stats
    stats = optimizer.get_performance_stats()
    print(f"Performance stats: {stats}")


def optimized_rolling_correlation_matrix(data: pd.DataFrame, window: int, **kwargs) -> pd.DataFrame:
    """Optimized rolling correlation matrix using VectorBT."""
    optimizer = get_vectorbt_rolling_optimizer()
    return optimizer.rolling_correlation_matrix(data, window, **kwargs)


def optimized_rolling_covariance_matrix(data: pd.DataFrame, window: int, **kwargs) -> pd.DataFrame:
    """Optimized rolling covariance matrix using VectorBT."""
    optimizer = get_vectorbt_rolling_optimizer()
    return optimizer.rolling_covariance_matrix(data, window, **kwargs)