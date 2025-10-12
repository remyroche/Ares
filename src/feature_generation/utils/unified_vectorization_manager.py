"""
Unified Vectorization Manager

This module provides a centralized vectorization management system that unifies
VectorBT optimizations, rolling operations, and batch processing for maximum
performance in feature generation.

Key Features:
- Unified interface for all vectorization operations
- VectorBTRollingOptimizer integration
- VectorBTBatchProcessor integration
- Memory-efficient processing
- Performance monitoring and statistics
- GPU acceleration support
- Parallel processing capabilities
"""

import numpy as np
import pandas as pd
import logging
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
import time
from contextlib import contextmanager
from dataclasses import dataclass
import warnings

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
    warnings.warn("VectorBT not available. Install with: pip install vectorbt for optimized performance")

# Optional GPU acceleration
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

# Import our optimization modules
try:
    from .vectorbt_rolling_optimizer import VectorBTRollingOptimizer, get_vectorbt_rolling_optimizer
except ImportError:
    # Fallback for direct import
    from vectorbt_rolling_optimizer import VectorBTRollingOptimizer, get_vectorbt_rolling_optimizer

try:
    from ..core.vectorbt_batch_processor import VectorBTBatchProcessor, BatchProcessingConfig
except ImportError:
    # Fallback for direct import
    VectorBTBatchProcessor = None
    BatchProcessingConfig = None

logger = logging.getLogger(__name__)

# Enhanced error handling with fast failing
class UnifiedVectorizationError(Exception):
    """Custom exception for unified vectorization errors with detailed context."""
    def __init__(self, message: str, operation: str = None, data_shape: tuple = None, 
                 config: str = None, original_error: Exception = None):
        self.operation = operation
        self.data_shape = data_shape
        self.config = config
        self.original_error = original_error
        
        # Build detailed error message
        context_parts = []
        if operation:
            context_parts.append(f"Operation: {operation}")
        if data_shape:
            context_parts.append(f"Data shape: {data_shape}")
        if config:
            context_parts.append(f"Config: {config}")
        
        context_str = ", ".join(context_parts)
        full_message = f"{message}"
        if context_str:
            full_message += f" (Context: {context_str})"
        if original_error:
            full_message += f" (Original: {str(original_error)})"
            
        super().__init__(full_message)

class VectorizationValidationError(Exception):
    """Custom exception for vectorization validation errors."""
    def __init__(self, message: str, validation_type: str = None, value: Any = None):
        self.validation_type = validation_type
        self.value = value
        full_message = f"{message}"
        if validation_type:
            full_message += f" (Validation: {validation_type})"
        if value is not None:
            full_message += f" (Value: {value})"
        super().__init__(full_message)


@dataclass
class VectorizationConfig:
    """Configuration for unified vectorization."""
    # VectorBT settings
    enable_vectorbt: bool = True
    enable_gpu: bool = False
    enable_parallel: bool = True
    
    # Memory management
    memory_efficient: bool = True
    max_memory_gb: float = 8.0
    chunk_size: int = 1000
    
    # Performance monitoring
    enable_monitoring: bool = True
    enable_profiling: bool = False
    
    # Batch processing
    batch_size: int = 10000
    enable_batch_processing: bool = True
    
    # Rolling operations
    rolling_optimization_threshold: int = 1000
    enable_rolling_optimization: bool = True
    
    def __post_init__(self):
        if not VECTORBT_AVAILABLE:
            self.enable_vectorbt = False
            logger.warning("VectorBT not available, disabling vectorization optimizations")
        
        if self.enable_gpu and not CUPY_AVAILABLE:
            self.enable_gpu = False
            logger.warning("GPU acceleration requested but CuPy not available")


class UnifiedVectorizationManager:
    """
    Unified manager for all vectorization operations using VectorBT optimizations.
    
    This class provides a single interface for:
    - VectorBT rolling operations
    - Batch processing
    - Memory optimization
    - Performance monitoring
    - GPU acceleration
    - Parallel processing
    """
    
    def __init__(self, config: Optional[VectorizationConfig] = None, 
                 fast_fail: bool = True, enable_logging: bool = True):
        """
        Initialize unified vectorization manager with enhanced logging and fast failing.
        
        Args:
            config: Vectorization configuration
            fast_fail: Enable fast failing instead of silent fallbacks
            enable_logging: Enable comprehensive logging with tprint
        """
        tprint_info("🚀 Initializing UnifiedVectorizationManager with enhanced logging and fast failing")
        
        self.config = config or VectorizationConfig()
        self.fast_fail = fast_fail
        self.enable_logging = enable_logging
        
        # Validate configuration
        self._validate_config(self.config)
        
        # Initialize components with error handling
        tprint_info("🔧 Initializing vectorization components")
        try:
            self.rolling_optimizer = get_vectorbt_rolling_optimizer(
                enable_gpu=self.config.enable_gpu,
                enable_parallel=self.config.enable_parallel,
                memory_efficient=self.config.memory_efficient,
                chunk_size=self.config.chunk_size,
                fast_fail=self.fast_fail,
                enable_logging=self.enable_logging
            )
            tprint_success("✅ Rolling optimizer initialized")
        except Exception as e:
            error_msg = f"Failed to initialize rolling optimizer: {e}"
            tprint_error(f"❌ {error_msg}")
            if self.fast_fail:
                raise UnifiedVectorizationError(error_msg, operation="initialization", original_error=e)
            else:
                tprint_warning("⚠️ Fast fail disabled, continuing without rolling optimizer")
                self.rolling_optimizer = None
        
        # Initialize batch processor with error handling
        if VectorBTBatchProcessor is not None and BatchProcessingConfig is not None:
            try:
                batch_config = BatchProcessingConfig(
                    batch_size=self.config.batch_size,
                    enable_gpu=self.config.enable_gpu,
                    enable_parallel=self.config.enable_parallel,
                    max_memory_gb=self.config.max_memory_gb,
                    chunk_size=self.config.chunk_size,
                    enable_memory_optimization=self.config.memory_efficient,
                    enable_progress_tracking=self.config.enable_monitoring
                )
                self.batch_processor = VectorBTBatchProcessor(batch_config)
                tprint_success("✅ Batch processor initialized")
            except Exception as e:
                error_msg = f"Failed to initialize batch processor: {e}"
                tprint_error(f"❌ {error_msg}")
                if self.fast_fail:
                    raise UnifiedVectorizationError(error_msg, operation="initialization", original_error=e)
                else:
                    tprint_warning("⚠️ Fast fail disabled, continuing without batch processor")
                    self.batch_processor = None
        else:
            tprint_warning("⚠️ VectorBTBatchProcessor not available, continuing without batch processor")
            self.batch_processor = None
        
        # Enhanced performance tracking with error tracking
        self.performance_stats = {
            'total_operations': 0,
            'vectorbt_operations': 0,
            'pandas_fallbacks': 0,
            'gpu_operations': 0,
            'batch_operations': 0,
            'rolling_operations': 0,
            'scaling_operations': 0,
            'memory_optimizations': 0,
            'total_time': 0.0,
            'memory_savings': 0.0,
            'cache_hits': 0,
            'cache_misses': 0,
            'errors': 0,
            'fast_failures': 0,
            'validation_errors': 0
        }
        
        # Cache for computed results
        self._result_cache = {}
        self._cache_enabled = True
        self._max_cache_size = 1000
        
        tprint_success(f"✅ UnifiedVectorizationManager initialized: VectorBT={self.config.enable_vectorbt}, GPU={self.config.enable_gpu}, Memory={self.config.memory_efficient}, FastFail={self.fast_fail}")
        logger.info(f"UnifiedVectorizationManager initialized: VectorBT={self.config.enable_vectorbt}, GPU={self.config.enable_gpu}, Memory={self.config.memory_efficient}")
    
    def rolling_operation(self, data: Union[pd.Series, pd.DataFrame], 
                         operation: str, window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """
        Perform optimized rolling operation with enhanced logging and validation.
        
        Args:
            data: Input data
            operation: Operation type ('mean', 'std', 'var', 'min', 'max', 'sum', etc.)
            window: Rolling window size
            **kwargs: Additional parameters
            
        Returns:
            Result of rolling operation
        """
        start_time = time.time()
        self.performance_stats['total_operations'] += 1
        self.performance_stats['rolling_operations'] += 1
        
        tprint_debug(f"🔄 Starting rolling operation: {operation}, window={window}, data_shape={data.shape if hasattr(data, 'shape') else 'unknown'}")
        
        # Validate inputs
        self._validate_rolling_inputs(data, operation, window)
        
        # Check cache first
        if self._cache_enabled:
            cache_key = self._generate_cache_key(data, operation, window, **kwargs)
            cached_result = self._get_from_cache(cache_key)
            if cached_result is not None:
                self.performance_stats['cache_hits'] += 1
                tprint_debug("💾 Cache hit for rolling operation")
                return cached_result
            self.performance_stats['cache_misses'] += 1
            tprint_debug("💾 Cache miss for rolling operation")
        
        # Check if rolling optimizer is available
        if self.rolling_optimizer is None:
            error_msg = "Rolling optimizer not available"
            tprint_error(f"❌ {error_msg}")
            if self.fast_fail:
                raise UnifiedVectorizationError(error_msg, operation=operation, data_shape=data.shape if hasattr(data, 'shape') else None)
            else:
                tprint_warning("⚠️ Fast fail disabled, using pandas fallback")
                return self._pandas_fallback_rolling(data, operation, window, **kwargs)
        
        try:
            # Use VectorBT rolling optimizer with detailed logging
            tprint_debug(f"🎯 Executing rolling {operation} with VectorBT optimizer")
            
            if operation == 'mean':
                result = self.rolling_optimizer.rolling_mean(data, window, **kwargs)
            elif operation == 'std':
                result = self.rolling_optimizer.rolling_std(data, window, **kwargs)
            elif operation == 'var':
                result = self.rolling_optimizer.rolling_var(data, window, **kwargs)
            elif operation == 'min':
                result = self.rolling_optimizer.rolling_min(data, window, **kwargs)
            elif operation == 'max':
                result = self.rolling_optimizer.rolling_max(data, window, **kwargs)
            elif operation == 'sum':
                result = self.rolling_optimizer.rolling_sum(data, window, **kwargs)
            elif operation == 'quantile':
                q = kwargs.pop('q', 0.5)
                result = self.rolling_optimizer.rolling_quantile(data, window, q=q, **kwargs)
            elif operation == 'skew':
                result = self.rolling_optimizer.rolling_skew(data, window, **kwargs)
            elif operation == 'kurt':
                result = self.rolling_optimizer.rolling_kurt(data, window, **kwargs)
            elif operation == 'corr':
                other = kwargs.pop('other', None)
                result = self.rolling_optimizer.rolling_corr(data, other, window, **kwargs)
            elif operation == 'cov':
                other = kwargs.pop('other', None)
                result = self.rolling_optimizer.rolling_cov(data, other, window, **kwargs)
            elif operation == 'apply':
                func = kwargs.pop('func', None)
                result = self.rolling_optimizer.rolling_apply(data, func, window, **kwargs)
            else:
                error_msg = f"Unsupported rolling operation: {operation}"
                tprint_error(f"❌ {error_msg}")
                if self.fast_fail:
                    raise UnifiedVectorizationError(error_msg, operation=operation)
                else:
                    tprint_warning("⚠️ Fast fail disabled, using pandas fallback")
                    return self._pandas_fallback_rolling(data, operation, window, **kwargs)
            
            # Update stats
            rolling_stats = self.rolling_optimizer.get_performance_stats()
            if rolling_stats.get('vectorbt_operations', 0) > 0:
                self.performance_stats['vectorbt_operations'] += 1
            if rolling_stats.get('gpu_operations', 0) > 0:
                self.performance_stats['gpu_operations'] += 1
            if rolling_stats.get('memory_optimizations', 0) > 0:
                self.performance_stats['memory_optimizations'] += 1
            
            # Cache result
            if self._cache_enabled:
                self._put_in_cache(cache_key, result)
                tprint_debug("💾 Result cached successfully")
            
            tprint_success(f"✅ Rolling {operation} completed successfully")
            return result
            
        except Exception as e:
            error_msg = f"Rolling operation {operation} failed"
            tprint_error(f"❌ {error_msg}: {e}")
            self.performance_stats['errors'] += 1
            
            if self.fast_fail:
                self.performance_stats['fast_failures'] += 1
                raise UnifiedVectorizationError(error_msg, operation=operation, data_shape=data.shape if hasattr(data, 'shape') else None, original_error=e)
            else:
                tprint_warning("⚠️ Fast fail disabled, using pandas fallback")
                self.performance_stats['pandas_fallbacks'] += 1
                return self._pandas_fallback_rolling(data, operation, window, **kwargs)
        
        finally:
            execution_time = time.time() - start_time
            self.performance_stats['total_time'] += execution_time
            tprint_performance(f"Rolling {operation}", execution_time)
    
    def scale_data(self, data: Union[pd.Series, pd.DataFrame], 
                   method: str = 'zscore', **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """
        Scale data using VectorBT scaling functions with enhanced logging and validation.
        
        Args:
            data: Input data
            method: Scaling method ('zscore', 'minmax', 'robust', 'quantile', 'winsorize')
            **kwargs: Additional parameters
            
        Returns:
            Scaled data
        """
        start_time = time.time()
        self.performance_stats['total_operations'] += 1
        self.performance_stats['scaling_operations'] += 1
        
        tprint_debug(f"🔄 Starting data scaling: method={method}, data_shape={data.shape if hasattr(data, 'shape') else 'unknown'}")
        
        # Validate inputs
        self._validate_scaling_inputs(data, method)
        
        if not VECTORBT_AVAILABLE or not self.config.enable_vectorbt:
            tprint_warning("⚠️ VectorBT not available, using pandas fallback for scaling")
            return self._pandas_fallback_scaling(data, method, **kwargs)
        
        try:
            tprint_debug(f"🎯 Executing {method} scaling with VectorBT")
            
            if method == 'zscore':
                result = zscore(data, **kwargs)
            elif method == 'minmax':
                result = scale(data, method='minmax', **kwargs)
            elif method == 'robust':
                result = scale(data, method='robust', **kwargs)
            elif method == 'quantile':
                result = quantile(data, **kwargs)
            elif method == 'winsorize':
                result = winsorize(data, **kwargs)
            elif method == 'rank':
                result = rank(data, **kwargs)
            elif method == 'clip':
                result = clip(data, **kwargs)
            else:
                error_msg = f"Unsupported scaling method: {method}"
                tprint_error(f"❌ {error_msg}")
                if self.fast_fail:
                    raise UnifiedVectorizationError(error_msg, operation="scaling", original_error=ValueError(error_msg))
                else:
                    tprint_warning("⚠️ Fast fail disabled, using pandas fallback")
                    return self._pandas_fallback_scaling(data, method, **kwargs)
            
            self.performance_stats['vectorbt_operations'] += 1
            tprint_success(f"✅ {method} scaling completed successfully")
            return result
            
        except Exception as e:
            error_msg = f"VectorBT scaling failed for {method}"
            tprint_error(f"❌ {error_msg}: {e}")
            self.performance_stats['errors'] += 1
            
            if self.fast_fail:
                self.performance_stats['fast_failures'] += 1
                raise UnifiedVectorizationError(error_msg, operation="scaling", data_shape=data.shape if hasattr(data, 'shape') else None, original_error=e)
            else:
                tprint_warning("⚠️ Fast fail disabled, using pandas fallback")
                return self._pandas_fallback_scaling(data, method, **kwargs)
        
        finally:
            execution_time = time.time() - start_time
            self.performance_stats['total_time'] += execution_time
            tprint_performance(f"Scaling {method}", execution_time)
    
    def batch_process_features(self, data: pd.DataFrame, 
                             feature_configs: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Process multiple features in batch with optimization and enhanced logging.
        
        Args:
            data: Input OHLCV data
            feature_configs: List of feature configuration dictionaries
            
        Returns:
            DataFrame with generated features
        """
        start_time = time.time()
        self.performance_stats['total_operations'] += 1
        self.performance_stats['batch_operations'] += 1
        
        tprint_info(f"🔄 Starting batch feature processing: {len(feature_configs)} features, data_shape={data.shape}")
        
        # Validate inputs
        if not isinstance(data, pd.DataFrame):
            error_msg = "Data must be a pandas DataFrame"
            tprint_error(f"❌ {error_msg}")
            if self.fast_fail:
                raise UnifiedVectorizationError(error_msg, operation="batch_processing", data_shape=data.shape if hasattr(data, 'shape') else None)
            else:
                tprint_warning("⚠️ Fast fail disabled, returning empty DataFrame")
                return pd.DataFrame()
        
        if not isinstance(feature_configs, list) or len(feature_configs) == 0:
            error_msg = "feature_configs must be a non-empty list"
            tprint_error(f"❌ {error_msg}")
            if self.fast_fail:
                raise UnifiedVectorizationError(error_msg, operation="batch_processing")
            else:
                tprint_warning("⚠️ Fast fail disabled, returning empty DataFrame")
                return pd.DataFrame()
        
        try:
            # Use VectorBT batch processor
            results = {}
            successful_features = 0
            failed_features = 0
            
            tprint_debug(f"🎯 Processing {len(feature_configs)} features")
            
            for i, config in enumerate(feature_configs):
                feature_name = config.get('name', f'feature_{i}')
                feature_type = config.get('type', 'rolling')
                params = config.get('params', {})
                
                tprint_debug(f"🔄 Processing feature {i+1}/{len(feature_configs)}: {feature_name} ({feature_type})")
                
                try:
                    if feature_type == 'rolling':
                        operation = params.get('operation', 'mean')
                        window = params.get('window', 20)
                        column = params.get('column', 'close')
                        
                        if column not in data.columns:
                            error_msg = f"Column '{column}' not found in data. Available: {list(data.columns)}"
                            tprint_error(f"❌ {error_msg}")
                            if self.fast_fail:
                                raise UnifiedVectorizationError(error_msg, operation="batch_processing", data_shape=data.shape)
                            else:
                                tprint_warning("⚠️ Fast fail disabled, skipping feature")
                                results[feature_name] = pd.Series(np.nan, index=data.index)
                                failed_features += 1
                                continue
                        
                        # Remove operation and window from params to avoid conflicts
                        rolling_params = {k: v for k, v in params.items() if k not in ['operation', 'window', 'column']}
                        results[feature_name] = self.rolling_operation(
                            data[column], operation, window, **rolling_params
                        )
                        successful_features += 1
                        tprint_success(f"✅ Feature {feature_name} completed successfully")
                    
                    elif feature_type == 'scaling':
                        method = params.get('method', 'zscore')
                        column = params.get('column', 'close')
                        
                        if column not in data.columns:
                            error_msg = f"Column '{column}' not found in data. Available: {list(data.columns)}"
                            tprint_error(f"❌ {error_msg}")
                            if self.fast_fail:
                                raise UnifiedVectorizationError(error_msg, operation="batch_processing", data_shape=data.shape)
                            else:
                                tprint_warning("⚠️ Fast fail disabled, skipping feature")
                                results[feature_name] = pd.Series(np.nan, index=data.index)
                                failed_features += 1
                                continue
                        
                        # Remove method and column from params to avoid conflicts
                        scaling_params = {k: v for k, v in params.items() if k not in ['method', 'column']}
                        results[feature_name] = self.scale_data(
                            data[column], method, **scaling_params
                        )
                        successful_features += 1
                        tprint_success(f"✅ Feature {feature_name} completed successfully")
                    
                    elif feature_type == 'custom':
                        func = params.get('function')
                        if not callable(func):
                            error_msg = f"Custom function for {feature_name} is not callable"
                            tprint_error(f"❌ {error_msg}")
                            if self.fast_fail:
                                raise UnifiedVectorizationError(error_msg, operation="batch_processing")
                            else:
                                tprint_warning("⚠️ Fast fail disabled, skipping feature")
                                results[feature_name] = pd.Series(np.nan, index=data.index)
                                failed_features += 1
                                continue
                        
                        results[feature_name] = func(data, **params)
                        successful_features += 1
                        tprint_success(f"✅ Feature {feature_name} completed successfully")
                    
                    else:
                        error_msg = f"Unsupported feature type: {feature_type}"
                        tprint_error(f"❌ {error_msg}")
                        if self.fast_fail:
                            raise UnifiedVectorizationError(error_msg, operation="batch_processing")
                        else:
                            tprint_warning("⚠️ Fast fail disabled, skipping feature")
                            results[feature_name] = pd.Series(np.nan, index=data.index)
                            failed_features += 1
                            continue
                    
                except Exception as e:
                    error_msg = f"Feature {feature_name} failed: {e}"
                    tprint_error(f"❌ {error_msg}")
                    self.performance_stats['errors'] += 1
                    
                    if self.fast_fail:
                        self.performance_stats['fast_failures'] += 1
                        raise UnifiedVectorizationError(error_msg, operation="batch_processing", original_error=e)
                    else:
                        tprint_warning("⚠️ Fast fail disabled, using NaN for failed feature")
                        results[feature_name] = pd.Series(np.nan, index=data.index)
                        failed_features += 1
            
            tprint_success(f"✅ Batch processing completed: {successful_features} successful, {failed_features} failed")
            return pd.DataFrame(results, index=data.index)
            
        except Exception as e:
            error_msg = f"Batch feature processing failed: {e}"
            tprint_error(f"❌ {error_msg}")
            self.performance_stats['errors'] += 1
            
            if self.fast_fail:
                self.performance_stats['fast_failures'] += 1
                raise UnifiedVectorizationError(error_msg, operation="batch_processing", data_shape=data.shape, original_error=e)
            else:
                tprint_warning("⚠️ Fast fail disabled, returning empty DataFrame")
                return pd.DataFrame()
            
        finally:
            execution_time = time.time() - start_time
            self.performance_stats['total_time'] += execution_time
            tprint_performance(f"Batch processing ({len(feature_configs)} features)", execution_time)
    
    def optimize_dataframe(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Optimize DataFrame for memory efficiency and VectorBT processing.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Optimized DataFrame
        """
        if not self.config.memory_efficient:
            return data
        
        try:
            optimized_data = data.copy()
            
            # Optimize data types
            for column in optimized_data.columns:
                if optimized_data[column].dtype == 'float64':
                    if (optimized_data[column].min() >= np.finfo(np.float32).min and 
                        optimized_data[column].max() <= np.finfo(np.float32).max):
                        optimized_data[column] = optimized_data[column].astype(np.float32)
                        self.performance_stats['memory_optimizations'] += 1
                
                elif optimized_data[column].dtype == 'int64':
                    if (optimized_data[column].min() >= np.iinfo(np.int32).min and 
                        optimized_data[column].max() <= np.iinfo(np.int32).max):
                        optimized_data[column] = optimized_data[column].astype(np.int32)
                        self.performance_stats['memory_optimizations'] += 1
            
            # Calculate memory savings
            original_memory = data.memory_usage(deep=True).sum()
            optimized_memory = optimized_data.memory_usage(deep=True).sum()
            memory_savings = (original_memory - optimized_memory) / original_memory * 100
            self.performance_stats['memory_savings'] += memory_savings
            
            return optimized_data
            
        except Exception as e:
            logger.warning(f"DataFrame optimization failed: {e}")
            return data
    
    def _pandas_fallback_rolling(self, data: Union[pd.Series, pd.DataFrame], 
                                operation: str, window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
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
        elif operation == 'quantile':
            q = kwargs.get('q', 0.5)
            return rolling_obj.quantile(q)
        elif operation == 'skew':
            return rolling_obj.skew()
        elif operation == 'kurt':
            return rolling_obj.kurt()
        elif operation == 'corr':
            other = kwargs.get('other')
            return rolling_obj.corr(other)
        elif operation == 'cov':
            other = kwargs.get('other')
            return rolling_obj.cov(other)
        elif operation == 'apply':
            func = kwargs.get('func')
            return rolling_obj.apply(func)
        else:
            raise ValueError(f"Unsupported pandas operation: {operation}")
    
    def _pandas_fallback_scaling(self, data: Union[pd.Series, pd.DataFrame], 
                                method: str, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Fallback scaling using pandas/numpy."""
        if method == 'zscore':
            return (data - data.mean()) / data.std()
        elif method == 'minmax':
            return (data - data.min()) / (data.max() - data.min())
        elif method == 'robust':
            median = data.median()
            mad = (data - median).abs().median()
            return (data - median) / mad
        else:
            raise ValueError(f"Unsupported scaling method: {method}")
    
    def _generate_cache_key(self, data: Union[pd.Series, pd.DataFrame], 
                           operation: str, window: int, **kwargs) -> str:
        """Generate cache key for operation."""
        import hashlib
        
        # Create hash of data characteristics and parameters
        data_hash = hashlib.md5(str(data.shape).encode()).hexdigest()[:8]
        params_hash = hashlib.md5(str(sorted(kwargs.items())).encode()).hexdigest()[:8]
        
        return f"{operation}_{window}_{data_hash}_{params_hash}"
    
    def _get_from_cache(self, cache_key: str) -> Optional[Union[pd.Series, pd.DataFrame]]:
        """Get result from cache."""
        if not self._cache_enabled:
            return None
        
        try:
            if cache_key in self._result_cache:
                return self._result_cache[cache_key]
        except Exception as e:
            logger.warning(f"Cache retrieval failed: {e}")
        
        return None
    
    def _put_in_cache(self, cache_key: str, result: Union[pd.Series, pd.DataFrame]):
        """Put result in cache."""
        if not self._cache_enabled:
            return
        
        try:
            # Limit cache size
            if len(self._result_cache) >= self._max_cache_size:
                # Remove oldest entries (simple FIFO)
                oldest_key = next(iter(self._result_cache))
                del self._result_cache[oldest_key]
            
            self._result_cache[cache_key] = result
            
        except Exception as e:
            logger.warning(f"Cache storage failed: {e}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        stats = self.performance_stats.copy()
        
        # Add rolling optimizer stats
        rolling_stats = self.rolling_optimizer.get_performance_stats()
        stats.update(rolling_stats)
        
        # Calculate efficiency metrics
        if stats['total_operations'] > 0:
            stats['average_operation_time'] = stats['total_time'] / stats['total_operations']
            stats['vectorbt_usage_rate'] = stats['vectorbt_operations'] / stats['total_operations']
            stats['gpu_usage_rate'] = stats['gpu_operations'] / stats['total_operations']
            stats['batch_usage_rate'] = stats['batch_operations'] / stats['total_operations']
            stats['rolling_usage_rate'] = stats['rolling_operations'] / stats['total_operations']
            stats['scaling_usage_rate'] = stats['scaling_operations'] / stats['total_operations']
            
            # Cache statistics
            total_cache_ops = stats['cache_hits'] + stats['cache_misses']
            if total_cache_ops > 0:
                stats['cache_hit_rate'] = (stats['cache_hits'] / total_cache_ops) * 100
            else:
                stats['cache_hit_rate'] = 0
        else:
            stats['average_operation_time'] = 0
            stats['vectorbt_usage_rate'] = 0
            stats['gpu_usage_rate'] = 0
            stats['batch_usage_rate'] = 0
            stats['rolling_usage_rate'] = 0
            stats['scaling_usage_rate'] = 0
            stats['cache_hit_rate'] = 0
        
        return stats
    
    def _validate_config(self, config: VectorizationConfig):
        """Validate configuration parameters with detailed error reporting."""
        tprint_debug("🔍 Validating UnifiedVectorizationManager configuration")
        
        if not isinstance(config, VectorizationConfig):
            raise VectorizationValidationError("Config must be a VectorizationConfig instance", "type_check", type(config))
        
        if not isinstance(config.enable_vectorbt, bool):
            raise VectorizationValidationError("enable_vectorbt must be a boolean", "type_check", config.enable_vectorbt)
        
        if not isinstance(config.enable_gpu, bool):
            raise VectorizationValidationError("enable_gpu must be a boolean", "type_check", config.enable_gpu)
        
        if not isinstance(config.memory_efficient, bool):
            raise VectorizationValidationError("memory_efficient must be a boolean", "type_check", config.memory_efficient)
        
        if not isinstance(config.chunk_size, int) or config.chunk_size <= 0:
            raise VectorizationValidationError("chunk_size must be a positive integer", "range_check", config.chunk_size)
        
        if not isinstance(config.batch_size, int) or config.batch_size <= 0:
            raise VectorizationValidationError("batch_size must be a positive integer", "range_check", config.batch_size)
        
        if config.max_memory_gb <= 0:
            raise VectorizationValidationError("max_memory_gb must be positive", "range_check", config.max_memory_gb)
        
        tprint_success("✅ Configuration validated successfully")
    
    def _validate_rolling_inputs(self, data: Union[pd.Series, pd.DataFrame], 
                                operation: str, window: int):
        """Validate rolling operation inputs with comprehensive checks."""
        tprint_debug(f"🔍 Validating rolling inputs for {operation}")
        
        # Check data type
        if not isinstance(data, (pd.Series, pd.DataFrame)):
            raise VectorizationValidationError("Data must be a pandas Series or DataFrame", "type_check", type(data))
        
        # Check data is not empty
        if len(data) == 0:
            raise VectorizationValidationError("Data cannot be empty", "empty_check", len(data))
        
        # Check window size
        if not isinstance(window, int) or window <= 0:
            raise VectorizationValidationError("Window must be a positive integer", "range_check", window)
        
        if window > len(data):
            raise VectorizationValidationError(f"Window size ({window}) cannot be larger than data length ({len(data)})", "range_check", window)
        
        # Check for supported operations
        supported_operations = ['mean', 'std', 'var', 'min', 'max', 'sum', 'quantile', 'skew', 'kurt', 'corr', 'cov', 'apply']
        if operation not in supported_operations:
            raise VectorizationValidationError(f"Unsupported operation: {operation}. Supported: {supported_operations}", "operation_check", operation)
        
        tprint_success(f"✅ Rolling inputs validated for {operation}")
    
    def _validate_scaling_inputs(self, data: Union[pd.Series, pd.DataFrame], method: str):
        """Validate scaling operation inputs."""
        tprint_debug(f"🔍 Validating scaling inputs for {method}")
        
        # Check data type
        if not isinstance(data, (pd.Series, pd.DataFrame)):
            raise VectorizationValidationError("Data must be a pandas Series or DataFrame", "type_check", type(data))
        
        # Check data is not empty
        if len(data) == 0:
            raise VectorizationValidationError("Data cannot be empty", "empty_check", len(data))
        
        # Check for supported methods
        supported_methods = ['zscore', 'minmax', 'robust', 'quantile', 'winsorize', 'rank', 'clip']
        if method not in supported_methods:
            raise VectorizationValidationError(f"Unsupported scaling method: {method}. Supported: {supported_methods}", "method_check", method)
        
        tprint_success(f"✅ Scaling inputs validated for {method}")

    def reset_stats(self):
        """Reset all performance statistics."""
        tprint_info("🔄 Resetting UnifiedVectorizationManager performance statistics")
        self.performance_stats = {
            'total_operations': 0,
            'vectorbt_operations': 0,
            'pandas_fallbacks': 0,
            'gpu_operations': 0,
            'batch_operations': 0,
            'rolling_operations': 0,
            'scaling_operations': 0,
            'memory_optimizations': 0,
            'total_time': 0.0,
            'memory_savings': 0.0,
            'cache_hits': 0,
            'cache_misses': 0,
            'errors': 0,
            'fast_failures': 0,
            'validation_errors': 0
        }
        
        if self.rolling_optimizer:
            self.rolling_optimizer.reset_stats()
        self._result_cache.clear()
        tprint_success("✅ Performance statistics reset")
    
    @contextmanager
    def performance_monitoring(self, operation_name: str):
        """Context manager for performance monitoring."""
        if not self.config.enable_monitoring:
            yield
            return
        
        start_time = time.time()
        start_memory = 0  # Could add memory monitoring here
        
        try:
            yield
        finally:
            end_time = time.time()
            execution_time = end_time - start_time
            
            logger.info(f"Operation {operation_name}: {execution_time:.3f}s")


# Global instance
_global_vectorization_manager = None


def get_unified_vectorization_manager(config: Optional[VectorizationConfig] = None) -> UnifiedVectorizationManager:
    """Get global unified vectorization manager instance."""
    global _global_vectorization_manager
    if _global_vectorization_manager is None:
        _global_vectorization_manager = UnifiedVectorizationManager(config)
    return _global_vectorization_manager


def create_optimized_vectorization_pipeline(enable_gpu: bool = False, 
                                          memory_efficient: bool = True) -> UnifiedVectorizationManager:
    """
    Create an optimized vectorization pipeline.
    
    Args:
        enable_gpu: Enable GPU acceleration
        memory_efficient: Enable memory optimization
        
    Returns:
        Unified vectorization manager
    """
    config = VectorizationConfig(
        enable_vectorbt=True,
        enable_gpu=enable_gpu,
        enable_parallel=True,
        memory_efficient=memory_efficient,
        max_memory_gb=8.0,
        chunk_size=1000,
        enable_monitoring=True,
        enable_profiling=False,
        batch_size=10000,
        enable_batch_processing=True,
        rolling_optimization_threshold=1000,
        enable_rolling_optimization=True
    )
    
    return UnifiedVectorizationManager(config)


# Example usage and testing
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    data = pd.DataFrame({
        'close': 100 + np.cumsum(np.random.randn(10000) * 0.01),
        'volume': np.random.randint(1000, 10000, 10000),
        'high': 100 + np.cumsum(np.random.randn(10000) * 0.01) + np.abs(np.random.randn(10000) * 0.5),
        'low': 100 + np.cumsum(np.random.randn(10000) * 0.01) - np.abs(np.random.randn(10000) * 0.5)
    })
    
    print("Original data shape:", data.shape)
    print("Original memory usage:", data.memory_usage(deep=True).sum() / (1024**3), "GB")
    
    # Create unified vectorization manager
    manager = get_unified_vectorization_manager(
        VectorizationConfig(
            enable_vectorbt=True,
            enable_gpu=False,
            memory_efficient=True,
            enable_monitoring=True
        )
    )
    
    # Test rolling operations
    print("\nTesting rolling operations...")
    rolling_mean = manager.rolling_operation(data['close'], 'mean', window=20)
    rolling_std = manager.rolling_operation(data['close'], 'std', window=20)
    print(f"Rolling mean shape: {rolling_mean.shape}")
    print(f"Rolling std shape: {rolling_std.shape}")
    
    # Test scaling
    print("\nTesting scaling...")
    scaled_close = manager.scale_data(data['close'], method='zscore')
    print(f"Scaled close shape: {scaled_close.shape}")
    
    # Test batch processing
    print("\nTesting batch processing...")
    feature_configs = [
        {'name': 'sma_20', 'type': 'rolling', 'params': {'operation': 'mean', 'window': 20, 'column': 'close'}},
        {'name': 'sma_50', 'type': 'rolling', 'params': {'operation': 'mean', 'window': 50, 'column': 'close'}},
        {'name': 'std_20', 'type': 'rolling', 'params': {'operation': 'std', 'window': 20, 'column': 'close'}},
        {'name': 'close_scaled', 'type': 'scaling', 'params': {'method': 'zscore', 'column': 'close'}},
        {'name': 'volume_scaled', 'type': 'scaling', 'params': {'method': 'minmax', 'column': 'volume'}}
    ]
    
    features = manager.batch_process_features(data, feature_configs)
    print(f"Generated features shape: {features.shape}")
    print(f"Feature columns: {list(features.columns)}")
    
    # Test memory optimization
    print("\nTesting memory optimization...")
    optimized_data = manager.optimize_dataframe(data)
    print(f"Optimized memory usage: {optimized_data.memory_usage(deep=True).sum() / (1024**3):.3f}GB")
    
    # Get performance stats
    stats = manager.get_performance_stats()
    print(f"\nPerformance stats: {stats}")
    
    print("\nUnified vectorization pipeline test completed successfully!")