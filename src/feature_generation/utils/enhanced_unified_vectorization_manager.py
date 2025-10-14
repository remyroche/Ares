"""
Enhanced Unified Vectorization Manager

This module provides an enhanced centralized vectorization management system that unifies
VectorBT optimizations, rolling operations, and batch processing with advanced features
including adaptive chunking, multi-level caching, and M1 GPU optimization.

Key Features:
- Enhanced VectorBTRollingOptimizer integration
- Advanced memory management and adaptive chunking
- Multi-level caching strategies (L1 memory, L2 disk)
- Mac M1 GPU optimization with Metal Performance Shaders
- Backward compatibility with existing UnifiedVectorizationManager
- Intelligent resource management and performance monitoring
"""

import numpy as np
import pandas as pd
import logging
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
import time
from contextlib import contextmanager
from dataclasses import dataclass
import warnings
import threading
from collections import deque
import hashlib
import pickle
import os
from pathlib import Path

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

# M1 GPU optimization imports
try:
    import torch
    TORCH_AVAILABLE = True
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        M1_GPU_AVAILABLE = True
        tprint_info("🍎 Mac M1 GPU (Metal Performance Shaders) detected")
    else:
        M1_GPU_AVAILABLE = False
        tprint_info("🍎 Mac M1 detected but MPS not available")
except ImportError:
    TORCH_AVAILABLE = False
    M1_GPU_AVAILABLE = False
    tprint_warning("⚠️ PyTorch not available for M1 GPU optimization")

# Optional GPU acceleration (CUDA)
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

# Import enhanced rolling optimizer
try:
    from .enhanced_vectorbt_rolling_optimizer import (
        EnhancedVectorBTRollingOptimizer, 
        MemoryConfig, 
        CacheConfig,
        get_vectorbt_rolling_optimizer
    )
except ImportError:
    # Fallback for direct import
    from enhanced_vectorbt_rolling_optimizer import (
        EnhancedVectorBTRollingOptimizer, 
        MemoryConfig, 
        CacheConfig,
        get_vectorbt_rolling_optimizer
    )

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
class EnhancedVectorizationConfig:
    """Enhanced configuration for unified vectorization with advanced features."""
    # VectorBT settings
    enable_vectorbt: bool = True
    enable_gpu: bool = False
    enable_parallel: bool = True
    
    # Memory management
    memory_efficient: bool = True
    max_memory_gb: float = 8.0
    chunk_size: int = 1000
    adaptive_chunking: bool = True
    memory_pooling: bool = True
    memory_pressure_threshold: float = 0.8
    
    # Caching settings
    enable_caching: bool = True
    l1_cache_size: int = 1000
    l2_cache_size: int = 10000
    l2_cache_dir: str = "./cache"
    cache_ttl: float = 3600.0
    cache_compression: bool = True
    
    # M1 GPU settings
    enable_m1_gpu: bool = True
    
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
        
        if self.enable_m1_gpu and not M1_GPU_AVAILABLE:
            self.enable_m1_gpu = False
            logger.warning("M1 GPU acceleration requested but not available")

class PerformanceMonitor:
    """Enhanced performance monitoring with real-time metrics."""
    
    def __init__(self, enable_monitoring: bool = True):
        self.enable_monitoring = enable_monitoring
        self.metrics = {
            'operation_times': deque(maxlen=1000),
            'memory_usage': deque(maxlen=1000),
            'cache_performance': deque(maxlen=1000),
            'error_rates': deque(maxlen=1000)
        }
        self._lock = threading.Lock()
        self._monitoring_thread = None
        self._monitoring = False
        
        if self.enable_monitoring:
            self._start_monitoring()
    
    def record_operation(self, operation_name: str, duration: float, memory_used: float = 0.0):
        """Record operation metrics."""
        if not self.enable_monitoring:
            return
        
        with self._lock:
            self.metrics['operation_times'].append({
                'operation': operation_name,
                'duration': duration,
                'timestamp': time.time(),
                'memory_used': memory_used
            })
    
    def record_cache_performance(self, hit: bool, operation: str):
        """Record cache performance metrics."""
        if not self.enable_monitoring:
            return
        
        with self._lock:
            self.metrics['cache_performance'].append({
                'hit': hit,
                'operation': operation,
                'timestamp': time.time()
            })
    
    def record_error(self, error_type: str, operation: str):
        """Record error metrics."""
        if not self.enable_monitoring:
            return
        
        with self._lock:
            self.metrics['error_rates'].append({
                'error_type': error_type,
                'operation': operation,
                'timestamp': time.time()
            })
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics."""
        with self._lock:
            if not self.metrics['operation_times']:
                return {'status': 'no_data'}
            
            durations = [op['duration'] for op in self.metrics['operation_times']]
            memory_usage = [op['memory_used'] for op in self.metrics['operation_times'] if op['memory_used'] > 0]
            
            cache_hits = sum(1 for perf in self.metrics['cache_performance'] if perf['hit'])
            cache_total = len(self.metrics['cache_performance'])
            
            return {
                'total_operations': len(self.metrics['operation_times']),
                'avg_duration': np.mean(durations) if durations else 0,
                'max_duration': np.max(durations) if durations else 0,
                'min_duration': np.min(durations) if durations else 0,
                'avg_memory_usage': np.mean(memory_usage) if memory_usage else 0,
                'max_memory_usage': np.max(memory_usage) if memory_usage else 0,
                'cache_hit_rate': (cache_hits / cache_total * 100) if cache_total > 0 else 0,
                'total_errors': len(self.metrics['error_rates']),
                'error_rate': (len(self.metrics['error_rates']) / len(self.metrics['operation_times']) * 100) if self.metrics['operation_times'] else 0
            }
    
    def _start_monitoring(self):
        """Start background monitoring."""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_performance, daemon=True)
        self._monitor_thread.start()
        tprint_info("📊 Started performance monitoring")
    
    def _monitor_performance(self):
        """Background performance monitoring thread."""
        while self._monitoring:
            try:
                # Monitor system resources
                import psutil
                memory_usage = psutil.virtual_memory().percent
                
                with self._lock:
                    self.metrics['memory_usage'].append({
                        'usage_percent': memory_usage,
                        'timestamp': time.time()
                    })
                
                time.sleep(10)  # Check every 10 seconds
            except Exception as e:
                tprint_error(f"❌ Performance monitoring error: {e}")
                time.sleep(30)
    
    def cleanup(self):
        """Cleanup performance monitor."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1)
        tprint_info("🧹 Performance monitor cleaned up")

class EnhancedUnifiedVectorizationManager:
    """
    Enhanced unified manager for all vectorization operations using VectorBT optimizations
    with advanced features including adaptive chunking, multi-level caching, and M1 GPU optimization.
    
    This class provides a single interface for:
    - Enhanced VectorBT rolling operations
    - Advanced memory management and adaptive chunking
    - Multi-level caching strategies
    - M1 GPU optimization
    - Batch processing
    - Performance monitoring
    - Parallel processing
    """
    
    def __init__(self, config: Optional[EnhancedVectorizationConfig] = None, 
                 fast_fail: bool = True, enable_logging: bool = True):
        """
        Initialize enhanced unified vectorization manager with advanced features.
        
        Args:
            config: Enhanced vectorization configuration
            fast_fail: Enable fast failing instead of silent fallbacks
            enable_logging: Enable comprehensive logging with tprint
        """
        tprint_info("🚀 Initializing Enhanced UnifiedVectorizationManager with advanced features")
        
        self.config = config or EnhancedVectorizationConfig()
        self.fast_fail = fast_fail
        self.enable_logging = enable_logging
        
        # Validate configuration
        self._validate_config(self.config)
        
        # Initialize enhanced components with error handling
        tprint_info("🔧 Initializing enhanced vectorization components")
        try:
            # Initialize memory and cache configurations
            memory_config = MemoryConfig(
                max_memory_gb=self.config.max_memory_gb,
                memory_pressure_threshold=self.config.memory_pressure_threshold,
                adaptive_chunking=self.config.adaptive_chunking,
                memory_pooling=self.config.memory_pooling,
                memory_monitoring=self.config.enable_monitoring
            )
            
            cache_config = CacheConfig(
                l1_cache_size=self.config.l1_cache_size,
                l2_cache_size=self.config.l2_cache_size,
                l2_cache_dir=self.config.l2_cache_dir,
                cache_ttl=self.config.cache_ttl,
                cache_compression=self.config.cache_compression
            )
            
            # Initialize enhanced rolling optimizer
            self.rolling_optimizer = get_vectorbt_rolling_optimizer(
                enable_gpu=self.config.enable_gpu,
                enable_parallel=self.config.enable_parallel,
                memory_efficient=self.config.memory_efficient,
                chunk_size=self.config.chunk_size,
                fast_fail=self.fast_fail,
                enable_logging=self.enable_logging,
                memory_config=memory_config,
                cache_config=cache_config,
                enable_m1_gpu=self.config.enable_m1_gpu,
                enable_adaptive_chunking=self.config.adaptive_chunking,
                enable_advanced_caching=self.config.enable_caching
            )
            tprint_success("✅ Enhanced rolling optimizer initialized")
        except Exception as e:
            error_msg = f"Failed to initialize enhanced rolling optimizer: {e}"
            tprint_error(f"❌ {error_msg}")
            if self.fast_fail:
                raise UnifiedVectorizationError(error_msg, operation="initialization", original_error=e)
            else:
                tprint_warning("⚠️ Fast fail disabled, continuing without enhanced rolling optimizer")
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
        
        # Initialize performance monitor
        self.performance_monitor = PerformanceMonitor(self.config.enable_monitoring)
        
        # Enhanced performance tracking with error tracking
        self.performance_stats = {
            'total_operations': 0,
            'vectorbt_operations': 0,
            'pandas_fallbacks': 0,
            'gpu_operations': 0,
            'm1_gpu_operations': 0,
            'batch_operations': 0,
            'rolling_operations': 0,
            'scaling_operations': 0,
            'memory_optimizations': 0,
            'adaptive_chunk_operations': 0,
            'cache_operations': 0,
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
        self._cache_enabled = self.config.enable_caching
        self._max_cache_size = self.config.l1_cache_size
        
        tprint_success(f"✅ Enhanced UnifiedVectorizationManager initialized: VectorBT={self.config.enable_vectorbt}, GPU={self.config.enable_gpu}, M1GPU={self.config.enable_m1_gpu}, AdaptiveChunking={self.config.adaptive_chunking}, AdvancedCaching={self.config.enable_caching}")
        logger.info(f"Enhanced UnifiedVectorizationManager initialized: VectorBT={self.config.enable_vectorbt}, GPU={self.config.enable_gpu}, M1GPU={self.config.enable_m1_gpu}")
    
    def rolling_operation(self, data: Union[pd.Series, pd.DataFrame], 
                         operation: str, window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """
        Perform enhanced rolling operation with adaptive chunking, caching, and M1 GPU optimization.
        
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
        
        tprint_debug(f"🔄 Starting enhanced rolling operation: {operation}, window={window}, data_shape={data.shape if hasattr(data, 'shape') else 'unknown'}")
        
        # Validate inputs
        self._validate_rolling_inputs(data, operation, window)
        
        # Check cache first
        if self._cache_enabled:
            cache_key = self._generate_cache_key(data, operation, window, **kwargs)
            cached_result = self._get_from_cache(cache_key)
            if cached_result is not None:
                self.performance_stats['cache_hits'] += 1
                self.performance_monitor.record_cache_performance(True, operation)
                tprint_debug("💾 Cache hit for rolling operation")
                return cached_result
            self.performance_stats['cache_misses'] += 1
            self.performance_monitor.record_cache_performance(False, operation)
            tprint_debug("💾 Cache miss for rolling operation")
        
        # Check if rolling optimizer is available
        if self.rolling_optimizer is None:
            error_msg = "Enhanced rolling optimizer not available"
            tprint_error(f"❌ {error_msg}")
            if self.fast_fail:
                raise UnifiedVectorizationError(error_msg, operation=operation, data_shape=data.shape if hasattr(data, 'shape') else None)
            else:
                tprint_warning("⚠️ Fast fail disabled, using pandas fallback")
                return self._pandas_fallback_rolling(data, operation, window, **kwargs)
        
        try:
            # Use enhanced VectorBT rolling optimizer with detailed logging
            tprint_debug(f"🎯 Executing enhanced rolling {operation} with VectorBT optimizer")
            
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
            if rolling_stats.get('m1_gpu_operations', 0) > 0:
                self.performance_stats['m1_gpu_operations'] += 1
            if rolling_stats.get('memory_optimizations', 0) > 0:
                self.performance_stats['memory_optimizations'] += 1
            if rolling_stats.get('adaptive_chunk_operations', 0) > 0:
                self.performance_stats['adaptive_chunk_operations'] += 1
            
            # Cache result
            if self._cache_enabled:
                self._put_in_cache(cache_key, result)
                tprint_debug("💾 Result cached successfully")
            
            tprint_success(f"✅ Enhanced rolling {operation} completed successfully")
            return result
            
        except Exception as e:
            error_msg = f"Enhanced rolling operation {operation} failed"
            tprint_error(f"❌ {error_msg}: {e}")
            self.performance_stats['errors'] += 1
            self.performance_monitor.record_error("rolling_operation", operation)
            
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
            self.performance_monitor.record_operation(f"rolling_{operation}", execution_time)
            tprint_performance(f"Enhanced rolling {operation}", execution_time)
    
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
        
        tprint_debug(f"🔄 Starting enhanced data scaling: method={method}, data_shape={data.shape if hasattr(data, 'shape') else 'unknown'}")
        
        # Validate inputs
        self._validate_scaling_inputs(data, method)
        
        if not VECTORBT_AVAILABLE or not self.config.enable_vectorbt:
            tprint_warning("⚠️ VectorBT not available, using pandas fallback for scaling")
            return self._pandas_fallback_scaling(data, method, **kwargs)
        
        try:
            tprint_debug(f"🎯 Executing enhanced {method} scaling with VectorBT")
            
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
            tprint_success(f"✅ Enhanced {method} scaling completed successfully")
            return result
            
        except Exception as e:
            error_msg = f"Enhanced VectorBT scaling failed for {method}"
            tprint_error(f"❌ {error_msg}: {e}")
            self.performance_stats['errors'] += 1
            self.performance_monitor.record_error("scaling", method)
            
            if self.fast_fail:
                self.performance_stats['fast_failures'] += 1
                raise UnifiedVectorizationError(error_msg, operation="scaling", data_shape=data.shape if hasattr(data, 'shape') else None, original_error=e)
            else:
                tprint_warning("⚠️ Fast fail disabled, using pandas fallback")
                return self._pandas_fallback_scaling(data, method, **kwargs)
        
        finally:
            execution_time = time.time() - start_time
            self.performance_stats['total_time'] += execution_time
            self.performance_monitor.record_operation(f"scaling_{method}", execution_time)
            tprint_performance(f"Enhanced scaling {method}", execution_time)
    
    def batch_process_features(self, data: pd.DataFrame, 
                             feature_configs: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Process multiple features in batch with enhanced optimization and logging.
        
        Args:
            data: Input OHLCV data
            feature_configs: List of feature configuration dictionaries
            
        Returns:
            DataFrame with generated features
        """
        start_time = time.time()
        self.performance_stats['total_operations'] += 1
        self.performance_stats['batch_operations'] += 1
        
        tprint_info(f"🔄 Starting enhanced batch feature processing: {len(feature_configs)} features, data_shape={data.shape}")
        
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
            # Use enhanced VectorBT batch processor
            results = {}
            successful_features = 0
            failed_features = 0
            
            tprint_debug(f"🎯 Processing {len(feature_configs)} features with enhanced optimization")
            
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
                    self.performance_monitor.record_error("batch_feature", feature_name)
                    
                    if self.fast_fail:
                        self.performance_stats['fast_failures'] += 1
                        raise UnifiedVectorizationError(error_msg, operation="batch_processing", original_error=e)
                    else:
                        tprint_warning("⚠️ Fast fail disabled, using NaN for failed feature")
                        results[feature_name] = pd.Series(np.nan, index=data.index)
                        failed_features += 1
            
            tprint_success(f"✅ Enhanced batch processing completed: {successful_features} successful, {failed_features} failed")
            return pd.DataFrame(results, index=data.index)
            
        except Exception as e:
            error_msg = f"Enhanced batch feature processing failed: {e}"
            tprint_error(f"❌ {error_msg}")
            self.performance_stats['errors'] += 1
            self.performance_monitor.record_error("batch_processing", "batch_processing")
            
            if self.fast_fail:
                self.performance_stats['fast_failures'] += 1
                raise UnifiedVectorizationError(error_msg, operation="batch_processing", data_shape=data.shape, original_error=e)
            else:
                tprint_warning("⚠️ Fast fail disabled, returning empty DataFrame")
                return pd.DataFrame()
            
        finally:
            execution_time = time.time() - start_time
            self.performance_stats['total_time'] += execution_time
            self.performance_monitor.record_operation("batch_processing", execution_time)
            tprint_performance(f"Enhanced batch processing ({len(feature_configs)} features)", execution_time)
    
    def optimize_dataframe(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Optimize DataFrame for memory efficiency and VectorBT processing with enhanced features.
        
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
            
            tprint_debug(f"🧠 Memory optimization: {memory_savings:.2f}% savings")
            return optimized_data
            
        except Exception as e:
            logger.warning(f"DataFrame optimization failed: {e}")
            return data
    
    def _pandas_fallback_rolling(self, data: Union[pd.Series, pd.DataFrame], 
                                operation: str, window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Fallback rolling operation using pandas (original implementation)."""
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
        """Fallback scaling using pandas/numpy (original implementation)."""
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
        """Generate cache key for operation (original implementation)."""
        import hashlib
        
        # Create hash of data characteristics and parameters
        data_hash = hashlib.md5(str(data.shape).encode()).hexdigest()[:8]
        params_hash = hashlib.md5(str(sorted(kwargs.items())).encode()).hexdigest()[:8]
        
        return f"{operation}_{window}_{data_hash}_{params_hash}"
    
    def _get_from_cache(self, cache_key: str) -> Optional[Union[pd.Series, pd.DataFrame]]:
        """Get result from cache (original implementation)."""
        if not self._cache_enabled:
            return None
        
        try:
            if cache_key in self._result_cache:
                return self._result_cache[cache_key]
        except Exception as e:
            logger.warning(f"Cache retrieval failed: {e}")
        
        return None
    
    def _put_in_cache(self, cache_key: str, result: Union[pd.Series, pd.DataFrame]):
        """Put result in cache (original implementation)."""
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
        """Get comprehensive enhanced performance statistics."""
        stats = self.performance_stats.copy()
        
        # Add rolling optimizer stats
        if self.rolling_optimizer:
            rolling_stats = self.rolling_optimizer.get_performance_stats()
            stats.update(rolling_stats)
        
        # Add performance monitor stats
        if self.performance_monitor:
            monitor_stats = self.performance_monitor.get_performance_summary()
            stats['performance_monitor'] = monitor_stats
        
        # Calculate efficiency metrics
        if stats['total_operations'] > 0:
            stats['average_operation_time'] = stats['total_time'] / stats['total_operations']
            stats['vectorbt_usage_rate'] = stats['vectorbt_operations'] / stats['total_operations']
            stats['gpu_usage_rate'] = stats['gpu_operations'] / stats['total_operations']
            stats['m1_gpu_usage_rate'] = stats['m1_gpu_operations'] / stats['total_operations']
            stats['batch_usage_rate'] = stats['batch_operations'] / stats['total_operations']
            stats['rolling_usage_rate'] = stats['rolling_operations'] / stats['total_operations']
            stats['scaling_usage_rate'] = stats['scaling_operations'] / stats['total_operations']
            stats['adaptive_chunk_usage_rate'] = stats['adaptive_chunk_operations'] / stats['total_operations']
            
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
            stats['m1_gpu_usage_rate'] = 0
            stats['batch_usage_rate'] = 0
            stats['rolling_usage_rate'] = 0
            stats['scaling_usage_rate'] = 0
            stats['adaptive_chunk_usage_rate'] = 0
            stats['cache_hit_rate'] = 0
        
        return stats
    
    def _validate_config(self, config: EnhancedVectorizationConfig):
        """Validate enhanced configuration parameters with detailed error reporting."""
        tprint_debug("🔍 Validating Enhanced UnifiedVectorizationManager configuration")
        
        if not isinstance(config, EnhancedVectorizationConfig):
            raise VectorizationValidationError("Config must be an EnhancedVectorizationConfig instance", "type_check", type(config))
        
        if not isinstance(config.enable_vectorbt, bool):
            raise VectorizationValidationError("enable_vectorbt must be a boolean", "type_check", config.enable_vectorbt)
        
        if not isinstance(config.enable_gpu, bool):
            raise VectorizationValidationError("enable_gpu must be a boolean", "type_check", config.enable_gpu)
        
        if not isinstance(config.enable_m1_gpu, bool):
            raise VectorizationValidationError("enable_m1_gpu must be a boolean", "type_check", config.enable_m1_gpu)
        
        if not isinstance(config.memory_efficient, bool):
            raise VectorizationValidationError("memory_efficient must be a boolean", "type_check", config.memory_efficient)
        
        if not isinstance(config.adaptive_chunking, bool):
            raise VectorizationValidationError("adaptive_chunking must be a boolean", "type_check", config.adaptive_chunking)
        
        if not isinstance(config.enable_caching, bool):
            raise VectorizationValidationError("enable_caching must be a boolean", "type_check", config.enable_caching)
        
        if not isinstance(config.chunk_size, int) or config.chunk_size <= 0:
            raise VectorizationValidationError("chunk_size must be a positive integer", "range_check", config.chunk_size)
        
        if not isinstance(config.batch_size, int) or config.batch_size <= 0:
            raise VectorizationValidationError("batch_size must be a positive integer", "range_check", config.batch_size)
        
        if config.max_memory_gb <= 0:
            raise VectorizationValidationError("max_memory_gb must be positive", "range_check", config.max_memory_gb)
        
        if config.memory_pressure_threshold <= 0 or config.memory_pressure_threshold > 1:
            raise VectorizationValidationError("memory_pressure_threshold must be between 0 and 1", "range_check", config.memory_pressure_threshold)
        
        tprint_success("✅ Enhanced configuration validated successfully")
    
    def _validate_rolling_inputs(self, data: Union[pd.Series, pd.DataFrame], 
                                operation: str, window: int):
        """Validate rolling operation inputs with comprehensive checks (original implementation)."""
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
        """Validate scaling operation inputs (original implementation)."""
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
        """Reset all enhanced performance statistics."""
        tprint_info("🔄 Resetting Enhanced UnifiedVectorizationManager performance statistics")
        self.performance_stats = {
            'total_operations': 0,
            'vectorbt_operations': 0,
            'pandas_fallbacks': 0,
            'gpu_operations': 0,
            'm1_gpu_operations': 0,
            'batch_operations': 0,
            'rolling_operations': 0,
            'scaling_operations': 0,
            'memory_optimizations': 0,
            'adaptive_chunk_operations': 0,
            'cache_operations': 0,
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
        tprint_success("✅ Enhanced performance statistics reset")
    
    @contextmanager
    def performance_monitoring(self, operation_name: str):
        """Context manager for performance monitoring (original implementation)."""
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
    
    def cleanup(self) -> None:
        """Enhanced cleanup with resource management."""
        tprint("🧹 Cleaning up Enhanced UnifiedVectorizationManager resources")
        
        try:
            # Cleanup rolling optimizer
            if self.rolling_optimizer:
                self.rolling_optimizer.cleanup()
                tprint("✅ Rolling optimizer cleaned up")
            
            # Cleanup performance monitor
            if self.performance_monitor:
                self.performance_monitor.cleanup()
                tprint("✅ Performance monitor cleaned up")
            
            # Clear caches
            self._result_cache.clear()
            tprint("✅ Result cache cleared")
            
            # Reset stats
            self.reset_stats()
            
            # Force garbage collection
            import gc
            gc.collect()
            tprint("✅ Garbage collection completed")
            
        except Exception as e:
            tprint_error(f"❌ ERROR: Enhanced UnifiedVectorizationManager cleanup failed: {e}")
            raise RuntimeError(f"Enhanced UnifiedVectorizationManager cleanup failed: {e}")
        
        tprint("✅ Enhanced UnifiedVectorizationManager cleanup completed")
    
    def __enter__(self) -> 'EnhancedUnifiedVectorizationManager':
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit with cleanup."""
        self.cleanup()


# Backward compatibility - create alias for original class name
UnifiedVectorizationManager = EnhancedUnifiedVectorizationManager

# Global instance
_global_vectorization_manager = None

def get_unified_vectorization_manager(config: Optional[EnhancedVectorizationConfig] = None) -> EnhancedUnifiedVectorizationManager:
    """Get global enhanced unified vectorization manager instance with backward compatibility."""
    global _global_vectorization_manager
    if _global_vectorization_manager is None:
        _global_vectorization_manager = EnhancedUnifiedVectorizationManager(config)
    return _global_vectorization_manager

def create_optimized_vectorization_pipeline(enable_gpu: bool = False, 
                                          memory_efficient: bool = True,
                                          enable_m1_gpu: bool = True,
                                          adaptive_chunking: bool = True,
                                          enable_caching: bool = True) -> EnhancedUnifiedVectorizationManager:
    """
    Create an enhanced optimized vectorization pipeline with advanced features.
    
    Args:
        enable_gpu: Enable GPU acceleration
        memory_efficient: Enable memory optimization
        enable_m1_gpu: Enable Mac M1 GPU optimization
        adaptive_chunking: Enable adaptive chunking
        enable_caching: Enable advanced caching
        
    Returns:
        Enhanced unified vectorization manager
    """
    config = EnhancedVectorizationConfig(
        enable_vectorbt=True,
        enable_gpu=enable_gpu,
        enable_parallel=True,
        memory_efficient=memory_efficient,
        max_memory_gb=8.0,
        chunk_size=1000,
        adaptive_chunking=adaptive_chunking,
        memory_pooling=True,
        memory_pressure_threshold=0.8,
        enable_caching=enable_caching,
        l1_cache_size=1000,
        l2_cache_size=10000,
        l2_cache_dir="./cache",
        cache_ttl=3600.0,
        cache_compression=True,
        enable_m1_gpu=enable_m1_gpu,
        enable_monitoring=True,
        enable_profiling=False,
        batch_size=10000,
        enable_batch_processing=True,
        rolling_optimization_threshold=1000,
        enable_rolling_optimization=True
    )
    
    return EnhancedUnifiedVectorizationManager(config)

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
    
    # Test enhanced unified vectorization manager with backward compatibility
    print("\nTesting Enhanced Unified Vectorization Manager with backward compatibility...")
    
    # Test with original parameters (backward compatibility)
    manager = EnhancedUnifiedVectorizationManager(
        EnhancedVectorizationConfig(
            enable_vectorbt=True,
            enable_gpu=False,
            memory_efficient=True,
            enable_monitoring=True
        )
    )
    
    # Test rolling operations
    print("\nTesting enhanced rolling operations...")
    rolling_mean = manager.rolling_operation(data['close'], 'mean', window=20)
    rolling_std = manager.rolling_operation(data['close'], 'std', window=20)
    print(f"Rolling mean shape: {rolling_mean.shape}")
    print(f"Rolling std shape: {rolling_std.shape}")
    
    # Test scaling
    print("\nTesting enhanced scaling...")
    scaled_close = manager.scale_data(data['close'], method='zscore')
    print(f"Scaled close shape: {scaled_close.shape}")
    
    # Test batch processing
    print("\nTesting enhanced batch processing...")
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
    print("\nTesting enhanced memory optimization...")
    optimized_data = manager.optimize_dataframe(data)
    print(f"Optimized memory usage: {optimized_data.memory_usage(deep=True).sum() / (1024**3):.3f}GB")
    
    # Get enhanced performance stats
    stats = manager.get_performance_stats()
    print(f"\nEnhanced performance stats: {stats}")
    
    # Test with enhanced features
    print("\nTesting enhanced features...")
    
    # Create enhanced configuration
    enhanced_config = EnhancedVectorizationConfig(
        enable_vectorbt=True,
        enable_gpu=False,
        enable_m1_gpu=True,
        memory_efficient=True,
        adaptive_chunking=True,
        memory_pooling=True,
        memory_pressure_threshold=0.7,
        enable_caching=True,
        l1_cache_size=500,
        l2_cache_size=2000,
        cache_ttl=1800.0,  # 30 minutes
        enable_monitoring=True
    )
    
    # Test with enhanced configuration
    enhanced_manager = EnhancedUnifiedVectorizationManager(enhanced_config)
    
    # Test enhanced operations
    enhanced_mean = enhanced_manager.rolling_operation(data['close'], 'mean', window=20)
    print(f"Enhanced rolling mean shape: {enhanced_mean.shape}")
    
    # Test caching (second call should hit cache)
    cached_mean = enhanced_manager.rolling_operation(data['close'], 'mean', window=20)
    print(f"Cached rolling mean shape: {cached_mean.shape}")
    
    # Enhanced performance stats
    enhanced_stats = enhanced_manager.get_performance_stats()
    print(f"Enhanced performance stats: {enhanced_stats}")
    
    print("\nEnhanced Unified Vectorization Manager test completed successfully!")