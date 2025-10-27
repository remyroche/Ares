"""
Unified Vectorization Manager for efficient vectorized computations.

This module provides a unified interface for vectorized operations across
different backends (VectorBT, NumPy, Pandas) with automatic optimization.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from dataclasses import dataclass
import time
import warnings

from src.utils.tprint import tprint, tprint_data_preview, tprint_data_format
from src.utils.logger import system_logger
from src.utils.hardware import get_optimal_workers, get_memory_info

# VectorBT imports
try:
    from src.vectorbt import vbt, rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max, rolling_sum, rolling_apply, VECTORBT_AVAILABLE
    from src.vectorbt import rolling_corr, rolling_cov, scale, rank, zscore, winsorize, clip, quantile
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

@dataclass
class VectorizationConfig:
    """Configuration for vectorization operations."""
    enable_vectorbt: bool = True
    enable_parallel: bool = True
    max_workers: Optional[int] = None
    memory_threshold_mb: int = 1000
    performance_threshold: int = 1000
    chunk_size: Optional[int] = None
    use_numba: bool = True

@dataclass
class VectorizationResult:
    """Result of a vectorization operation."""
    data: Union[pd.Series, pd.DataFrame, np.ndarray]
    execution_time: float
    method_used: str
    memory_used_mb: float
    performance_gain: Optional[float] = None
    metadata: Dict[str, Any] = None

class UnifiedVectorizationManager:
    """Unified manager for vectorized operations with automatic optimization."""
    
    def __init__(self, config: Optional[VectorizationConfig] = None):
        self.config = config or VectorizationConfig()
        self.logger = system_logger.getChild('UnifiedVectorizationManager')
        
        # Initialize hardware-optimized settings
        self._initialize_hardware_settings()
        
        # Performance tracking
        self.operation_stats = {
            'total_operations': 0,
            'vectorbt_operations': 0,
            'numpy_operations': 0,
            'pandas_operations': 0,
            'parallel_operations': 0,
            'total_time_saved': 0.0,
            'memory_optimizations': 0
        }
        
        tprint("🚀 UnifiedVectorizationManager initialized", "INFO")
        tprint(f"⚙️ Config: VectorBT={self.config.enable_vectorbt}, Parallel={self.config.enable_parallel}, Workers={self.config.max_workers}", "INFO")
    
    def _initialize_hardware_settings(self):
        """Initialize hardware-optimized settings."""
        try:
            # Get optimal worker count
            if self.config.max_workers is None:
                self.config.max_workers = get_optimal_workers()
            
            # Get memory info and adjust thresholds
            memory_info = get_memory_info()
            available_memory_gb = memory_info.get('available_gb', 8)
            
            # Adjust memory threshold based on available memory
            if available_memory_gb < 4:
                self.config.memory_threshold_mb = 500
                self.config.performance_threshold = 500
            elif available_memory_gb > 16:
                self.config.memory_threshold_mb = 2000
                self.config.performance_threshold = 2000
            
            # Adjust chunk size based on memory
            if self.config.chunk_size is None:
                self.config.chunk_size = min(10000, max(1000, int(available_memory_gb * 1000)))
            
            tprint(f"💻 Hardware settings: {self.config.max_workers} workers, {self.config.memory_threshold_mb}MB threshold, {self.config.chunk_size} chunk size", "INFO")
            
        except Exception as e:
            self.logger.warning(f"Failed to initialize hardware settings: {e}")
            # Use conservative defaults
            self.config.max_workers = 4
            self.config.memory_threshold_mb = 1000
            self.config.chunk_size = 5000
    
    def _should_use_vectorbt(self, data_size: int, memory_usage_mb: float) -> bool:
        """Determine if VectorBT should be used."""
        return (self.config.enable_vectorbt and 
                VECTORBT_AVAILABLE and
                data_size >= self.config.performance_threshold and
                memory_usage_mb < self.config.memory_threshold_mb)
    
    def _should_use_parallel(self, data_size: int, operation_complexity: str = "medium") -> bool:
        """Determine if parallel processing should be used."""
        if not self.config.enable_parallel:
            return False
        
        # Simple heuristic based on data size and operation complexity
        complexity_multipliers = {
            "simple": 1,
            "medium": 2,
            "complex": 4
        }
        
        threshold = self.config.performance_threshold * complexity_multipliers.get(operation_complexity, 2)
        return data_size >= threshold
    
    def _get_memory_usage_mb(self, data: Union[pd.Series, pd.DataFrame, np.ndarray]) -> float:
        """Calculate memory usage of data in MB."""
        try:
            if isinstance(data, pd.Series):
                return data.memory_usage(deep=True) / 1024 / 1024
            elif isinstance(data, pd.DataFrame):
                return data.memory_usage(deep=True).sum() / 1024 / 1024
            elif isinstance(data, np.ndarray):
                return data.nbytes / 1024 / 1024
            else:
                return 0.0
        except Exception:
            return 0.0
    
    def vectorized_rolling_mean(self, data: pd.Series, window: int, **kwargs) -> VectorizationResult:
        """Vectorized rolling mean with automatic optimization."""
        start_time = time.time()
        start_memory = self._get_memory_usage_mb(data)
        
        data_size = len(data)
        memory_usage = start_memory
        
        if self._should_use_vectorbt(data_size, memory_usage):
            try:
                result_data = rolling_mean(data, window=window, **kwargs)
                method_used = "VectorBT"
                self.operation_stats['vectorbt_operations'] += 1
                tprint(f"📊 Vectorized rolling mean (VectorBT): window={window}, size={data_size}", "INFO")
            except Exception as e:
                self.logger.warning(f"VectorBT rolling mean failed: {e}, using pandas fallback")
                result_data = data.rolling(window=window).mean(**kwargs)
                method_used = "Pandas (VectorBT fallback)"
                self.operation_stats['pandas_operations'] += 1
        else:
            result_data = data.rolling(window=window).mean(**kwargs)
            method_used = "Pandas"
            self.operation_stats['pandas_operations'] += 1
        
        execution_time = time.time() - start_time
        end_memory = self._get_memory_usage_mb(result_data)
        
        self.operation_stats['total_operations'] += 1
        
        return VectorizationResult(
            data=result_data,
            execution_time=execution_time,
            method_used=method_used,
            memory_used_mb=end_memory,
            metadata={
                'window': window,
                'data_size': data_size,
                'vectorbt_used': method_used.startswith('VectorBT')
            }
        )
    
    def vectorized_rolling_std(self, data: pd.Series, window: int, **kwargs) -> VectorizationResult:
        """Vectorized rolling standard deviation with automatic optimization."""
        start_time = time.time()
        start_memory = self._get_memory_usage_mb(data)
        
        data_size = len(data)
        memory_usage = start_memory
        
        if self._should_use_vectorbt(data_size, memory_usage):
            try:
                result_data = rolling_std(data, window=window, **kwargs)
                method_used = "VectorBT"
                self.operation_stats['vectorbt_operations'] += 1
                tprint(f"📊 Vectorized rolling std (VectorBT): window={window}, size={data_size}", "INFO")
            except Exception as e:
                self.logger.warning(f"VectorBT rolling std failed: {e}, using pandas fallback")
                result_data = data.rolling(window=window).std(**kwargs)
                method_used = "Pandas (VectorBT fallback)"
                self.operation_stats['pandas_operations'] += 1
        else:
            result_data = data.rolling(window=window).std(**kwargs)
            method_used = "Pandas"
            self.operation_stats['pandas_operations'] += 1
        
        execution_time = time.time() - start_time
        end_memory = self._get_memory_usage_mb(result_data)
        
        self.operation_stats['total_operations'] += 1
        
        return VectorizationResult(
            data=result_data,
            execution_time=execution_time,
            method_used=method_used,
            memory_used_mb=end_memory,
            metadata={
                'window': window,
                'data_size': data_size,
                'vectorbt_used': method_used.startswith('VectorBT')
            }
        )
    
    def vectorized_rolling_min(self, data: pd.Series, window: int, **kwargs) -> VectorizationResult:
        """Vectorized rolling minimum with automatic optimization."""
        start_time = time.time()
        start_memory = self._get_memory_usage_mb(data)
        
        data_size = len(data)
        memory_usage = start_memory
        
        if self._should_use_vectorbt(data_size, memory_usage):
            try:
                result_data = rolling_min(data, window=window, **kwargs)
                method_used = "VectorBT"
                self.operation_stats['vectorbt_operations'] += 1
                tprint(f"📊 Vectorized rolling min (VectorBT): window={window}, size={data_size}", "INFO")
            except Exception as e:
                self.logger.warning(f"VectorBT rolling min failed: {e}, using pandas fallback")
                result_data = data.rolling(window=window).min(**kwargs)
                method_used = "Pandas (VectorBT fallback)"
                self.operation_stats['pandas_operations'] += 1
        else:
            result_data = data.rolling(window=window).min(**kwargs)
            method_used = "Pandas"
            self.operation_stats['pandas_operations'] += 1
        
        execution_time = time.time() - start_time
        end_memory = self._get_memory_usage_mb(result_data)
        
        self.operation_stats['total_operations'] += 1
        
        return VectorizationResult(
            data=result_data,
            execution_time=execution_time,
            method_used=method_used,
            memory_used_mb=end_memory,
            metadata={
                'window': window,
                'data_size': data_size,
                'vectorbt_used': method_used.startswith('VectorBT')
            }
        )
    
    def vectorized_rolling_max(self, data: pd.Series, window: int, **kwargs) -> VectorizationResult:
        """Vectorized rolling maximum with automatic optimization."""
        start_time = time.time()
        start_memory = self._get_memory_usage_mb(data)
        
        data_size = len(data)
        memory_usage = start_memory
        
        if self._should_use_vectorbt(data_size, memory_usage):
            try:
                result_data = rolling_max(data, window=window, **kwargs)
                method_used = "VectorBT"
                self.operation_stats['vectorbt_operations'] += 1
                tprint(f"📊 Vectorized rolling max (VectorBT): window={window}, size={data_size}", "INFO")
            except Exception as e:
                self.logger.warning(f"VectorBT rolling max failed: {e}, using pandas fallback")
                result_data = data.rolling(window=window).max(**kwargs)
                method_used = "Pandas (VectorBT fallback)"
                self.operation_stats['pandas_operations'] += 1
        else:
            result_data = data.rolling(window=window).max(**kwargs)
            method_used = "Pandas"
            self.operation_stats['pandas_operations'] += 1
        
        execution_time = time.time() - start_time
        end_memory = self._get_memory_usage_mb(result_data)
        
        self.operation_stats['total_operations'] += 1
        
        return VectorizationResult(
            data=result_data,
            execution_time=execution_time,
            method_used=method_used,
            memory_used_mb=end_memory,
            metadata={
                'window': window,
                'data_size': data_size,
                'vectorbt_used': method_used.startswith('VectorBT')
            }
        )
    
    def vectorized_rolling_sum(self, data: pd.Series, window: int, **kwargs) -> VectorizationResult:
        """Vectorized rolling sum with automatic optimization."""
        start_time = time.time()
        start_memory = self._get_memory_usage_mb(data)
        
        data_size = len(data)
        memory_usage = start_memory
        
        if self._should_use_vectorbt(data_size, memory_usage):
            try:
                result_data = rolling_sum(data, window=window, **kwargs)
                method_used = "VectorBT"
                self.operation_stats['vectorbt_operations'] += 1
                tprint(f"📊 Vectorized rolling sum (VectorBT): window={window}, size={data_size}", "INFO")
            except Exception as e:
                self.logger.warning(f"VectorBT rolling sum failed: {e}, using pandas fallback")
                result_data = data.rolling(window=window).sum(**kwargs)
                method_used = "Pandas (VectorBT fallback)"
                self.operation_stats['pandas_operations'] += 1
        else:
            result_data = data.rolling(window=window).sum(**kwargs)
            method_used = "Pandas"
            self.operation_stats['pandas_operations'] += 1
        
        execution_time = time.time() - start_time
        end_memory = self._get_memory_usage_mb(result_data)
        
        self.operation_stats['total_operations'] += 1
        
        return VectorizationResult(
            data=result_data,
            execution_time=execution_time,
            method_used=method_used,
            memory_used_mb=end_memory,
            metadata={
                'window': window,
                'data_size': data_size,
                'vectorbt_used': method_used.startswith('VectorBT')
            }
        )
    
    def vectorized_rolling_apply(self, data: pd.Series, func: Callable, window: int, **kwargs) -> VectorizationResult:
        """Vectorized rolling apply with automatic optimization."""
        start_time = time.time()
        start_memory = self._get_memory_usage_mb(data)
        
        data_size = len(data)
        memory_usage = start_memory
        
        if self._should_use_vectorbt(data_size, memory_usage):
            try:
                result_data = rolling_apply(data, func, window=window, **kwargs)
                method_used = "VectorBT"
                self.operation_stats['vectorbt_operations'] += 1
                tprint(f"📊 Vectorized rolling apply (VectorBT): window={window}, func={func.__name__}, size={data_size}", "INFO")
            except Exception as e:
                self.logger.warning(f"VectorBT rolling apply failed: {e}, using pandas fallback")
                result_data = data.rolling(window=window).apply(func, **kwargs)
                method_used = "Pandas (VectorBT fallback)"
                self.operation_stats['pandas_operations'] += 1
        else:
            result_data = data.rolling(window=window).apply(func, **kwargs)
            method_used = "Pandas"
            self.operation_stats['pandas_operations'] += 1
        
        execution_time = time.time() - start_time
        end_memory = self._get_memory_usage_mb(result_data)
        
        self.operation_stats['total_operations'] += 1
        
        return VectorizationResult(
            data=result_data,
            execution_time=execution_time,
            method_used=method_used,
            memory_used_mb=end_memory,
            metadata={
                'window': window,
                'func_name': func.__name__,
                'data_size': data_size,
                'vectorbt_used': method_used.startswith('VectorBT')
            }
        )
    
    def vectorized_correlation(self, data1: pd.Series, data2: pd.Series, window: int, **kwargs) -> VectorizationResult:
        """Vectorized rolling correlation with automatic optimization."""
        start_time = time.time()
        start_memory = self._get_memory_usage_mb(data1) + self._get_memory_usage_mb(data2)
        
        data_size = len(data1)
        memory_usage = start_memory
        
        if self._should_use_vectorbt(data_size, memory_usage) and len(data1) == len(data2):
            try:
                result_data = rolling_corr(data1, data2, window=window, **kwargs)
                method_used = "VectorBT"
                self.operation_stats['vectorbt_operations'] += 1
                tprint(f"📊 Vectorized rolling corr (VectorBT): window={window}, size={data_size}", "INFO")
            except Exception as e:
                self.logger.warning(f"VectorBT rolling corr failed: {e}, using pandas fallback")
                result_data = data1.rolling(window=window).corr(data2, **kwargs)
                method_used = "Pandas (VectorBT fallback)"
                self.operation_stats['pandas_operations'] += 1
        else:
            result_data = data1.rolling(window=window).corr(data2, **kwargs)
            method_used = "Pandas"
            self.operation_stats['pandas_operations'] += 1
        
        execution_time = time.time() - start_time
        end_memory = self._get_memory_usage_mb(result_data)
        
        self.operation_stats['total_operations'] += 1
        
        return VectorizationResult(
            data=result_data,
            execution_time=execution_time,
            method_used=method_used,
            memory_used_mb=end_memory,
            metadata={
                'window': window,
                'data_size': data_size,
                'vectorbt_used': method_used.startswith('VectorBT')
            }
        )
    
    def vectorized_statistical_operations(self, data: pd.Series, operations: List[str], window: int = None) -> Dict[str, VectorizationResult]:
        """Perform multiple vectorized statistical operations efficiently."""
        results = {}
        
        tprint(f"📊 Performing {len(operations)} vectorized statistical operations", "INFO")
        
        for operation in operations:
            if operation == 'mean' and window:
                results['mean'] = self.vectorized_rolling_mean(data, window)
            elif operation == 'std' and window:
                results['std'] = self.vectorized_rolling_std(data, window)
            elif operation == 'min' and window:
                results['min'] = self.vectorized_rolling_min(data, window)
            elif operation == 'max' and window:
                results['max'] = self.vectorized_rolling_max(data, window)
            elif operation == 'sum' and window:
                results['sum'] = self.vectorized_rolling_sum(data, window)
            else:
                self.logger.warning(f"Unsupported operation: {operation}")
        
        return results
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        total_operations = self.operation_stats['total_operations']
        
        if total_operations > 0:
            vectorbt_ratio = self.operation_stats['vectorbt_operations'] / total_operations
            numpy_ratio = self.operation_stats['numpy_operations'] / total_operations
            pandas_ratio = self.operation_stats['pandas_operations'] / total_operations
            parallel_ratio = self.operation_stats['parallel_operations'] / total_operations
        else:
            vectorbt_ratio = numpy_ratio = pandas_ratio = parallel_ratio = 0.0
        
        summary = {
            'total_operations': total_operations,
            'vectorbt_operations': self.operation_stats['vectorbt_operations'],
            'numpy_operations': self.operation_stats['numpy_operations'],
            'pandas_operations': self.operation_stats['pandas_operations'],
            'parallel_operations': self.operation_stats['parallel_operations'],
            'vectorbt_ratio': vectorbt_ratio,
            'numpy_ratio': numpy_ratio,
            'pandas_ratio': pandas_ratio,
            'parallel_ratio': parallel_ratio,
            'total_time_saved': self.operation_stats['total_time_saved'],
            'memory_optimizations': self.operation_stats['memory_optimizations'],
            'vectorbt_available': VECTORBT_AVAILABLE,
            'config': {
                'enable_vectorbt': self.config.enable_vectorbt,
                'enable_parallel': self.config.enable_parallel,
                'max_workers': self.config.max_workers,
                'memory_threshold_mb': self.config.memory_threshold_mb,
                'performance_threshold': self.config.performance_threshold
            }
        }
        
        tprint(f"📈 Performance Summary: {vectorbt_ratio:.1%} VectorBT, {numpy_ratio:.1%} NumPy, {pandas_ratio:.1%} Pandas, {parallel_ratio:.1%} Parallel", "INFO")
        return summary
    
    def reset_stats(self):
        """Reset performance statistics."""
        self.operation_stats = {
            'total_operations': 0,
            'vectorbt_operations': 0,
            'numpy_operations': 0,
            'pandas_operations': 0,
            'parallel_operations': 0,
            'total_time_saved': 0.0,
            'memory_optimizations': 0
        }
        tprint("🔄 Performance statistics reset", "INFO")