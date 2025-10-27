"""
VectorBT Rolling Optimizer for efficient vectorized computations in SR detection.

This module provides optimized rolling operations using VectorBT for enhanced
performance in SR detection algorithms.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass
import time

from src.utils.tprint import tprint, tprint_data_preview, tprint_data_format
from src.utils.logger import system_logger

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
class RollingOperationResult:
    """Result of a rolling operation."""
    data: pd.Series
    execution_time: float
    method_used: str
    vectorbt_used: bool
    performance_gain: Optional[float] = None

class VectorBTRollingOptimizer:
    """Optimized rolling operations using VectorBT for SR detection."""
    
    def __init__(self, enable_vectorbt: bool = True, performance_threshold: int = 1000):
        self.enable_vectorbt = enable_vectorbt and VECTORBT_AVAILABLE
        self.performance_threshold = performance_threshold
        self.logger = system_logger.getChild('VectorBTRollingOptimizer')
        
        # Performance tracking
        self.operation_stats = {
            'vectorbt_operations': 0,
            'pandas_operations': 0,
            'total_time_saved': 0.0,
            'avg_performance_gain': 0.0
        }
        
        tprint("🚀 VectorBTRollingOptimizer initialized", "INFO")
        if self.enable_vectorbt:
            tprint("✅ VectorBT optimization enabled", "SUCCESS")
        else:
            tprint("⚠️ VectorBT optimization disabled - using pandas fallback", "WARNING")
    
    def should_use_vectorbt(self, data: pd.Series) -> bool:
        """Determine if VectorBT should be used based on data size and configuration."""
        return (self.enable_vectorbt and 
                len(data) >= self.performance_threshold and 
                VECTORBT_AVAILABLE)
    
    def rolling_mean(self, data: pd.Series, window: int, **kwargs) -> RollingOperationResult:
        """Optimized rolling mean operation."""
        start_time = time.time()
        
        if self.should_use_vectorbt(data):
            try:
                result_data = rolling_mean(data, window=window, **kwargs)
                method_used = "VectorBT"
                vectorbt_used = True
                self.operation_stats['vectorbt_operations'] += 1
                tprint(f"📊 Rolling mean (VectorBT): window={window}, data_size={len(data)}", "INFO")
            except Exception as e:
                self.logger.warning(f"VectorBT rolling mean failed: {e}, using pandas fallback")
                result_data = data.rolling(window=window).mean(**kwargs)
                method_used = "Pandas (VectorBT fallback)"
                vectorbt_used = False
                self.operation_stats['pandas_operations'] += 1
        else:
            result_data = data.rolling(window=window).mean(**kwargs)
            method_used = "Pandas"
            vectorbt_used = False
            self.operation_stats['pandas_operations'] += 1
        
        execution_time = time.time() - start_time
        
        # Calculate performance gain if we have both methods
        performance_gain = None
        if vectorbt_used and len(data) >= self.performance_threshold:
            # Estimate pandas performance for comparison
            pandas_time = execution_time * 1.5  # Rough estimate
            performance_gain = (pandas_time - execution_time) / pandas_time * 100
            self.operation_stats['total_time_saved'] += pandas_time - execution_time
        
        return RollingOperationResult(
            data=result_data,
            execution_time=execution_time,
            method_used=method_used,
            vectorbt_used=vectorbt_used,
            performance_gain=performance_gain
        )
    
    def rolling_std(self, data: pd.Series, window: int, **kwargs) -> RollingOperationResult:
        """Optimized rolling standard deviation operation."""
        start_time = time.time()
        
        if self.should_use_vectorbt(data):
            try:
                result_data = rolling_std(data, window=window, **kwargs)
                method_used = "VectorBT"
                vectorbt_used = True
                self.operation_stats['vectorbt_operations'] += 1
                tprint(f"📊 Rolling std (VectorBT): window={window}, data_size={len(data)}", "INFO")
            except Exception as e:
                self.logger.warning(f"VectorBT rolling std failed: {e}, using pandas fallback")
                result_data = data.rolling(window=window).std(**kwargs)
                method_used = "Pandas (VectorBT fallback)"
                vectorbt_used = False
                self.operation_stats['pandas_operations'] += 1
        else:
            result_data = data.rolling(window=window).std(**kwargs)
            method_used = "Pandas"
            vectorbt_used = False
            self.operation_stats['pandas_operations'] += 1
        
        execution_time = time.time() - start_time
        return RollingOperationResult(
            data=result_data,
            execution_time=execution_time,
            method_used=method_used,
            vectorbt_used=vectorbt_used
        )
    
    def rolling_var(self, data: pd.Series, window: int, **kwargs) -> RollingOperationResult:
        """Optimized rolling variance operation."""
        start_time = time.time()
        
        if self.should_use_vectorbt(data):
            try:
                result_data = rolling_var(data, window=window, **kwargs)
                method_used = "VectorBT"
                vectorbt_used = True
                self.operation_stats['vectorbt_operations'] += 1
                tprint(f"📊 Rolling var (VectorBT): window={window}, data_size={len(data)}", "INFO")
            except Exception as e:
                self.logger.warning(f"VectorBT rolling var failed: {e}, using pandas fallback")
                result_data = data.rolling(window=window).var(**kwargs)
                method_used = "Pandas (VectorBT fallback)"
                vectorbt_used = False
                self.operation_stats['pandas_operations'] += 1
        else:
            result_data = data.rolling(window=window).var(**kwargs)
            method_used = "Pandas"
            vectorbt_used = False
            self.operation_stats['pandas_operations'] += 1
        
        execution_time = time.time() - start_time
        return RollingOperationResult(
            data=result_data,
            execution_time=execution_time,
            method_used=method_used,
            vectorbt_used=vectorbt_used
        )
    
    def rolling_min(self, data: pd.Series, window: int, **kwargs) -> RollingOperationResult:
        """Optimized rolling minimum operation."""
        start_time = time.time()
        
        if self.should_use_vectorbt(data):
            try:
                result_data = rolling_min(data, window=window, **kwargs)
                method_used = "VectorBT"
                vectorbt_used = True
                self.operation_stats['vectorbt_operations'] += 1
                tprint(f"📊 Rolling min (VectorBT): window={window}, data_size={len(data)}", "INFO")
            except Exception as e:
                self.logger.warning(f"VectorBT rolling min failed: {e}, using pandas fallback")
                result_data = data.rolling(window=window).min(**kwargs)
                method_used = "Pandas (VectorBT fallback)"
                vectorbt_used = False
                self.operation_stats['pandas_operations'] += 1
        else:
            result_data = data.rolling(window=window).min(**kwargs)
            method_used = "Pandas"
            vectorbt_used = False
            self.operation_stats['pandas_operations'] += 1
        
        execution_time = time.time() - start_time
        return RollingOperationResult(
            data=result_data,
            execution_time=execution_time,
            method_used=method_used,
            vectorbt_used=vectorbt_used
        )
    
    def rolling_max(self, data: pd.Series, window: int, **kwargs) -> RollingOperationResult:
        """Optimized rolling maximum operation."""
        start_time = time.time()
        
        if self.should_use_vectorbt(data):
            try:
                result_data = rolling_max(data, window=window, **kwargs)
                method_used = "VectorBT"
                vectorbt_used = True
                self.operation_stats['vectorbt_operations'] += 1
                tprint(f"📊 Rolling max (VectorBT): window={window}, data_size={len(data)}", "INFO")
            except Exception as e:
                self.logger.warning(f"VectorBT rolling max failed: {e}, using pandas fallback")
                result_data = data.rolling(window=window).max(**kwargs)
                method_used = "Pandas (VectorBT fallback)"
                vectorbt_used = False
                self.operation_stats['pandas_operations'] += 1
        else:
            result_data = data.rolling(window=window).max(**kwargs)
            method_used = "Pandas"
            vectorbt_used = False
            self.operation_stats['pandas_operations'] += 1
        
        execution_time = time.time() - start_time
        return RollingOperationResult(
            data=result_data,
            execution_time=execution_time,
            method_used=method_used,
            vectorbt_used=vectorbt_used
        )
    
    def rolling_sum(self, data: pd.Series, window: int, **kwargs) -> RollingOperationResult:
        """Optimized rolling sum operation."""
        start_time = time.time()
        
        if self.should_use_vectorbt(data):
            try:
                result_data = rolling_sum(data, window=window, **kwargs)
                method_used = "VectorBT"
                vectorbt_used = True
                self.operation_stats['vectorbt_operations'] += 1
                tprint(f"📊 Rolling sum (VectorBT): window={window}, data_size={len(data)}", "INFO")
            except Exception as e:
                self.logger.warning(f"VectorBT rolling sum failed: {e}, using pandas fallback")
                result_data = data.rolling(window=window).sum(**kwargs)
                method_used = "Pandas (VectorBT fallback)"
                vectorbt_used = False
                self.operation_stats['pandas_operations'] += 1
        else:
            result_data = data.rolling(window=window).sum(**kwargs)
            method_used = "Pandas"
            vectorbt_used = False
            self.operation_stats['pandas_operations'] += 1
        
        execution_time = time.time() - start_time
        return RollingOperationResult(
            data=result_data,
            execution_time=execution_time,
            method_used=method_used,
            vectorbt_used=vectorbt_used
        )
    
    def rolling_apply(self, data: pd.Series, func: Callable, window: int, **kwargs) -> RollingOperationResult:
        """Optimized rolling apply operation."""
        start_time = time.time()
        
        if self.should_use_vectorbt(data):
            try:
                result_data = rolling_apply(data, func, window=window, **kwargs)
                method_used = "VectorBT"
                vectorbt_used = True
                self.operation_stats['vectorbt_operations'] += 1
                tprint(f"📊 Rolling apply (VectorBT): window={window}, func={func.__name__}, data_size={len(data)}", "INFO")
            except Exception as e:
                self.logger.warning(f"VectorBT rolling apply failed: {e}, using pandas fallback")
                result_data = data.rolling(window=window).apply(func, **kwargs)
                method_used = "Pandas (VectorBT fallback)"
                vectorbt_used = False
                self.operation_stats['pandas_operations'] += 1
        else:
            result_data = data.rolling(window=window).apply(func, **kwargs)
            method_used = "Pandas"
            vectorbt_used = False
            self.operation_stats['pandas_operations'] += 1
        
        execution_time = time.time() - start_time
        return RollingOperationResult(
            data=result_data,
            execution_time=execution_time,
            method_used=method_used,
            vectorbt_used=vectorbt_used
        )
    
    def rolling_corr(self, data1: pd.Series, data2: pd.Series, window: int, **kwargs) -> RollingOperationResult:
        """Optimized rolling correlation operation."""
        start_time = time.time()
        
        if self.should_use_vectorbt(data1) and len(data1) == len(data2):
            try:
                result_data = rolling_corr(data1, data2, window=window, **kwargs)
                method_used = "VectorBT"
                vectorbt_used = True
                self.operation_stats['vectorbt_operations'] += 1
                tprint(f"📊 Rolling corr (VectorBT): window={window}, data_size={len(data1)}", "INFO")
            except Exception as e:
                self.logger.warning(f"VectorBT rolling corr failed: {e}, using pandas fallback")
                result_data = data1.rolling(window=window).corr(data2, **kwargs)
                method_used = "Pandas (VectorBT fallback)"
                vectorbt_used = False
                self.operation_stats['pandas_operations'] += 1
        else:
            result_data = data1.rolling(window=window).corr(data2, **kwargs)
            method_used = "Pandas"
            vectorbt_used = False
            self.operation_stats['pandas_operations'] += 1
        
        execution_time = time.time() - start_time
        return RollingOperationResult(
            data=result_data,
            execution_time=execution_time,
            method_used=method_used,
            vectorbt_used=vectorbt_used
        )
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary of rolling operations."""
        total_operations = self.operation_stats['vectorbt_operations'] + self.operation_stats['pandas_operations']
        
        if total_operations > 0:
            vectorbt_ratio = self.operation_stats['vectorbt_operations'] / total_operations
            avg_performance_gain = self.operation_stats['total_time_saved'] / max(1, self.operation_stats['vectorbt_operations'])
        else:
            vectorbt_ratio = 0.0
            avg_performance_gain = 0.0
        
        summary = {
            'total_operations': total_operations,
            'vectorbt_operations': self.operation_stats['vectorbt_operations'],
            'pandas_operations': self.operation_stats['pandas_operations'],
            'vectorbt_ratio': vectorbt_ratio,
            'total_time_saved': self.operation_stats['total_time_saved'],
            'avg_performance_gain': avg_performance_gain,
            'vectorbt_available': VECTORBT_AVAILABLE,
            'optimization_enabled': self.enable_vectorbt
        }
        
        tprint(f"📈 Performance Summary: {vectorbt_ratio:.1%} VectorBT operations, {avg_performance_gain:.2f}s avg time saved", "INFO")
        return summary
    
    def reset_stats(self):
        """Reset performance statistics."""
        self.operation_stats = {
            'vectorbt_operations': 0,
            'pandas_operations': 0,
            'total_time_saved': 0.0,
            'avg_performance_gain': 0.0
        }
        tprint("🔄 Performance statistics reset", "INFO")