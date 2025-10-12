"""
Unified Vectorization Manager

This module provides a centralized manager that coordinates all VectorBT and vectorization
optimizations for maximum performance and consistency across the feature generation pipeline.

Key Features:
- Unified interface for all optimization components
- Intelligent method selection based on data characteristics
- Comprehensive performance monitoring
- Memory management and batch processing
- Graceful fallbacks and error handling
"""

import numpy as np
import pandas as pd
import logging
import time
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
from contextlib import contextmanager
from dataclasses import dataclass
import warnings

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

# Import optimization components
from .vectorbt_rolling_optimizer import get_vectorbt_rolling_optimizer, VectorBTRollingOptimizer
from .vectorization_optimizer import get_vectorization_optimizer, VectorizationOptimizer, VectorizationConfig

logger = logging.getLogger(__name__)

@dataclass
class OptimizationConfig:
    """Configuration for unified vectorization optimization."""
    # VectorBT Configuration
    enable_vectorbt: bool = True
    vectorbt_threshold: int = 1000  # Minimum rows for VectorBT
    enable_gpu: bool = False
    enable_parallel: bool = True
    
    # Memory Management
    memory_limit_gb: float = 8.0
    enable_memory_optimization: bool = True
    chunk_size: int = 10000
    
    # Performance Monitoring
    enable_profiling: bool = True
    enable_caching: bool = True
    cache_size: int = 1000
    
    # Batch Processing
    enable_batch_processing: bool = True
    batch_size: int = 1000
    max_workers: int = None  # Auto-detect

class UnifiedVectorizationManager:
    """
    Unified manager for all VectorBT and vectorization optimizations.
    
    This class provides a single interface to coordinate all optimization components
    and ensure consistent, high-performance feature generation across the pipeline.
    """
    
    def __init__(self, config: Optional[OptimizationConfig] = None):
        """
        Initialize the unified vectorization manager.
        
        Args:
            config: Optimization configuration
        """
        self.config = config or OptimizationConfig()
        self.logger = logger.getChild('UnifiedVectorizationManager')
        
        # Initialize optimization components
        self.rolling_optimizer = None
        self.vectorization_optimizer = None
        
        # Performance tracking
        self.performance_stats = {
            'total_operations': 0,
            'vectorbt_operations': 0,
            'vectorization_operations': 0,
            'batch_operations': 0,
            'memory_optimizations': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'total_time': 0.0,
            'fallback_operations': 0
        }
        
        # Cache for computed results
        self._cache = {}
        self._cache_enabled = self.config.enable_caching
        
        # Initialize components
        self._initialize_components()
        
        self.logger.info("✅ Unified Vectorization Manager initialized")
    
    def _initialize_components(self):
        """Initialize all optimization components."""
        try:
            # Initialize VectorBT rolling optimizer
            if self.config.enable_vectorbt and VECTORBT_AVAILABLE:
                self.rolling_optimizer = get_vectorbt_rolling_optimizer(
                    enable_gpu=self.config.enable_gpu,
                    enable_parallel=self.config.enable_parallel
                )
                self.logger.info("✅ VectorBT Rolling Optimizer initialized")
            
            # Initialize vectorization optimizer
            vectorization_config = VectorizationConfig(
                enable_gpu_acceleration=self.config.enable_gpu,
                enable_parallel=self.config.enable_parallel,
                memory_limit_gb=self.config.memory_limit_gb,
                chunk_size=self.config.chunk_size,
                enable_profiling=self.config.enable_profiling
            )
            self.vectorization_optimizer = get_vectorization_optimizer(vectorization_config)
            self.logger.info("✅ Vectorization Optimizer initialized")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Some optimization components not available: {e}")
    
    def optimize_dataframe(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Optimize DataFrame for vectorized processing.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Optimized DataFrame
        """
        start_time = time.time()
        
        try:
            # Use vectorization optimizer for comprehensive optimization
            if self.vectorization_optimizer:
                optimized_data = self.vectorization_optimizer.optimize_dataframe_processing(data)
                self.performance_stats['vectorization_operations'] += 1
            else:
                optimized_data = self._basic_dataframe_optimization(data)
            
            # Additional VectorBT-specific optimizations
            if self.config.enable_vectorbt and VECTORBT_AVAILABLE:
                optimized_data = self._optimize_for_vectorbt(optimized_data)
            
            self.performance_stats['memory_optimizations'] += 1
            self.performance_stats['total_time'] += time.time() - start_time
            
            return optimized_data
            
        except Exception as e:
            self.logger.warning(f"DataFrame optimization failed: {e}")
            return data
    
    def _basic_dataframe_optimization(self, data: pd.DataFrame) -> pd.DataFrame:
        """Basic DataFrame optimization without advanced components."""
        optimized_data = data.copy()
        
        # Optimize numeric columns
        for col in optimized_data.select_dtypes(include=[np.number]).columns:
            if optimized_data[col].dtype == np.float64:
                if (optimized_data[col].max() < np.finfo(np.float32).max and
                    optimized_data[col].min() > np.finfo(np.float32).min):
                    optimized_data[col] = optimized_data[col].astype(np.float32)
            elif optimized_data[col].dtype == np.int64:
                if (optimized_data[col].max() < np.iinfo(np.int32).max and
                    optimized_data[col].min() > np.iinfo(np.int32).min):
                    optimized_data[col] = optimized_data[col].astype(np.int32)
        
        return optimized_data
    
    def _optimize_for_vectorbt(self, data: pd.DataFrame) -> pd.DataFrame:
        """Additional optimizations specifically for VectorBT."""
        try:
            optimized_data = data.copy()
            
            # Convert to VectorBT array wrappers for better performance
            for column in optimized_data.columns:
                if optimized_data[column].dtype in ['float32', 'float64', 'int32', 'int64']:
                    try:
                        optimized_data[column] = vbt.array_wrapper(
                            optimized_data[column],
                            freq=data.index.freq if hasattr(data.index, 'freq') else None
                        )
                    except Exception as e:
                        self.logger.debug(f"VectorBT array wrapper conversion failed for {column}: {e}")
                        continue
            
            return optimized_data
            
        except Exception as e:
            self.logger.warning(f"VectorBT optimization failed: {e}")
            return data
    
    def rolling_operation(self, data: Union[pd.Series, pd.DataFrame], 
                        operation: str, window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """
        Perform optimized rolling operation with intelligent method selection.
        
        Args:
            data: Input data (Series or DataFrame)
            operation: Operation type ('mean', 'std', 'var', 'min', 'max', 'sum', etc.)
            window: Rolling window size
            **kwargs: Additional parameters
            
        Returns:
            Result of rolling operation
        """
        start_time = time.time()
        self.performance_stats['total_operations'] += 1
        
        # Check cache first
        if self._cache_enabled:
            cache_key = self._generate_cache_key(data, operation, window, **kwargs)
            cached_result = self._get_from_cache(cache_key)
            if cached_result is not None:
                self.performance_stats['cache_hits'] += 1
                return cached_result
            self.performance_stats['cache_misses'] += 1
        
        try:
            # Determine optimal method
            if self._should_use_vectorbt(data, window):
                result = self._vectorbt_rolling_operation(data, operation, window, **kwargs)
                self.performance_stats['vectorbt_operations'] += 1
            elif self.vectorization_optimizer:
                result = self._vectorization_rolling_operation(data, operation, window, **kwargs)
                self.performance_stats['vectorization_operations'] += 1
            else:
                result = self._pandas_rolling_operation(data, operation, window, **kwargs)
                self.performance_stats['fallback_operations'] += 1
            
            # Cache result
            if self._cache_enabled:
                self._put_in_cache(cache_key, result)
            
            self.performance_stats['total_time'] += time.time() - start_time
            return result
            
        except Exception as e:
            self.logger.warning(f"Rolling operation failed: {e}, using fallback")
            result = self._pandas_rolling_operation(data, operation, window, **kwargs)
            self.performance_stats['fallback_operations'] += 1
            self.performance_stats['total_time'] += time.time() - start_time
            return result
    
    def _should_use_vectorbt(self, data: Union[pd.Series, pd.DataFrame], window: int) -> bool:
        """Determine if VectorBT should be used for this operation."""
        if not self.config.enable_vectorbt or not VECTORBT_AVAILABLE or not self.rolling_optimizer:
            return False
        
        data_size = len(data) if hasattr(data, '__len__') else 0
        return data_size >= self.config.vectorbt_threshold
    
    def _vectorbt_rolling_operation(self, data: Union[pd.Series, pd.DataFrame], 
                                  operation: str, window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Perform rolling operation using VectorBT."""
        return self.rolling_optimizer._rolling_operation(data, operation, window, **kwargs)
    
    def _vectorization_rolling_operation(self, data: Union[pd.Series, pd.DataFrame], 
                                       operation: str, window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Perform rolling operation using vectorization optimizer."""
        if isinstance(data, pd.Series):
            return self.vectorization_optimizer._vectorbt_rolling_operation(data, operation, window, **kwargs)
        else:
            # For DataFrame, process each column
            results = {}
            for col in data.columns:
                results[col] = self.vectorization_optimizer._vectorbt_rolling_operation(
                    data[col], operation, window, **kwargs
                )
            return pd.DataFrame(results, index=data.index)
    
    def _pandas_rolling_operation(self, data: Union[pd.Series, pd.DataFrame], 
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
        else:
            raise ValueError(f"Unsupported operation: {operation}")
    
    def technical_indicator(self, data: pd.DataFrame, indicator: str, **kwargs) -> pd.Series:
        """
        Calculate technical indicator using VectorBT native implementations.
        
        Args:
            data: OHLCV data
            indicator: Indicator name
            **kwargs: Indicator parameters
            
        Returns:
            Indicator values
        """
        if not self.config.enable_vectorbt or not VECTORBT_AVAILABLE:
            return self._pandas_technical_indicator(data, indicator, **kwargs)
        
        try:
            if indicator == 'rsi':
                return vbt.RSI.run(data['close'], **kwargs).rsi
            elif indicator == 'macd':
                return vbt.MACD.run(data['close'], **kwargs).macd
            elif indicator == 'macd_signal':
                return vbt.MACD.run(data['close'], **kwargs).signal
            elif indicator == 'macd_histogram':
                return vbt.MACD.run(data['close'], **kwargs).histogram
            elif indicator == 'atr':
                return vbt.ATR.run(data['high'], data['low'], data['close'], **kwargs).atr
            elif indicator == 'bbands_upper':
                return vbt.BBANDS.run(data['close'], **kwargs).upper
            elif indicator == 'bbands_middle':
                return vbt.BBANDS.run(data['close'], **kwargs).middle
            elif indicator == 'bbands_lower':
                return vbt.BBANDS.run(data['close'], **kwargs).lower
            elif indicator == 'bbands_width':
                return vbt.BBANDS.run(data['close'], **kwargs).width
            elif indicator == 'bbands_percent':
                return vbt.BBANDS.run(data['close'], **kwargs).percent
            elif indicator == 'stoch_k':
                return vbt.STOCH.run(data['high'], data['low'], data['close'], **kwargs).stoch_k
            elif indicator == 'stoch_d':
                return vbt.STOCH.run(data['high'], data['low'], data['close'], **kwargs).stoch_d
            elif indicator == 'obv':
                return vbt.OBV.run(data['close'], data['volume'], **kwargs).obv
            elif indicator == 'sma':
                return vbt.MA.run(data['close'], **kwargs).ma
            elif indicator == 'ema':
                return vbt.EMA.run(data['close'], **kwargs).ema
            elif indicator == 'wma':
                return vbt.WMA.run(data['close'], **kwargs).wma
            elif indicator == 'willr':
                return vbt.WILLR.run(data['high'], data['low'], data['close'], **kwargs).willr
            elif indicator == 'cci':
                return vbt.CCI.run(data['high'], data['low'], data['close'], **kwargs).cci
            elif indicator == 'mfi':
                return vbt.MFI.run(data['high'], data['low'], data['close'], data['volume'], **kwargs).mfi
            elif indicator == 'adx':
                return vbt.ADX.run(data['high'], data['low'], data['close'], **kwargs).adx
            elif indicator == 'roc':
                return vbt.ROC.run(data['close'], **kwargs).roc
            elif indicator == 'mom':
                return vbt.MOM.run(data['close'], **kwargs).mom
            else:
                raise ValueError(f"Unsupported indicator: {indicator}")
        
        except Exception as e:
            self.logger.warning(f"VectorBT indicator {indicator} failed: {e}, using fallback")
            return self._pandas_technical_indicator(data, indicator, **kwargs)
    
    def _pandas_technical_indicator(self, data: pd.DataFrame, indicator: str, **kwargs) -> pd.Series:
        """Fallback technical indicator calculation using pandas."""
        if indicator == 'sma':
            return data['close'].rolling(window=kwargs.get('window', 20)).mean()
        elif indicator == 'ema':
            return data['close'].ewm(span=kwargs.get('window', 20)).mean()
        elif indicator == 'rsi':
            # Simple RSI calculation
            delta = data['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            window = kwargs.get('window', 14)
            avg_gain = gain.rolling(window=window).mean()
            avg_loss = loss.rolling(window=window).mean()
            rs = avg_gain / avg_loss
            return 100 - (100 / (1 + rs))
        else:
            # Return NaN series for unsupported indicators
            return pd.Series(np.nan, index=data.index)
    
    def batch_operations(self, data: pd.DataFrame, 
                        operations: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Perform multiple operations in batch for efficiency.
        
        Args:
            data: Input DataFrame
            operations: List of operation dictionaries
            
        Returns:
            DataFrame with results
        """
        if not self.config.enable_batch_processing:
            return self._sequential_operations(data, operations)
        
        start_time = time.time()
        self.performance_stats['batch_operations'] += 1
        
        try:
            results = {}
            
            # Group operations by type for efficient processing
            rolling_ops = [op for op in operations if op.get('type') == 'rolling']
            indicator_ops = [op for op in operations if op.get('type') == 'indicator']
            scaling_ops = [op for op in operations if op.get('type') == 'scaling']
            
            # Process rolling operations
            if rolling_ops:
                rolling_results = self._process_rolling_operations_batch(data, rolling_ops)
                results.update(rolling_results)
            
            # Process indicator operations
            if indicator_ops:
                indicator_results = self._process_indicator_operations_batch(data, indicator_ops)
                results.update(indicator_results)
            
            # Process scaling operations
            if scaling_ops:
                scaling_results = self._process_scaling_operations_batch(data, scaling_ops)
                results.update(scaling_results)
            
            self.performance_stats['total_time'] += time.time() - start_time
            return pd.DataFrame(results, index=data.index)
            
        except Exception as e:
            self.logger.error(f"Batch operations failed: {e}")
            return self._sequential_operations(data, operations)
    
    def _process_rolling_operations_batch(self, data: pd.DataFrame, 
                                        operations: List[Dict[str, Any]]) -> Dict[str, pd.Series]:
        """Process rolling operations in batch."""
        results = {}
        
        for op in operations:
            feature_name = op['name']
            column = op.get('column', 'close')
            operation = op['operation']
            window = op['window']
            
            if column in data.columns:
                results[feature_name] = self.rolling_operation(
                    data[column], operation, window, **op.get('kwargs', {})
                )
            else:
                self.logger.warning(f"Column {column} not found for operation {feature_name}")
                results[feature_name] = pd.Series(np.nan, index=data.index)
        
        return results
    
    def _process_indicator_operations_batch(self, data: pd.DataFrame, 
                                          operations: List[Dict[str, Any]]) -> Dict[str, pd.Series]:
        """Process indicator operations in batch."""
        results = {}
        
        for op in operations:
            feature_name = op['name']
            indicator = op['indicator']
            params = op.get('params', {})
            
            try:
                results[feature_name] = self.technical_indicator(data, indicator, **params)
            except Exception as e:
                self.logger.warning(f"Indicator {indicator} failed: {e}")
                results[feature_name] = pd.Series(np.nan, index=data.index)
        
        return results
    
    def _process_scaling_operations_batch(self, data: pd.DataFrame, 
                                        operations: List[Dict[str, Any]]) -> Dict[str, pd.Series]:
        """Process scaling operations in batch."""
        results = {}
        
        for op in operations:
            feature_name = op['name']
            column = op.get('column', 'close')
            method = op.get('method', 'zscore')
            
            if column in data.columns:
                try:
                    if method == 'zscore':
                        results[feature_name] = zscore(data[column])
                    elif method == 'minmax':
                        results[feature_name] = scale(data[column], method='minmax')
                    elif method == 'robust':
                        results[feature_name] = scale(data[column], method='robust')
                    elif method == 'rank':
                        results[feature_name] = rank(data[column])
                    elif method == 'winsorize':
                        results[feature_name] = winsorize(data[column])
                    else:
                        results[feature_name] = data[column]  # No scaling
                except Exception as e:
                    self.logger.warning(f"Scaling {method} failed: {e}")
                    results[feature_name] = data[column]
            else:
                self.logger.warning(f"Column {column} not found for scaling {feature_name}")
                results[feature_name] = pd.Series(np.nan, index=data.index)
        
        return results
    
    def _sequential_operations(self, data: pd.DataFrame, 
                             operations: List[Dict[str, Any]]) -> pd.DataFrame:
        """Process operations sequentially as fallback."""
        results = {}
        
        for op in operations:
            op_type = op.get('type')
            feature_name = op['name']
            
            try:
                if op_type == 'rolling':
                    column = op.get('column', 'close')
                    operation = op['operation']
                    window = op['window']
                    
                    if column in data.columns:
                        results[feature_name] = self.rolling_operation(
                            data[column], operation, window, **op.get('kwargs', {})
                        )
                    else:
                        results[feature_name] = pd.Series(np.nan, index=data.index)
                
                elif op_type == 'indicator':
                    indicator = op['indicator']
                    params = op.get('params', {})
                    results[feature_name] = self.technical_indicator(data, indicator, **params)
                
                elif op_type == 'scaling':
                    column = op.get('column', 'close')
                    method = op.get('method', 'zscore')
                    
                    if column in data.columns:
                        if method == 'zscore':
                            results[feature_name] = zscore(data[column])
                        elif method == 'minmax':
                            results[feature_name] = scale(data[column], method='minmax')
                        else:
                            results[feature_name] = data[column]
                    else:
                        results[feature_name] = pd.Series(np.nan, index=data.index)
                
            except Exception as e:
                self.logger.warning(f"Operation {feature_name} failed: {e}")
                results[feature_name] = pd.Series(np.nan, index=data.index)
        
        return pd.DataFrame(results, index=data.index)
    
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
            if cache_key in self._cache:
                return self._cache[cache_key]
        except Exception as e:
            self.logger.warning(f"Cache retrieval failed: {e}")
        
        return None
    
    def _put_in_cache(self, cache_key: str, result: Union[pd.Series, pd.DataFrame]):
        """Put result in cache."""
        if not self._cache_enabled:
            return
        
        try:
            # Limit cache size
            if len(self._cache) >= self.config.cache_size:
                # Remove oldest entries (simple FIFO)
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
            
            self._cache[cache_key] = result
            
        except Exception as e:
            self.logger.warning(f"Cache storage failed: {e}")
    
    @contextmanager
    def batch_processing(self):
        """Context manager for batch processing operations."""
        try:
            self.logger.debug("Starting batch processing")
            yield self
        finally:
            self.logger.debug("Batch processing completed")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        stats = self.performance_stats.copy()
        
        if stats['total_operations'] > 0:
            stats['vectorbt_usage_percentage'] = (
                stats['vectorbt_operations'] / stats['total_operations'] * 100
            )
            stats['vectorization_usage_percentage'] = (
                stats['vectorization_operations'] / stats['total_operations'] * 100
            )
            stats['fallback_usage_percentage'] = (
                stats['fallback_operations'] / stats['total_operations'] * 100
            )
            stats['average_operation_time'] = (
                stats['total_time'] / stats['total_operations']
            )
            
            # Cache statistics
            total_cache_ops = stats['cache_hits'] + stats['cache_misses']
            if total_cache_ops > 0:
                stats['cache_hit_rate'] = (stats['cache_hits'] / total_cache_ops) * 100
            else:
                stats['cache_hit_rate'] = 0
        else:
            stats['vectorbt_usage_percentage'] = 0
            stats['vectorization_usage_percentage'] = 0
            stats['fallback_usage_percentage'] = 0
            stats['average_operation_time'] = 0
            stats['cache_hit_rate'] = 0
        
        return stats
    
    def reset_stats(self):
        """Reset performance statistics."""
        self.performance_stats = {
            'total_operations': 0,
            'vectorbt_operations': 0,
            'vectorization_operations': 0,
            'batch_operations': 0,
            'memory_optimizations': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'total_time': 0.0,
            'fallback_operations': 0
        }
    
    def cleanup(self):
        """Cleanup resources and clear cache."""
        try:
            self._cache.clear()
            if self.vectorization_optimizer:
                self.vectorization_optimizer.cleanup()
            self.logger.info("🧹 Unified Vectorization Manager cleanup completed")
        except Exception as e:
            self.logger.error(f"Cleanup error: {e}")

# Global instance
_unified_manager: Optional[UnifiedVectorizationManager] = None

def get_unified_vectorization_manager(config: Optional[OptimizationConfig] = None) -> UnifiedVectorizationManager:
    """Get or create the global unified vectorization manager instance."""
    global _unified_manager
    
    if _unified_manager is None:
        _unified_manager = UnifiedVectorizationManager(config)
    
    return _unified_manager

def optimize_dataframe_unified(data: pd.DataFrame, 
                             config: Optional[OptimizationConfig] = None) -> pd.DataFrame:
    """Convenience function to optimize DataFrame using unified manager."""
    manager = get_unified_vectorization_manager(config)
    return manager.optimize_dataframe(data)

def rolling_operation_unified(data: Union[pd.Series, pd.DataFrame], 
                            operation: str, window: int, 
                            config: Optional[OptimizationConfig] = None, **kwargs) -> Union[pd.Series, pd.DataFrame]:
    """Convenience function for rolling operations using unified manager."""
    manager = get_unified_vectorization_manager(config)
    return manager.rolling_operation(data, operation, window, **kwargs)

def technical_indicator_unified(data: pd.DataFrame, indicator: str, 
                              config: Optional[OptimizationConfig] = None, **kwargs) -> pd.Series:
    """Convenience function for technical indicators using unified manager."""
    manager = get_unified_vectorization_manager(config)
    return manager.technical_indicator(data, indicator, **kwargs)

def batch_operations_unified(data: pd.DataFrame, operations: List[Dict[str, Any]], 
                           config: Optional[OptimizationConfig] = None) -> pd.DataFrame:
    """Convenience function for batch operations using unified manager."""
    manager = get_unified_vectorization_manager(config)
    return manager.batch_operations(data, operations)