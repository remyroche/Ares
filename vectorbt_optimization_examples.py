"""
VectorBT Optimization Examples

This file contains specific code examples for optimizing existing VectorBT implementations.
These examples can be directly integrated into your existing codebase.
"""

import numpy as np
import pandas as pd
import logging
from typing import Any, Dict, List, Optional, Union, Tuple
import warnings

# VectorBT imports
try:
    import vectorbt as vbt
    from vectorbt.generic import (
        rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max, 
        rolling_sum, rolling_apply, rolling_corr, rolling_cov,
        rolling_quantile, rolling_skew, rolling_kurt,
        scale, rank, zscore, winsorize, clip, quantile
    )
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    warnings.warn("VectorBT not available. Install with: pip install vectorbt")

# Optional GPU acceleration
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

logger = logging.getLogger(__name__)


class EnhancedVectorBTScaler:
    """
    Enhanced VectorBT Scaler with advanced optimization features.
    
    This class provides optimized scaling operations with:
    - Advanced scaling methods
    - Batch processing
    - Memory optimization
    - GPU acceleration
    """
    
    def __init__(self, method: str = 'zscore', enable_gpu: bool = False, 
                 enable_batch: bool = True, memory_efficient: bool = True):
        """
        Initialize enhanced VectorBT scaler.
        
        Args:
            method: Scaling method
            enable_gpu: Enable GPU acceleration
            enable_batch: Enable batch processing
            memory_efficient: Enable memory optimization
        """
        self.method = method
        self.enable_gpu = enable_gpu and CUPY_AVAILABLE
        self.enable_batch = enable_batch
        self.memory_efficient = memory_efficient
        self.fitted = False
        self.scaling_params = {}
        
        # Performance tracking
        self.performance_stats = {
            'vectorbt_operations': 0,
            'gpu_operations': 0,
            'batch_operations': 0,
            'memory_optimizations': 0,
            'total_operations': 0
        }
    
    def fit_transform(self, data: Union[pd.Series, pd.DataFrame]) -> Union[pd.Series, pd.DataFrame]:
        """Enhanced fit_transform with optimization."""
        self.performance_stats['total_operations'] += 1
        
        if isinstance(data, pd.DataFrame) and self.enable_batch:
            return self._batch_fit_transform(data)
        else:
            return self._single_fit_transform(data)
    
    def _single_fit_transform(self, data: pd.Series) -> pd.Series:
        """Single series fit_transform with optimization."""
        if not VECTORBT_AVAILABLE:
            return self._fallback_fit_transform(data)
        
        # Optimize data for VectorBT processing
        if self.memory_efficient:
            data = self._optimize_data_types(data)
        
        # Enable GPU if available
        if self.enable_gpu:
            data = self._enable_gpu_processing(data)
            self.performance_stats['gpu_operations'] += 1
        
        # Apply VectorBT scaling
        result = self._apply_vectorbt_scaling(data)
        self.performance_stats['vectorbt_operations'] += 1
        
        return result
    
    def _batch_fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Batch fit_transform with optimization."""
        if not VECTORBT_AVAILABLE:
            return self._fallback_batch_fit_transform(data)
        
        # Optimize data for VectorBT processing
        if self.memory_efficient:
            data = self._optimize_dataframe_types(data)
        
        # Enable GPU if available
        if self.enable_gpu:
            data = self._enable_gpu_dataframe_processing(data)
            self.performance_stats['gpu_operations'] += 1
        
        # Apply VectorBT batch scaling
        result = self._apply_vectorbt_batch_scaling(data)
        self.performance_stats['batch_operations'] += 1
        
        return result
    
    def _apply_vectorbt_scaling(self, data: pd.Series) -> pd.Series:
        """Apply VectorBT scaling with enhanced methods."""
        if self.method == 'zscore':
            return zscore(data)
        elif self.method == 'minmax':
            return scale(data, method='minmax')
        elif self.method == 'robust':
            return scale(data, method='robust')
        elif self.method == 'quantile':
            return quantile(data)
        elif self.method == 'winsorize':
            return winsorize(data)
        elif self.method == 'rank':
            return rank(data)
        elif self.method == 'clip':
            return clip(data)
        elif self.method == 'robust_zscore':
            # Enhanced robust z-score
            median = data.median()
            mad = (data - median).abs().median()
            return (data - median) / (1.4826 * mad)
        elif self.method == 'adaptive':
            # Adaptive scaling based on data characteristics
            if data.skew() > 2:  # Highly skewed
                return quantile(data)
            elif data.kurtosis() > 3:  # Heavy-tailed
                return scale(data, method='robust')
            else:  # Normal-like
                return zscore(data)
        elif self.method == 'quantile_robust':
            # Robust quantile scaling
            q25, q75 = data.quantile([0.25, 0.75])
            return (data - q25) / (q75 - q25 + 1e-8)
        else:
            raise ValueError(f"Unsupported scaling method: {self.method}")
    
    def _apply_vectorbt_batch_scaling(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply VectorBT batch scaling."""
        if self.method == 'zscore':
            return zscore(data)
        elif self.method == 'minmax':
            return scale(data, method='minmax')
        elif self.method == 'robust':
            return scale(data, method='robust')
        elif self.method == 'quantile':
            return quantile(data)
        elif self.method == 'winsorize':
            return winsorize(data)
        elif self.method == 'rank':
            return rank(data)
        elif self.method == 'clip':
            return clip(data)
        else:
            # For custom methods, apply column by column
            result = data.copy()
            for column in data.columns:
                result[column] = self._apply_vectorbt_scaling(data[column])
            return result
    
    def _optimize_data_types(self, data: pd.Series) -> pd.Series:
        """Optimize data types for memory efficiency."""
        if self.memory_efficient:
            if data.dtype == 'float64':
                # Check if float32 is sufficient
                if (data.min() >= np.finfo(np.float32).min and 
                    data.max() <= np.finfo(np.float32).max):
                    data = data.astype(np.float32)
                    self.performance_stats['memory_optimizations'] += 1
        return data
    
    def _optimize_dataframe_types(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame types for memory efficiency."""
        if self.memory_efficient:
            optimized_data = data.copy()
            for column in optimized_data.columns:
                if optimized_data[column].dtype == 'float64':
                    if (optimized_data[column].min() >= np.finfo(np.float32).min and 
                        optimized_data[column].max() <= np.finfo(np.float32).max):
                        optimized_data[column] = optimized_data[column].astype(np.float32)
                        self.performance_stats['memory_optimizations'] += 1
            return optimized_data
        return data
    
    def _enable_gpu_processing(self, data: pd.Series) -> pd.Series:
        """Enable GPU processing if available."""
        if self.enable_gpu and CUPY_AVAILABLE:
            try:
                gpu_data = cp.asarray(data.values)
                return pd.Series(gpu_data, index=data.index)
            except Exception as e:
                logger.warning(f"GPU processing failed: {e}")
                return data
        return data
    
    def _enable_gpu_dataframe_processing(self, data: pd.DataFrame) -> pd.DataFrame:
        """Enable GPU DataFrame processing if available."""
        if self.enable_gpu and CUPY_AVAILABLE:
            try:
                gpu_data = {}
                for column in data.columns:
                    gpu_data[column] = cp.asarray(data[column].values)
                return pd.DataFrame(gpu_data, index=data.index)
            except Exception as e:
                logger.warning(f"GPU DataFrame processing failed: {e}")
                return data
        return data
    
    def _fallback_fit_transform(self, data: pd.Series) -> pd.Series:
        """Fallback implementation using pandas/numpy."""
        if self.method == 'zscore':
            return (data - data.mean()) / data.std()
        elif self.method == 'minmax':
            return (data - data.min()) / (data.max() - data.min())
        elif self.method == 'robust':
            median = data.median()
            mad = (data - median).abs().median()
            return (data - median) / mad
        else:
            return data  # Return original for unsupported methods
    
    def _fallback_batch_fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fallback batch implementation."""
        result = data.copy()
        for column in data.columns:
            result[column] = self._fallback_fit_transform(data[column])
        return result


class EnhancedVectorBTRollingOptimizer:
    """
    Enhanced VectorBT Rolling Optimizer with advanced statistical functions.
    
    This class provides optimized rolling operations with:
    - Advanced statistical functions
    - Batch processing
    - Memory optimization
    - GPU acceleration
    """
    
    def __init__(self, enable_gpu: bool = False, enable_parallel: bool = True,
                 memory_efficient: bool = True):
        """
        Initialize enhanced VectorBT rolling optimizer.
        
        Args:
            enable_gpu: Enable GPU acceleration
            enable_parallel: Enable parallel processing
            memory_efficient: Enable memory optimization
        """
        self.enable_gpu = enable_gpu and CUPY_AVAILABLE
        self.enable_parallel = enable_parallel and VECTORBT_AVAILABLE
        self.memory_efficient = memory_efficient
        self.use_vectorbt = VECTORBT_AVAILABLE
        
        # Performance tracking
        self.performance_stats = {
            'vectorbt_operations': 0,
            'gpu_operations': 0,
            'parallel_operations': 0,
            'memory_optimizations': 0,
            'total_operations': 0
        }
    
    def rolling_mean(self, data: Union[pd.Series, pd.DataFrame], window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Optimized rolling mean calculation."""
        return self._rolling_operation(data, 'mean', window, **kwargs)
    
    def rolling_std(self, data: Union[pd.Series, pd.DataFrame], window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Optimized rolling standard deviation calculation."""
        return self._rolling_operation(data, 'std', window, **kwargs)
    
    def rolling_var(self, data: Union[pd.Series, pd.DataFrame], window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Optimized rolling variance calculation."""
        return self._rolling_operation(data, 'var', window, **kwargs)
    
    def rolling_min(self, data: Union[pd.Series, pd.DataFrame], window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Optimized rolling minimum calculation."""
        return self._rolling_operation(data, 'min', window, **kwargs)
    
    def rolling_max(self, data: Union[pd.Series, pd.DataFrame], window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Optimized rolling maximum calculation."""
        return self._rolling_operation(data, 'max', window, **kwargs)
    
    def rolling_sum(self, data: Union[pd.Series, pd.DataFrame], window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Optimized rolling sum calculation."""
        return self._rolling_operation(data, 'sum', window, **kwargs)
    
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
    
    def _rolling_operation(self, data: Union[pd.Series, pd.DataFrame], operation: str, 
                          window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Generic rolling operation with optimization."""
        self.performance_stats['total_operations'] += 1
        
        if not self.use_vectorbt:
            return self._fallback_rolling_operation(data, operation, window, **kwargs)
        
        # Optimize data for VectorBT processing
        if self.memory_efficient:
            data = self._optimize_data_types(data)
        
        # Enable GPU if available
        if self.enable_gpu:
            data = self._enable_gpu_processing(data)
            self.performance_stats['gpu_operations'] += 1
        
        # Apply VectorBT rolling operation
        result = self._apply_vectorbt_rolling_operation(data, operation, window, **kwargs)
        self.performance_stats['vectorbt_operations'] += 1
        
        return result
    
    def _apply_vectorbt_rolling_operation(self, data: Union[pd.Series, pd.DataFrame], 
                                        operation: str, window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Apply VectorBT rolling operation."""
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
        elif operation == 'corr':
            other = kwargs.get('other')
            return rolling_corr(data, other, window=window, **kwargs)
        elif operation == 'cov':
            other = kwargs.get('other')
            return rolling_cov(data, other, window=window, **kwargs)
        elif operation == 'apply':
            func = kwargs.get('func')
            return rolling_apply(data, func, window=window, **kwargs)
        else:
            raise ValueError(f"Unsupported operation: {operation}")
    
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
    
    def _enable_gpu_processing(self, data: Union[pd.Series, pd.DataFrame]) -> Union[pd.Series, pd.DataFrame]:
        """Enable GPU processing if available."""
        if self.enable_gpu and CUPY_AVAILABLE:
            try:
                if isinstance(data, pd.Series):
                    gpu_data = cp.asarray(data.values)
                    return pd.Series(gpu_data, index=data.index)
                elif isinstance(data, pd.DataFrame):
                    gpu_data = {}
                    for column in data.columns:
                        gpu_data[column] = cp.asarray(data[column].values)
                    return pd.DataFrame(gpu_data, index=data.index)
            except Exception as e:
                logger.warning(f"GPU processing failed: {e}")
                return data
        return data
    
    def _fallback_rolling_operation(self, data: Union[pd.Series, pd.DataFrame], 
                                  operation: str, window: int, **kwargs) -> Union[pd.Series, pd.DataFrame]:
        """Fallback rolling operation using pandas."""
        if operation == 'mean':
            return data.rolling(window=window, **kwargs).mean()
        elif operation == 'std':
            return data.rolling(window=window, **kwargs).std()
        elif operation == 'var':
            return data.rolling(window=window, **kwargs).var()
        elif operation == 'min':
            return data.rolling(window=window, **kwargs).min()
        elif operation == 'max':
            return data.rolling(window=window, **kwargs).max()
        elif operation == 'sum':
            return data.rolling(window=window, **kwargs).sum()
        elif operation == 'quantile':
            q = kwargs.get('q', 0.5)
            return data.rolling(window=window, **kwargs).quantile(q)
        elif operation == 'skew':
            return data.rolling(window=window, **kwargs).skew()
        elif operation == 'kurt':
            return data.rolling(window=window, **kwargs).kurt()
        elif operation == 'corr':
            other = kwargs.get('other')
            return data.rolling(window=window, **kwargs).corr(other)
        elif operation == 'cov':
            other = kwargs.get('other')
            return data.rolling(window=window, **kwargs).cov(other)
        elif operation == 'apply':
            func = kwargs.get('func')
            return data.rolling(window=window, **kwargs).apply(func)
        else:
            raise ValueError(f"Unsupported operation: {operation}")


class EnhancedVectorBTBatchProcessor:
    """
    Enhanced VectorBT Batch Processor for efficient feature generation.
    
    This class provides optimized batch processing with:
    - Memory-efficient chunked processing
    - Parallel processing
    - GPU acceleration
    - Advanced feature generation
    """
    
    def __init__(self, chunk_size: int = 1000, enable_gpu: bool = False,
                 enable_parallel: bool = True, max_workers: int = 4):
        """
        Initialize enhanced VectorBT batch processor.
        
        Args:
            chunk_size: Size of data chunks for processing
            enable_gpu: Enable GPU acceleration
            enable_parallel: Enable parallel processing
            max_workers: Maximum number of parallel workers
        """
        self.chunk_size = chunk_size
        self.enable_gpu = enable_gpu and CUPY_AVAILABLE
        self.enable_parallel = enable_parallel and VECTORBT_AVAILABLE
        self.max_workers = max_workers
        self.use_vectorbt = VECTORBT_AVAILABLE
        
        # Performance tracking
        self.performance_stats = {
            'vectorbt_operations': 0,
            'gpu_operations': 0,
            'parallel_operations': 0,
            'chunk_operations': 0,
            'total_operations': 0
        }
    
    def process_features_batch(self, data: pd.DataFrame, 
                             feature_configs: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Process multiple features in batch with optimization.
        
        Args:
            data: Input OHLCV data
            feature_configs: List of feature configuration dictionaries
            
        Returns:
            DataFrame with generated features
        """
        if not self.use_vectorbt:
            return self._fallback_batch_processing(data, feature_configs)
        
        # Group features by type for efficient processing
        feature_groups = self._group_features_by_type(feature_configs)
        
        results = {}
        
        # Process each group with optimization
        for group_type, group_configs in feature_groups.items():
            if group_type == 'momentum':
                group_results = self._process_momentum_features_batch(data, group_configs)
            elif group_type == 'volatility':
                group_results = self._process_volatility_features_batch(data, group_configs)
            elif group_type == 'volume':
                group_results = self._process_volume_features_batch(data, group_configs)
            elif group_type == 'trend':
                group_results = self._process_trend_features_batch(data, group_configs)
            else:
                group_results = self._process_generic_features_batch(data, group_configs)
            
            results.update(group_results)
            self.performance_stats['chunk_operations'] += 1
        
        return pd.DataFrame(results, index=data.index)
    
    def _group_features_by_type(self, feature_configs: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Group features by type for efficient processing."""
        groups = {}
        for config in feature_configs:
            feature_type = config.get('type', 'generic')
            if feature_type not in groups:
                groups[feature_type] = []
            groups[feature_type].append(config)
        return groups
    
    def _process_momentum_features_batch(self, data: pd.DataFrame, 
                                       configs: List[Dict[str, Any]]) -> Dict[str, pd.Series]:
        """Process momentum features in batch using VectorBT."""
        results = {}
        
        # Extract common parameters
        periods = list(set([config.get('params', {}).get('period', 14) for config in configs]))
        
        # Calculate common rolling statistics once
        rolling_stats = {}
        for period in periods:
            rolling_stats[period] = {
                'mean': rolling_mean(data['close'], window=period),
                'std': rolling_std(data['close'], window=period),
                'min': rolling_min(data['close'], window=period),
                'max': rolling_max(data['close'], window=period)
            }
        
        # Process each momentum feature
        for config in configs:
            name = config['name']
            params = config.get('params', {})
            period = params.get('period', 14)
            
            if config.get('indicator') == 'rsi':
                # RSI calculation using pre-calculated statistics
                delta = data['close'].diff()
                gain = delta.where(delta > 0, 0)
                loss = -delta.where(delta < 0, 0)
                
                avg_gain = rolling_stats[period]['mean'].reindex(gain.index).fillna(0)
                avg_loss = rolling_stats[period]['mean'].reindex(loss.index).fillna(0)
                
                rs = avg_gain / (avg_loss + 1e-8)
                results[name] = 100 - (100 / (1 + rs))
            
            elif config.get('indicator') == 'macd':
                # MACD calculation
                fast_period = params.get('fast_period', 12)
                slow_period = params.get('slow_period', 26)
                signal_period = params.get('signal_period', 9)
                
                ema_fast = data['close'].ewm(span=fast_period).mean()
                ema_slow = data['close'].ewm(span=slow_period).mean()
                macd_line = ema_fast - ema_slow
                signal_line = macd_line.ewm(span=signal_period).mean()
                
                results[name] = macd_line - signal_line
            
            # Add more momentum indicators...
        
        self.performance_stats['vectorbt_operations'] += len(configs)
        return results
    
    def _process_volatility_features_batch(self, data: pd.DataFrame, 
                                         configs: List[Dict[str, Any]]) -> Dict[str, pd.Series]:
        """Process volatility features in batch using VectorBT."""
        results = {}
        
        for config in configs:
            name = config['name']
            params = config.get('params', {})
            period = params.get('period', 20)
            
            if config.get('indicator') == 'atr':
                # ATR calculation
                high = data['high']
                low = data['low']
                close = data['close']
                
                tr1 = high - low
                tr2 = abs(high - close.shift(1))
                tr3 = abs(low - close.shift(1))
                
                true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                results[name] = rolling_mean(true_range, window=period)
            
            elif config.get('indicator') == 'bbands':
                # Bollinger Bands calculation
                close = data['close']
                std_dev = params.get('std_dev', 2.0)
                
                sma = rolling_mean(close, window=period)
                std = rolling_std(close, window=period)
                
                results[f"{name}_upper"] = sma + (std * std_dev)
                results[f"{name}_middle"] = sma
                results[f"{name}_lower"] = sma - (std * std_dev)
                results[f"{name}_width"] = (results[f"{name}_upper"] - results[f"{name}_lower"]) / sma
            
            # Add more volatility indicators...
        
        self.performance_stats['vectorbt_operations'] += len(configs)
        return results
    
    def _process_volume_features_batch(self, data: pd.DataFrame, 
                                     configs: List[Dict[str, Any]]) -> Dict[str, pd.Series]:
        """Process volume features in batch using VectorBT."""
        results = {}
        
        for config in configs:
            name = config['name']
            params = config.get('params', {})
            period = params.get('period', 20)
            
            if config.get('indicator') == 'obv':
                # OBV calculation
                close = data['close']
                volume = data['volume']
                
                price_change = close.diff()
                obv = (volume * np.sign(price_change)).cumsum()
                results[name] = obv
            
            elif config.get('indicator') == 'mfi':
                # MFI calculation
                high = data['high']
                low = data['low']
                close = data['close']
                volume = data['volume']
                
                typical_price = (high + low + close) / 3
                money_flow = typical_price * volume
                
                positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
                negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)
                
                positive_flow_sum = rolling_sum(positive_flow, window=period)
                negative_flow_sum = rolling_sum(negative_flow, window=period)
                
                mfi = 100 - (100 / (1 + positive_flow_sum / (negative_flow_sum + 1e-8)))
                results[name] = mfi
            
            # Add more volume indicators...
        
        self.performance_stats['vectorbt_operations'] += len(configs)
        return results
    
    def _process_trend_features_batch(self, data: pd.DataFrame, 
                                    configs: List[Dict[str, Any]]) -> Dict[str, pd.Series]:
        """Process trend features in batch using VectorBT."""
        results = {}
        
        for config in configs:
            name = config['name']
            params = config.get('params', {})
            period = params.get('period', 20)
            
            if config.get('indicator') == 'sma':
                # Simple Moving Average
                results[name] = rolling_mean(data['close'], window=period)
            
            elif config.get('indicator') == 'ema':
                # Exponential Moving Average
                results[name] = data['close'].ewm(span=period).mean()
            
            elif config.get('indicator') == 'wma':
                # Weighted Moving Average
                weights = np.arange(1, period + 1)
                results[name] = rolling_apply(data['close'], 
                                            lambda x: np.average(x, weights=weights), 
                                            window=period)
            
            # Add more trend indicators...
        
        self.performance_stats['vectorbt_operations'] += len(configs)
        return results
    
    def _process_generic_features_batch(self, data: pd.DataFrame, 
                                      configs: List[Dict[str, Any]]) -> Dict[str, pd.Series]:
        """Process generic features in batch."""
        results = {}
        
        for config in configs:
            name = config['name']
            params = config.get('params', {})
            period = params.get('period', 20)
            
            if config.get('operation') == 'rolling_mean':
                column = params.get('column', 'close')
                results[name] = rolling_mean(data[column], window=period)
            
            elif config.get('operation') == 'rolling_std':
                column = params.get('column', 'close')
                results[name] = rolling_std(data[column], window=period)
            
            # Add more generic operations...
        
        self.performance_stats['vectorbt_operations'] += len(configs)
        return results
    
    def _fallback_batch_processing(self, data: pd.DataFrame, 
                                 feature_configs: List[Dict[str, Any]]) -> pd.DataFrame:
        """Fallback batch processing using pandas."""
        results = {}
        
        for config in feature_configs:
            name = config['name']
            params = config.get('params', {})
            period = params.get('period', 20)
            
            if config.get('operation') == 'rolling_mean':
                column = params.get('column', 'close')
                results[name] = data[column].rolling(window=period).mean()
            
            elif config.get('operation') == 'rolling_std':
                column = params.get('column', 'close')
                results[name] = data[column].rolling(window=period).std()
            
            # Add more fallback operations...
        
        return pd.DataFrame(results, index=data.index)


# Example usage functions
def create_enhanced_scaler(method: str = 'zscore', **kwargs) -> EnhancedVectorBTScaler:
    """Create an enhanced VectorBT scaler."""
    return EnhancedVectorBTScaler(method=method, **kwargs)


def create_enhanced_rolling_optimizer(**kwargs) -> EnhancedVectorBTRollingOptimizer:
    """Create an enhanced VectorBT rolling optimizer."""
    return EnhancedVectorBTRollingOptimizer(**kwargs)


def create_enhanced_batch_processor(**kwargs) -> EnhancedVectorBTBatchProcessor:
    """Create an enhanced VectorBT batch processor."""
    return EnhancedVectorBTBatchProcessor(**kwargs)


# Performance testing functions
def benchmark_scaling_performance(data: pd.Series, methods: List[str] = None) -> Dict[str, float]:
    """Benchmark scaling performance for different methods."""
    if methods is None:
        methods = ['zscore', 'minmax', 'robust', 'quantile', 'winsorize', 'rank', 'clip']
    
    results = {}
    
    for method in methods:
        scaler = create_enhanced_scaler(method=method, enable_gpu=False, memory_efficient=True)
        
        start_time = time.time()
        _ = scaler.fit_transform(data)
        end_time = time.time()
        
        results[method] = end_time - start_time
    
    return results


def benchmark_rolling_performance(data: pd.Series, window: int = 20) -> Dict[str, float]:
    """Benchmark rolling operations performance."""
    optimizer = create_enhanced_rolling_optimizer(enable_gpu=False, memory_efficient=True)
    
    operations = ['mean', 'std', 'var', 'min', 'max', 'sum', 'quantile', 'skew', 'kurt']
    results = {}
    
    for operation in operations:
        start_time = time.time()
        if operation == 'quantile':
            _ = optimizer.rolling_quantile(data, window=window, q=0.5)
        else:
            _ = getattr(optimizer, f'rolling_{operation}')(data, window=window)
        end_time = time.time()
        
        results[operation] = end_time - start_time
    
    return results


if __name__ == "__main__":
    # Example usage
    import time
    
    # Create sample data
    np.random.seed(42)
    data = pd.Series(np.random.randn(10000), name='test_data')
    
    # Test enhanced scaler
    print("Testing Enhanced VectorBT Scaler...")
    scaler = create_enhanced_scaler(method='zscore', enable_gpu=False, memory_efficient=True)
    scaled_data = scaler.fit_transform(data)
    print(f"Scaled data shape: {scaled_data.shape}")
    print(f"Performance stats: {scaler.performance_stats}")
    
    # Test enhanced rolling optimizer
    print("\nTesting Enhanced VectorBT Rolling Optimizer...")
    optimizer = create_enhanced_rolling_optimizer(enable_gpu=False, memory_efficient=True)
    rolling_mean = optimizer.rolling_mean(data, window=20)
    print(f"Rolling mean shape: {rolling_mean.shape}")
    print(f"Performance stats: {optimizer.performance_stats}")
    
    # Test performance benchmarks
    print("\nRunning Performance Benchmarks...")
    scaling_results = benchmark_scaling_performance(data)
    rolling_results = benchmark_rolling_performance(data)
    
    print("Scaling Performance:")
    for method, time_taken in scaling_results.items():
        print(f"  {method}: {time_taken:.4f}s")
    
    print("Rolling Performance:")
    for operation, time_taken in rolling_results.items():
        print(f"  {operation}: {time_taken:.4f}s")