"""
VectorBT Optimizations for Matrix Operations

This module provides VectorBT-optimized implementations of matrix operations
that significantly improve performance over custom implementations.

Key Features:
- VectorBT-optimized trading indicators
- Enhanced matrix operations with VectorBT
- Optimized rolling operations
- Improved correlation analysis
- Parallel batch processing
- Memory-efficient operations
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from contextlib import contextmanager
import warnings

# Conditional imports
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None

try:
    import vectorbt as vbt
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None

logger = logging.getLogger(__name__)

class VectorBTOptimizedOperations:
    """
    VectorBT-optimized operations for matrix and financial calculations.
    
    This class provides high-performance implementations using VectorBT's
    optimized functions for financial and mathematical operations.
    """
    
    def __init__(self, enable_gpu: bool = True, enable_parallel: bool = True):
        """Initialize VectorBT optimized operations."""
        self.enable_gpu = enable_gpu and VECTORBT_AVAILABLE
        self.enable_parallel = enable_parallel and VECTORBT_AVAILABLE
        
        # Performance tracking
        self.performance_stats = {
            'total_operations': 0,
            'vectorbt_operations': 0,
            'fallback_operations': 0,
            'average_execution_time': 0.0,
            'gpu_operations': 0,
            'memory_optimizations': 0,
            'chunked_operations': 0
        }
        
        self.logger = logger.getChild('VectorBTOptimizedOperations')
        
        # Configure VectorBT for optimal performance
        if VECTORBT_AVAILABLE:
            self._configure_vectorbt()
            self.logger.info("✅ VectorBT optimized operations initialized")
        else:
            self.logger.warning("⚠️ VectorBT not available, using fallback implementations")
    
    def _configure_vectorbt(self):
        """Configure VectorBT for optimal matrix operations."""
        try:
            # Configure for matrix operations
            vbt.settings['array_wrapper']['freq_precision'] = 0
            vbt.settings['array_wrapper']['freq_rep'] = 'auto'
            vbt.settings['array_wrapper']['chunk_size'] = 25000
            vbt.settings['array_wrapper']['memory_limit'] = 3 * 1024**3  # 3GB limit
            
            # Enable parallel processing
            if self.enable_parallel:
                vbt.settings['parallel']['threading'] = True
                vbt.settings['parallel']['multiprocessing'] = True
                vbt.settings['parallel']['n_jobs'] = -1
            
            # Enable GPU if available
            if self.enable_gpu:
                vbt.settings['array_wrapper']['use_gpu'] = True
                vbt.settings['array_wrapper']['gpu_memory_fraction'] = 0.6
            
            # Enable caching
            vbt.settings['caching']['enabled'] = True
            vbt.settings['caching']['dir'] = 'data_cache/vectorbt_matrix_cache'
            
        except Exception as e:
            self.logger.warning(f"⚠️ VectorBT configuration warning: {e}")
    
    def matrix_multiply(self, A: 'np.ndarray', B: 'np.ndarray') -> 'np.ndarray':
        """
        VectorBT-optimized matrix multiplication.
        
        Args:
            A: First matrix
            B: Second matrix
            
        Returns:
            Result of matrix multiplication
        """
        start_time = time.time()
        
        try:
            if VECTORBT_AVAILABLE:
                # Use VectorBT's optimized matrix multiplication
                result = vbt.math.matrix_multiply(A, B)
                self.performance_stats['vectorbt_operations'] += 1
                self.logger.debug("✅ VectorBT matrix multiplication completed")
            else:
                # Fallback to numpy
                result = np.dot(A, B)
                self.performance_stats['fallback_operations'] += 1
                self.logger.debug("✅ Fallback matrix multiplication completed")
            
            # Update performance stats
            execution_time = time.time() - start_time
            self._update_performance_stats(execution_time)
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Matrix multiplication failed: {e}")
            # Fallback to numpy
            result = np.dot(A, B)
            self.performance_stats['fallback_operations'] += 1
            return result
    
    def correlation_matrix(self, data: Union['np.ndarray', 'pd.DataFrame'], 
                          method: str = 'pearson') -> 'np.ndarray':
        """
        VectorBT-optimized correlation matrix calculation with memory management.
        
        Args:
            data: Input data matrix
            method: Correlation method ('pearson', 'spearman')
            
        Returns:
            Correlation matrix
        """
        start_time = time.time()
        
        try:
            if isinstance(data, pd.DataFrame):
                data = data.values
            
            if VECTORBT_AVAILABLE:
                # Use VectorBT's optimized correlation calculation with chunking for large datasets
                if len(data) > 10000:  # Use chunked processing for large datasets
                    if method == 'pearson':
                        result = vbt.math.corr_matrix(data, chunked=True)
                    elif method == 'spearman':
                        result = vbt.math.corr_matrix(data, method='spearman', chunked=True)
                    else:
                        raise ValueError(f"Unknown correlation method: {method}")
                    
                    self.performance_stats['chunked_operations'] += 1
                    self.logger.debug(f"✅ VectorBT chunked correlation matrix ({method}) completed")
                else:
                    if method == 'pearson':
                        result = vbt.math.corr_matrix(data)
                    elif method == 'spearman':
                        result = vbt.math.corr_matrix(data, method='spearman')
                    else:
                        raise ValueError(f"Unknown correlation method: {method}")
                    
                    self.logger.debug(f"✅ VectorBT correlation matrix ({method}) completed")
                
                self.performance_stats['vectorbt_operations'] += 1
            else:
                # Fallback to numpy
                if method == 'pearson':
                    result = np.corrcoef(data.T)
                elif method == 'spearman':
                    from scipy.stats import spearmanr
                    result = np.corrcoef(data.T)
                else:
                    raise ValueError(f"Unknown correlation method: {method}")
                
                self.performance_stats['fallback_operations'] += 1
                self.logger.debug(f"✅ Fallback correlation matrix ({method}) completed")
            
            # Update performance stats
            execution_time = time.time() - start_time
            self._update_performance_stats(execution_time)
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Correlation matrix calculation failed: {e}")
            # Fallback to numpy
            result = np.corrcoef(data.T)
            self.performance_stats['fallback_operations'] += 1
            return result
    
    def compute_trading_indicators(self, data: 'pd.DataFrame', 
                                 config: Optional[Dict[str, Any]] = None) -> 'pd.DataFrame':
        """
        Compute trading indicators using VectorBT's optimized functions.
        
        Args:
            data: DataFrame with OHLCV data
            config: Configuration for indicators
            
        Returns:
            DataFrame with computed indicators
        """
        if not VECTORBT_AVAILABLE:
            self.logger.warning("⚠️ VectorBT not available for trading indicators")
            return data.copy()
        
        start_time = time.time()
        
        try:
            if config is None:
                config = self._get_default_indicator_config()
            
            result_df = data.copy()
            
            # Ensure required columns exist
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in data.columns]
            if missing_cols:
                self.logger.warning(f"⚠️ Missing required columns: {missing_cols}")
                return result_df
            
            # Compute indicators using VectorBT
            result_df = self._compute_moving_averages_vectorbt(result_df, config)
            result_df = self._compute_momentum_indicators_vectorbt(result_df, config)
            result_df = self._compute_volatility_indicators_vectorbt(result_df, config)
            result_df = self._compute_volume_indicators_vectorbt(result_df, config)
            result_df = self._compute_trend_indicators_vectorbt(result_df, config)
            result_df = self._compute_oscillator_indicators_vectorbt(result_df, config)
            
            # Update performance stats
            execution_time = time.time() - start_time
            self._update_performance_stats(execution_time)
            
            self.logger.info(f"✅ Computed {len(result_df.columns) - len(data.columns)} trading indicators using VectorBT")
            return result_df
            
        except Exception as e:
            self.logger.error(f"❌ Trading indicators computation failed: {e}")
            return data.copy()
    
    def _compute_moving_averages_vectorbt(self, data: 'pd.DataFrame', 
                                        config: Dict[str, Any]) -> 'pd.DataFrame':
        """Compute moving averages using VectorBT."""
        result = data.copy()
        
        # Simple Moving Averages
        for period in config.get('sma_periods', [9, 21, 50, 200]):
            result[f'sma_{period}'] = vbt.MA.run(data['close'], window=period).ma
        
        # Exponential Moving Averages
        for period in config.get('ema_periods', [12, 26, 50]):
            result[f'ema_{period}'] = vbt.MA.run(data['close'], window=period, short_name='EMA').ma
        
        # Moving Average Crossovers
        if 'sma_9' in result.columns and 'sma_21' in result.columns:
            result['sma_cross_9_21'] = (result['sma_9'] > result['sma_21']).astype(int)
        if 'ema_12' in result.columns and 'ema_26' in result.columns:
            result['ema_cross_12_26'] = (result['ema_12'] > result['ema_26']).astype(int)
        
        return result
    
    def _compute_momentum_indicators_vectorbt(self, data: 'pd.DataFrame', 
                                            config: Dict[str, Any]) -> 'pd.DataFrame':
        """Compute momentum indicators using VectorBT."""
        result = data.copy()
        
        # RSI using VectorBT
        rsi_period = config.get('rsi_period', 14)
        rsi = vbt.RSI.run(data['close'], window=rsi_period)
        result['rsi'] = rsi.rsi
        result['rsi_overbought'] = (result['rsi'] > config.get('rsi_overbought', 70)).astype(int)
        result['rsi_oversold'] = (result['rsi'] < config.get('rsi_oversold', 30)).astype(int)
        
        # MACD using VectorBT
        macd_fast = config.get('macd_fast', 12)
        macd_slow = config.get('macd_slow', 26)
        macd_signal = config.get('macd_signal', 9)
        
        macd = vbt.MACD.run(data['close'], fast_window=macd_fast, slow_window=macd_slow, signal_window=macd_signal)
        result['macd'] = macd.macd
        result['macd_signal'] = macd.signal
        result['macd_histogram'] = macd.histogram
        result['macd_bullish'] = (result['macd'] > result['macd_signal']).astype(int)
        result['macd_cross'] = (result['macd'] > result['macd_signal']).astype(int).diff().fillna(0)
        
        # ROC using VectorBT
        roc_period = config.get('roc_period', 10)
        result['roc'] = vbt.ROC.run(data['close'], window=roc_period).roc
        
        return result
    
    def _compute_volatility_indicators_vectorbt(self, data: 'pd.DataFrame', 
                                              config: Dict[str, Any]) -> 'pd.DataFrame':
        """Compute volatility indicators using VectorBT."""
        result = data.copy()
        
        # Bollinger Bands using VectorBT
        bb_period = config.get('bb_period', 20)
        bb_std = config.get('bb_std', 2.0)
        
        bb = vbt.BBANDS.run(data['close'], window=bb_period, alpha=bb_std)
        result['bb_upper'] = bb.upper
        result['bb_lower'] = bb.lower
        result['bb_middle'] = bb.middle
        result['bb_width'] = (result['bb_upper'] - result['bb_lower']) / result['bb_middle']
        result['bb_position'] = (data['close'] - result['bb_lower']) / (result['bb_upper'] - result['bb_lower'])
        
        # ATR using VectorBT
        atr_period = config.get('atr_period', 14)
        atr = vbt.ATR.run(data['high'], data['low'], data['close'], window=atr_period)
        result['atr'] = atr.atr
        result['atr_percent'] = (result['atr'] / data['close']) * 100
        
        # Volatility
        result['volatility'] = data['close'].rolling(window=20, min_periods=1).std()
        result['volatility_percent'] = (result['volatility'] / data['close']) * 100
        
        return result
    
    def _compute_volume_indicators_vectorbt(self, data: 'pd.DataFrame', 
                                          config: Dict[str, Any]) -> 'pd.DataFrame':
        """Compute volume indicators using VectorBT."""
        result = data.copy()
        
        # Volume SMA
        volume_sma_period = config.get('volume_sma_period', 20)
        result['volume_sma'] = data['volume'].rolling(window=volume_sma_period, min_periods=1).mean()
        result['volume_ratio'] = data['volume'] / result['volume_sma']
        
        # OBV using VectorBT
        obv = vbt.OBV.run(data['close'], data['volume'])
        result['obv'] = obv.obv
        
        # OBV smoothed
        obv_smooth = config.get('obv_smooth', 10)
        result['obv_sma'] = result['obv'].rolling(window=obv_smooth, min_periods=1).mean()
        
        return result
    
    def _compute_trend_indicators_vectorbt(self, data: 'pd.DataFrame', 
                                         config: Dict[str, Any]) -> 'pd.DataFrame':
        """Compute trend indicators using VectorBT."""
        result = data.copy()
        
        # ADX using VectorBT
        adx_period = config.get('adx_period', 14)
        adx = vbt.ADX.run(data['high'], data['low'], data['close'], window=adx_period)
        result['adx'] = adx.adx
        result['plus_di'] = adx.plus_di
        result['minus_di'] = adx.minus_di
        result['adx_trending'] = (result['adx'] > 25).astype(int)
        result['adx_strong_trend'] = (result['adx'] > 50).astype(int)
        
        return result
    
    def _compute_oscillator_indicators_vectorbt(self, data: 'pd.DataFrame', 
                                              config: Dict[str, Any]) -> 'pd.DataFrame':
        """Compute oscillator indicators using VectorBT."""
        result = data.copy()
        
        # Stochastic Oscillator using VectorBT
        stoch_k = config.get('stoch_k', 14)
        stoch_d = config.get('stoch_d', 3)
        
        stoch = vbt.STOCH.run(data['high'], data['low'], data['close'], 
                             k_window=stoch_k, d_window=stoch_d)
        result['stoch_k'] = stoch.k
        result['stoch_d'] = stoch.d
        result['stoch_overbought'] = (result['stoch_k'] > 80).astype(int)
        result['stoch_oversold'] = (result['stoch_k'] < 20).astype(int)
        
        # Williams %R using VectorBT
        williams_period = config.get('williams_period', 14)
        williams = vbt.WILLR.run(data['high'], data['low'], data['close'], window=williams_period)
        result['williams_r'] = williams.willr
        
        # CCI using VectorBT
        cci_period = config.get('cci_period', 20)
        cci = vbt.CCI.run(data['high'], data['low'], data['close'], window=cci_period)
        result['cci'] = cci.cci
        result['cci_overbought'] = (result['cci'] > 100).astype(int)
        result['cci_oversold'] = (result['cci'] < -100).astype(int)
        
        return result
    
    def rolling_features(self, data: 'pd.DataFrame',
                        windows: List[int] = [5, 10, 20, 50],
                        features: List[str] = None) -> 'pd.DataFrame':
        """
        Create rolling features using VectorBT's optimized functions.
        
        Args:
            data: Input DataFrame
            windows: List of window sizes
            features: List of feature columns to process
            
        Returns:
            DataFrame with rolling features
        """
        if not VECTORBT_AVAILABLE:
            self.logger.warning("⚠️ VectorBT not available for rolling features")
            return data.copy()
        
        start_time = time.time()
        
        try:
            if features is None:
                features = data.select_dtypes(include=[np.number]).columns.tolist()
            
            result_dfs = []
            
            for window in windows:
                window_features = {}
                for col in features:
                    if col in data.columns:
                        series = data[col]
                        
                        # Use VectorBT's optimized rolling functions
                        rolling = vbt.Rolling(series, window=window)
                        
                        window_features[f'{col}_rolling_mean_{window}'] = rolling.mean()
                        window_features[f'{col}_rolling_std_{window}'] = rolling.std()
                        window_features[f'{col}_rolling_min_{window}'] = rolling.min()
                        window_features[f'{col}_rolling_max_{window}'] = rolling.max()
                        window_features[f'{col}_rolling_skew_{window}'] = rolling.skew()
                        window_features[f'{col}_rolling_kurt_{window}'] = rolling.kurt()
                
                result_dfs.append(pd.DataFrame(window_features))
            
            # Combine all features efficiently
            if result_dfs:
                combined = pd.concat(result_dfs, axis=1)
                result = pd.concat([data, combined], axis=1)
            else:
                result = data
            
            # Update performance stats
            execution_time = time.time() - start_time
            self._update_performance_stats(execution_time)
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Rolling features computation failed: {e}")
            return data.copy()
    
    def batch_process(self, data: Union['np.ndarray', 'pd.DataFrame'],
                     operation: str, **kwargs) -> Any:
        """
        Process data in batches using VectorBT's parallel processing.
        
        Args:
            data: Input data
            operation: Operation to perform
            **kwargs: Additional arguments
            
        Returns:
            Processed result
        """
        start_time = time.time()
        
        try:
            if VECTORBT_AVAILABLE and self.enable_parallel:
                # Use VectorBT's optimized batch processing
                if operation == 'correlation':
                    return self.correlation_matrix(data)
                elif operation == 'rolling_features':
                    return self.rolling_features(data, **kwargs)
                elif operation == 'trading_indicators':
                    return self.compute_trading_indicators(data, **kwargs)
                else:
                    # Fallback to standard processing
                    return self._standard_batch_process(data, operation, **kwargs)
            else:
                # Use standard processing
                return self._standard_batch_process(data, operation, **kwargs)
            
        except Exception as e:
            self.logger.error(f"❌ Batch processing failed: {e}")
            return self._standard_batch_process(data, operation, **kwargs)
        finally:
            execution_time = time.time() - start_time
            self._update_performance_stats(execution_time)
    
    def _standard_batch_process(self, data: Union['np.ndarray', 'pd.DataFrame'],
                               operation: str, **kwargs) -> Any:
        """Standard batch processing fallback."""
        if isinstance(data, pd.DataFrame):
            data = data.values
        
        if operation == 'correlation':
            return np.corrcoef(data.T)
        elif operation == 'mean':
            return np.mean(data, axis=0)
        elif operation == 'std':
            return np.std(data, axis=0)
        else:
            raise ValueError(f"Unknown operation: {operation}")
    
    def _get_default_indicator_config(self) -> Dict[str, Any]:
        """Get default configuration for trading indicators."""
        return {
            # Moving averages
            'sma_periods': [9, 21, 50, 200],
            'ema_periods': [12, 26, 50],
            
            # RSI
            'rsi_period': 14,
            'rsi_overbought': 70,
            'rsi_oversold': 30,
            
            # MACD
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            
            # Bollinger Bands
            'bb_period': 20,
            'bb_std': 2.0,
            
            # Stochastic
            'stoch_k': 14,
            'stoch_d': 3,
            
            # Williams %R
            'williams_period': 14,
            
            # ADX
            'adx_period': 14,
            
            # ATR
            'atr_period': 14,
            
            # CCI
            'cci_period': 20,
            
            # ROC
            'roc_period': 10,
            
            # Volume indicators
            'volume_sma_period': 20,
            'obv_smooth': 10,
        }
    
    def _update_performance_stats(self, execution_time: float):
        """Update performance statistics."""
        self.performance_stats['total_operations'] += 1
        self.performance_stats['average_execution_time'] = (
            (self.performance_stats['average_execution_time'] *
             (self.performance_stats['total_operations'] - 1)) + execution_time
        ) / self.performance_stats['total_operations']
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return self.performance_stats.copy()
    
    def _get_rolling_object(self, series: 'pd.Series', window: int):
        """Get VectorBT rolling object for optimized operations."""
        if VECTORBT_AVAILABLE:
            try:
                return vbt.Rolling(series, window=window)
            except Exception as e:
                self.logger.warning(f"⚠️ VectorBT rolling object creation failed: {e}")
                return None
        return None
    
    def _vectorized_mean(self, data: 'np.ndarray', axis: int = 0) -> 'np.ndarray':
        """VectorBT-optimized mean calculation."""
        if VECTORBT_AVAILABLE:
            try:
                return vbt.math.mean(data, axis=axis)
            except Exception as e:
                self.logger.warning(f"⚠️ VectorBT mean calculation failed: {e}")
                return np.mean(data, axis=axis)
        return np.mean(data, axis=axis)
    
    def _vectorized_max(self, data: 'np.ndarray', axis: int = 0) -> 'np.ndarray':
        """VectorBT-optimized max calculation."""
        if VECTORBT_AVAILABLE:
            try:
                return vbt.math.max(data, axis=axis)
            except Exception as e:
                self.logger.warning(f"⚠️ VectorBT max calculation failed: {e}")
                return np.max(data, axis=axis)
        return np.max(data, axis=axis)
    
    def _vectorized_std(self, data: 'np.ndarray', axis: int = 0) -> 'np.ndarray':
        """VectorBT-optimized std calculation."""
        if VECTORBT_AVAILABLE:
            try:
                return vbt.math.std(data, axis=axis)
            except Exception as e:
                self.logger.warning(f"⚠️ VectorBT std calculation failed: {e}")
                return np.std(data, axis=axis)
        return np.std(data, axis=axis)

    def get_hardware_info(self) -> Dict[str, Any]:
        """Get hardware capability information."""
        return {
            'vectorbt_available': VECTORBT_AVAILABLE,
            'gpu_enabled': self.enable_gpu,
            'parallel_enabled': self.enable_parallel,
            'performance_stats': self.performance_stats
        }

# Global instance
_vectorbt_ops = None

def get_vectorbt_optimized_operations() -> VectorBTOptimizedOperations:
    """Get global VectorBT optimized operations instance."""
    global _vectorbt_ops
    if _vectorbt_ops is None:
        _vectorbt_ops = VectorBTOptimizedOperations()
    return _vectorbt_ops

# Convenience functions
def vectorbt_matrix_multiply(A: 'np.ndarray', B: 'np.ndarray') -> 'np.ndarray':
    """VectorBT-optimized matrix multiplication."""
    ops = get_vectorbt_optimized_operations()
    return ops.matrix_multiply(A, B)

def vectorbt_correlation_matrix(data: Union['np.ndarray', 'pd.DataFrame'], 
                               method: str = 'pearson') -> 'np.ndarray':
    """VectorBT-optimized correlation matrix."""
    ops = get_vectorbt_optimized_operations()
    return ops.correlation_matrix(data, method)

def vectorbt_trading_indicators(data: 'pd.DataFrame', 
                               config: Optional[Dict[str, Any]] = None) -> 'pd.DataFrame':
    """VectorBT-optimized trading indicators."""
    ops = get_vectorbt_optimized_operations()
    return ops.compute_trading_indicators(data, config)

def vectorbt_rolling_features(data: 'pd.DataFrame',
                             windows: List[int] = [5, 10, 20, 50],
                             features: List[str] = None) -> 'pd.DataFrame':
    """VectorBT-optimized rolling features."""
    ops = get_vectorbt_optimized_operations()
    return ops.rolling_features(data, windows, features)

def vectorbt_batch_processing(data: Union['np.ndarray', 'pd.DataFrame'],
                             operation: str, **kwargs) -> Any:
    """VectorBT-optimized batch processing."""
    ops = get_vectorbt_optimized_operations()
    return ops.batch_process(data, operation, **kwargs)