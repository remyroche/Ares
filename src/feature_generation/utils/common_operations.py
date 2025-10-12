"""
Common Operations Utilities

This module provides common operations that are frequently used across
feature generators, eliminating code duplication and ensuring consistency.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from dataclasses import dataclass
from enum import Enum

from .centralized_rolling_manager import get_centralized_rolling_manager, RollingOperation
from .scaler_factory import get_scaler_factory, ScalerType
from .unified_vectorization_manager import get_unified_vectorization_manager

logger = logging.getLogger(__name__)

@dataclass
class TechnicalIndicatorConfig:
    """Configuration for technical indicator calculations."""
    window: int
    method: str = 'vectorbt'
    smoothing: Optional[int] = None
    normalization: Optional[str] = None
    custom_params: Optional[Dict[str, Any]] = None

class CommonOperations:
    """
    Common operations utilities for feature generators.
    
    This class provides frequently used operations to eliminate code duplication
    across feature generators.
    """
    
    def __init__(self):
        """Initialize common operations utilities."""
        self.rolling_manager = get_centralized_rolling_manager()
        self.scaler_factory = get_scaler_factory()
        self.vectorization_manager = get_unified_vectorization_manager()
        
        # Performance tracking
        self._performance_stats = {
            'operations_executed': 0,
            'vectorbt_operations': 0,
            'pandas_fallbacks': 0,
            'batch_operations': 0,
            'normalization_operations': 0
        }
    
    def calculate_rolling_statistics(self, data: pd.Series, window: int, 
                                   operations: List[Union[str, RollingOperation]] = None,
                                   **kwargs) -> Dict[str, pd.Series]:
        """
        Calculate multiple rolling statistics efficiently.
        
        Args:
            data: Input data series
            window: Rolling window size
            operations: List of operations to perform
            **kwargs: Additional operation parameters
            
        Returns:
            Dictionary mapping operation names to resulting series
        """
        if operations is None:
            operations = [RollingOperation.MEAN, RollingOperation.STD]
        
        results = {}
        
        for operation in operations:
            if isinstance(operation, str):
                operation = RollingOperation(operation.lower())
            
            try:
                result = self.rolling_manager.rolling_operation(operation, data, window, **kwargs)
                results[operation.value] = result
                self._performance_stats['operations_executed'] += 1
            except Exception as e:
                logger.warning(f"Failed to calculate {operation.value}: {e}")
        
        return results
    
    def calculate_technical_indicator(self, data: pd.DataFrame, indicator: str, 
                                    params: Dict[str, Any]) -> Union[pd.Series, pd.DataFrame]:
        """
        Calculate technical indicators using centralized utilities.
        
        Args:
            data: Input DataFrame with OHLCV data
            indicator: Name of the technical indicator
            params: Parameters for the indicator
            
        Returns:
            Calculated indicator values
        """
        try:
            if indicator.lower() == 'sma':
                return self._calculate_sma(data, params)
            elif indicator.lower() == 'ema':
                return self._calculate_ema(data, params)
            elif indicator.lower() == 'rsi':
                return self._calculate_rsi(data, params)
            elif indicator.lower() == 'macd':
                return self._calculate_macd(data, params)
            elif indicator.lower() == 'bollinger_bands':
                return self._calculate_bollinger_bands(data, params)
            elif indicator.lower() == 'atr':
                return self._calculate_atr(data, params)
            elif indicator.lower() == 'stochastic':
                return self._calculate_stochastic(data, params)
            elif indicator.lower() == 'williams_r':
                return self._calculate_williams_r(data, params)
            else:
                raise ValueError(f"Unsupported technical indicator: {indicator}")
        except Exception as e:
            logger.error(f"Failed to calculate {indicator}: {e}")
            return pd.Series(dtype=float, index=data.index)
    
    def normalize_feature(self, data: pd.Series, method: str = 'zscore', 
                         feature_type: str = 'default', **kwargs) -> pd.Series:
        """
        Normalize feature data using centralized scalers.
        
        Args:
            data: Input data series
            method: Normalization method
            feature_type: Type of feature for appropriate scaler selection
            **kwargs: Additional scaler parameters
            
        Returns:
            Normalized data series
        """
        try:
            scaler = self.scaler_factory.get_scaler_for_feature_type(feature_type)
            normalized_data = scaler.fit_transform(data)
            self._performance_stats['normalization_operations'] += 1
            return normalized_data
        except Exception as e:
            logger.warning(f"Normalization failed: {e}, returning original data")
            return data
    
    def calculate_price_levels(self, data: pd.DataFrame, 
                              levels: List[str] = None) -> Dict[str, pd.Series]:
        """
        Calculate common price levels (OHLC, HL2, HLC3, etc.).
        
        Args:
            data: Input DataFrame with OHLCV data
            levels: List of price levels to calculate
            
        Returns:
            Dictionary mapping level names to calculated series
        """
        if levels is None:
            levels = ['close', 'hl2', 'hlc3', 'ohlc4']
        
        results = {}
        
        for level in levels:
            try:
                if level == 'close' and 'close' in data.columns:
                    results[level] = data['close']
                elif level == 'hl2' and all(col in data.columns for col in ['high', 'low']):
                    results[level] = (data['high'] + data['low']) / 2
                elif level == 'hlc3' and all(col in data.columns for col in ['high', 'low', 'close']):
                    results[level] = (data['high'] + data['low'] + data['close']) / 3
                elif level == 'ohlc4' and all(col in data.columns for col in ['open', 'high', 'low', 'close']):
                    results[level] = (data['open'] + data['high'] + data['low'] + data['close']) / 4
                elif level == 'vwap' and all(col in data.columns for col in ['high', 'low', 'close', 'volume']):
                    results[level] = self._calculate_vwap(data)
                else:
                    logger.warning(f"Required columns not available for {level}")
            except Exception as e:
                logger.warning(f"Failed to calculate {level}: {e}")
        
        return results
    
    def calculate_returns(self, data: pd.Series, method: str = 'simple', 
                         periods: List[int] = None) -> Dict[str, pd.Series]:
        """
        Calculate returns using different methods and periods.
        
        Args:
            data: Input price series
            method: Return calculation method ('simple', 'log', 'cumulative')
            periods: List of periods for calculation
            
        Returns:
            Dictionary mapping period names to return series
        """
        if periods is None:
            periods = [1, 5, 10, 20]
        
        results = {}
        
        for period in periods:
            try:
                if method == 'simple':
                    returns = data.pct_change(periods=period)
                elif method == 'log':
                    returns = np.log(data / data.shift(period))
                elif method == 'cumulative':
                    returns = (data / data.shift(period)) - 1
                else:
                    raise ValueError(f"Unsupported return method: {method}")
                
                results[f'{method}_returns_{period}'] = returns
            except Exception as e:
                logger.warning(f"Failed to calculate {method} returns for period {period}: {e}")
        
        return results
    
    def calculate_volatility_measures(self, data: pd.DataFrame, 
                                    window: int = 20) -> Dict[str, pd.Series]:
        """
        Calculate various volatility measures.
        
        Args:
            data: Input DataFrame with OHLCV data
            window: Rolling window size
            
        Returns:
            Dictionary mapping measure names to calculated series
        """
        results = {}
        
        try:
            # Price-based volatility
            if 'close' in data.columns:
                returns = data['close'].pct_change()
                results['price_volatility'] = self.rolling_manager.rolling_std(returns, window)
                results['price_variance'] = self.rolling_manager.rolling_var(returns, window)
            
            # Volume-based volatility
            if 'volume' in data.columns:
                volume_returns = data['volume'].pct_change()
                results['volume_volatility'] = self.rolling_manager.rolling_std(volume_returns, window)
            
            # High-Low volatility
            if all(col in data.columns for col in ['high', 'low']):
                hl_range = (data['high'] - data['low']) / data['close']
                results['hl_volatility'] = self.rolling_manager.rolling_mean(hl_range, window)
            
            # Parkinson volatility
            if all(col in data.columns for col in ['high', 'low']):
                parkinson = np.log(data['high'] / data['low']) ** 2
                results['parkinson_volatility'] = self.rolling_manager.rolling_mean(parkinson, window)
            
        except Exception as e:
            logger.warning(f"Failed to calculate volatility measures: {e}")
        
        return results
    
    def calculate_momentum_indicators(self, data: pd.DataFrame, 
                                    window: int = 14) -> Dict[str, pd.Series]:
        """
        Calculate momentum indicators.
        
        Args:
            data: Input DataFrame with OHLCV data
            window: Rolling window size
            
        Returns:
            Dictionary mapping indicator names to calculated series
        """
        results = {}
        
        try:
            if 'close' in data.columns:
                close = data['close']
                
                # Rate of Change
                results['roc'] = (close / close.shift(window)) - 1
                
                # Momentum
                results['momentum'] = close - close.shift(window)
                
                # Price acceleration
                momentum = results['momentum']
                results['acceleration'] = momentum - momentum.shift(1)
                
                # Price velocity
                results['velocity'] = momentum / window
                
        except Exception as e:
            logger.warning(f"Failed to calculate momentum indicators: {e}")
        
        return results
    
    def batch_process_features(self, data: pd.DataFrame, 
                              feature_configs: List[Dict[str, Any]]) -> Dict[str, pd.Series]:
        """
        Process multiple features in batch for efficiency.
        
        Args:
            data: Input DataFrame
            feature_configs: List of feature configuration dictionaries
            
        Returns:
            Dictionary mapping feature names to calculated series
        """
        results = {}
        
        try:
            # Use vectorization manager for batch processing
            batch_operations = []
            
            for config in feature_configs:
                feature_name = config.get('name', 'unknown')
                operation = config.get('operation')
                params = config.get('params', {})
                
                if operation == 'rolling_mean':
                    series = self.rolling_manager.rolling_mean(
                        data[config['column']], 
                        window=params.get('window', 20)
                    )
                    results[feature_name] = series
                
                elif operation == 'rolling_std':
                    series = self.rolling_manager.rolling_std(
                        data[config['column']], 
                        window=params.get('window', 20)
                    )
                    results[feature_name] = series
                
                elif operation == 'technical_indicator':
                    indicator_result = self.calculate_technical_indicator(
                        data, 
                        config['indicator'], 
                        params
                    )
                    if isinstance(indicator_result, pd.Series):
                        results[feature_name] = indicator_result
                    elif isinstance(indicator_result, pd.DataFrame):
                        for col in indicator_result.columns:
                            results[f"{feature_name}_{col}"] = indicator_result[col]
            
            self._performance_stats['batch_operations'] += 1
            
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
        
        return results
    
    # Private helper methods for specific indicators
    
    def _calculate_sma(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """Calculate Simple Moving Average."""
        window = params.get('window', 20)
        column = params.get('column', 'close')
        return self.rolling_manager.rolling_mean(data[column], window)
    
    def _calculate_ema(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """Calculate Exponential Moving Average."""
        window = params.get('window', 20)
        column = params.get('column', 'close')
        alpha = 2.0 / (window + 1)
        return data[column].ewm(alpha=alpha).mean()
    
    def _calculate_rsi(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """Calculate Relative Strength Index."""
        window = params.get('window', 14)
        column = params.get('column', 'close')
        
        close = data[column]
        delta = close.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = self.rolling_manager.rolling_mean(gain, window)
        avg_loss = self.rolling_manager.rolling_mean(loss, window)
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_macd(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """Calculate MACD."""
        fast_window = params.get('fast_window', 12)
        slow_window = params.get('slow_window', 26)
        signal_window = params.get('signal_window', 9)
        column = params.get('column', 'close')
        
        close = data[column]
        
        ema_fast = close.ewm(span=fast_window).mean()
        ema_slow = close.ewm(span=slow_window).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal_window).mean()
        histogram = macd_line - signal_line
        
        return pd.DataFrame({
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        })
    
    def _calculate_bollinger_bands(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """Calculate Bollinger Bands."""
        window = params.get('window', 20)
        std_dev = params.get('std_dev', 2)
        column = params.get('column', 'close')
        
        close = data[column]
        sma = self.rolling_manager.rolling_mean(close, window)
        std = self.rolling_manager.rolling_std(close, window)
        
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        
        return pd.DataFrame({
            'upper': upper_band,
            'middle': sma,
            'lower': lower_band,
            'width': upper_band - lower_band,
            'position': (close - lower_band) / (upper_band - lower_band)
        })
    
    def _calculate_atr(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """Calculate Average True Range."""
        window = params.get('window', 14)
        
        high = data['high']
        low = data['low']
        close = data['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = self.rolling_manager.rolling_mean(true_range, window)
        
        return atr
    
    def _calculate_stochastic(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """Calculate Stochastic Oscillator."""
        k_window = params.get('k_window', 14)
        d_window = params.get('d_window', 3)
        
        high = data['high']
        low = data['low']
        close = data['close']
        
        lowest_low = self.rolling_manager.rolling_min(low, k_window)
        highest_high = self.rolling_manager.rolling_max(high, k_window)
        
        k_percent = 100 * (close - lowest_low) / (highest_high - lowest_low)
        d_percent = self.rolling_manager.rolling_mean(k_percent, d_window)
        
        return pd.DataFrame({
            'k_percent': k_percent,
            'd_percent': d_percent
        })
    
    def _calculate_williams_r(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """Calculate Williams %R."""
        window = params.get('window', 14)
        
        high = data['high']
        low = data['low']
        close = data['close']
        
        highest_high = self.rolling_manager.rolling_max(high, window)
        lowest_low = self.rolling_manager.rolling_min(low, window)
        
        williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
        
        return williams_r
    
    def _calculate_vwap(self, data: pd.DataFrame) -> pd.Series:
        """Calculate Volume Weighted Average Price."""
        if not all(col in data.columns for col in ['high', 'low', 'close', 'volume']):
            raise ValueError("VWAP requires high, low, close, and volume columns")
        
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        vwap = (typical_price * data['volume']).cumsum() / data['volume'].cumsum()
        
        return vwap
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return self._performance_stats.copy()
    
    def reset_performance_stats(self):
        """Reset performance statistics."""
        self._performance_stats = {
            'operations_executed': 0,
            'vectorbt_operations': 0,
            'pandas_fallbacks': 0,
            'batch_operations': 0,
            'normalization_operations': 0
        }

# Global instance
_common_operations = None

def get_common_operations() -> CommonOperations:
    """Get the global common operations instance."""
    global _common_operations
    if _common_operations is None:
        _common_operations = CommonOperations()
    return _common_operations