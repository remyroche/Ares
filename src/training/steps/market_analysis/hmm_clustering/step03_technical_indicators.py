from src.training.steps.standardized_parquet_handler import standardized_parquet_handler
#!/usr/bin/env python3
"""Step03 Technical Indicators Utility.

Centralized technical indicator calculations to eliminate code duplication
across modules. All indicators are vectorized and optimized for performance.
"""

import pandas as pd
from typing import Dict, Tuple, Optional, Union, List
import warnings
from numba import jit, prange
import logging
from src.utils.comprehensive_function_logger import log_step_functions, log_important_calls, log_all_calls, log_internal_call, log_step_progress, log_data_operation
import numpy as np

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class TechnicalIndicators:
    """Centralized technical indicator calculations."""

    @log_important_calls
    def __init__(self, config: Optional[Dict] = None):
        """Initialize with optional configuration."""
        self.config = config or {}
        self._cache = {}
        self._enable_caching = self.config.get('enable_caching', True)

    @log_all_calls
    def _get_cached_or_compute(self, key: str, compute_func, *args, **kwargs):
        """Get from cache or compute and cache result."""
        if not self._enable_caching or key not in self._cache:
            result = compute_func(*args, **kwargs)
            if self._enable_caching:
                self._cache[key] = result
            return result
        return self._cache[key]
    
    def clear_cache(self):
        """Clear the calculation cache."""
        self._cache.clear()
    
    def calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        cache_key = f"rsi_{hash(prices.name)}_{window}_{len(prices)}"

        def _compute_rsi():
            delta = prices.diff()
            gain = delta.where(delta > 0, 0).rolling(window = window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window = window).mean()
            rs = gain / loss
            rsi = 100 - 100 / (1 + rs)
            return rsi
        
        return self._get_cached_or_compute(cache_key, _compute_rsi)
    
    def calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """Calculate MACD (Moving Average Convergence Divergence)."""
        cache_key = f"macd_{hash(prices.name)}_{fast}_{slow}_{signal}_{len(prices)}"

        def _compute_macd():
            ema_fast = prices.ewm(span = fast).mean()
            ema_slow = prices.ewm(span = slow).mean()
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span = signal).mean()
            histogram = macd_line - signal_line
            
            return {
                'macd': macd_line,
                'signal': signal_line,
                'histogram': histogram
            }
        
        return self._get_cached_or_compute(cache_key, _compute_macd)
    
    def calculate_bollinger_bands(self, prices: pd.Series, window: int = 20, num_std: float = 2) -> Dict[str, pd.Series]:
        """Calculate Bollinger Bands."""
        cache_key = f"bb_{hash(prices.name)}_{window}_{num_std}_{len(prices)}"

        def _compute_bb():
            sma = prices.rolling(window = window).mean()
            std = prices.rolling(window = window).std()
            upper = sma + std * num_std
            lower = sma - std * num_std
            width = (upper - lower) / sma
            position = (prices - lower) / (upper - lower)
            
            return {
                'upper': upper,
                'middle': sma,
                'lower': lower,
                'width': width,
                'position': position
            }
        
        return self._get_cached_or_compute(cache_key, _compute_bb)
    
    def calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        cache_key = f"atr_{hash(high.name)}_{window}_{len(high)}"

        def _compute_atr():
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            tr = pd.concat([tr1, tr2, tr3], axis = 1).max(axis = 1)
            atr = tr.rolling(window = window).mean()
            return atr
        
        return self._get_cached_or_compute(cache_key, _compute_atr)
    
    def calculate_adx(self, high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Average Directional Index."""
        cache_key = f"adx_{hash(high.name)}_{window}_{len(high)}"

        def _compute_adx():
            # True Range
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            tr = pd.concat([tr1, tr2, tr3], axis = 1).max(axis = 1)
            
            # Directional Movement
            dm_plus = high - high.shift(1)
            dm_minus = low.shift(1) - low
            dm_plus = dm_plus.where((dm_plus > dm_minus) & (dm_plus > 0), 0)
            dm_minus = dm_minus.where((dm_minus > dm_plus) & (dm_minus > 0), 0)
            
            # Smoothed values
            tr_smooth = tr.rolling(window = window).mean()
            dm_plus_smooth = dm_plus.rolling(window = window).mean()
            dm_minus_smooth = dm_minus.rolling(window = window).mean()
            
            # Directional Indicators
            di_plus = 100 * (dm_plus_smooth / tr_smooth)
            di_minus = 100 * (dm_minus_smooth / tr_smooth)
            
            # DX and ADX
            dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
            adx = dx.rolling(window = window).mean()
            
            return adx
        
        return self._get_cached_or_compute(cache_key, _compute_adx)
    
    def calculate_moving_averages(self, prices: pd.Series, windows: List[int]) -> Dict[str, pd.Series]:
        """Calculate multiple moving averages."""
        cache_key = f"ma_{hash(prices.name)}_{str(windows)}_{len(prices)}"

        def _compute_ma():
            result = {}
            for window in windows:
                result[f'sma_{window}'] = prices.rolling(window = window).mean()
                result[f'ema_{window}'] = prices.ewm(span = window).mean()
            return result
        
        return self._get_cached_or_compute(cache_key, _compute_ma)
    
    def calculate_momentum_indicators(self, prices: pd.Series, windows: List[int]) -> Dict[str, pd.Series]:
        """Calculate momentum indicators for multiple windows."""
        cache_key = f"momentum_{hash(prices.name)}_{str(windows)}_{len(prices)}"

        def _compute_momentum():
            result = {}
            for window in windows:
                result[f'momentum_{window}'] = prices.pct_change(window)
                result[f'momentum_acc_{window}'] = prices.pct_change(window).diff()
            return result
        
        return self._get_cached_or_compute(cache_key, _compute_momentum)
    
    def calculate_volume_indicators(self, volume: pd.Series, prices: pd.Series, windows: List[int]) -> Dict[str, pd.Series]:
        """Calculate volume-based indicators."""
        cache_key = f"volume_{hash(volume.name)}_{str(windows)}_{len(volume)}"

        def _compute_volume():
            result = {}
            for window in windows:
                result[f'volume_ma_{window}'] = volume.rolling(window = window).mean()
                result[f'volume_ratio_{window}'] = volume / volume.rolling(window = window).mean()
                result[f'volume_momentum_{window}'] = volume.pct_change(window)
            
            # Volume-Price Trend
            price_change = prices.pct_change()
            result['vpt'] = (price_change * volume).cumsum()
            
            # Volume spikes
            volume_ma_20 = volume.rolling(20).mean()
            volume_std_20 = volume.rolling(20).std()
            result['volume_spike'] = (volume > volume_ma_20 + 2 * volume_std_20).astype(int)
            
            return result
        
        return self._get_cached_or_compute(cache_key, _compute_volume)
    
    def calculate_volatility_indicators(self, prices: pd.Series, windows: List[int]) -> Dict[str, pd.Series]:
        """Calculate volatility indicators."""
        cache_key = f"volatility_{hash(prices.name)}_{str(windows)}_{len(prices)}"

        def _compute_volatility():
            result = {}
            returns = prices.pct_change()
            
            for window in windows:
                result[f'volatility_{window}'] = returns.rolling(window = window).std()
                result[f'vol_momentum_{window}'] = returns.rolling(window = window).std().pct_change()
            
            # Volatility of volatility
            vol_20 = returns.rolling(20).std()
            result['vol_of_vol'] = vol_20.rolling(10).std()
            
            # Volatility regime classification
            vol_100 = returns.rolling(100).std()
            low_threshold = vol_100.rolling(100).quantile(0.33)
            high_threshold = vol_100.rolling(100).quantile(0.67)
            
            vol_regime = pd.Series(1, index = prices.index)
            vol_regime[vol_100 > high_threshold] = 3
            vol_regime[(vol_100 > low_threshold) & (vol_100 <= high_threshold)] = 2
            result['vol_regime'] = vol_regime.fillna(1)
            
            return result
        
        return self._get_cached_or_compute(cache_key, _compute_volatility)
    
    def calculate_price_position_indicators(self, high: pd.Series, low: pd.Series, close: pd.Series, windows: List[int]) -> Dict[str, pd.Series]:
        """Calculate price position indicators."""
        cache_key = f"price_pos_{hash(close.name)}_{str(windows)}_{len(close)}"

        def _compute_price_pos():
            result = {}
            for window in windows:
                rolling_high = high.rolling(window = window).max()
                rolling_low = low.rolling(window = window).min()
                result[f'price_position_{window}'] = (close - rolling_low) / (rolling_high - rolling_low)
            
            # Price range
            result['price_range'] = (high - low) / close
            
            # Gap indicators
            result['gap_up'] = (close > high.shift(1)).astype(int)
            result['gap_down'] = (close < low.shift(1)).astype(int)
            
            return result
        
        return self._get_cached_or_compute(cache_key, _compute_price_pos)
    
    def calculate_all_indicators(self, data: pd.DataFrame, config: Optional[Dict] = None) -> pd.DataFrame:
        """Calculate all technical indicators for a dataset."""
        if config is None:
            config = self.config
        
        indicators = pd.DataFrame(index = data.index)
        
        # Price-based indicators
        if 'close' in data.columns:
            # RSI
            indicators['rsi'] = self.calculate_rsi(data['close'])
            
            # MACD
            macd_data = self.calculate_macd(data['close'])
            indicators['macd'] = macd_data['macd']
            indicators['macd_signal'] = macd_data['signal']
            indicators['macd_histogram'] = macd_data['histogram']
            
            # Bollinger Bands
            bb_data = self.calculate_bollinger_bands(data['close'])
            indicators['bb_upper'] = bb_data['upper']
            indicators['bb_middle'] = bb_data['middle']
            indicators['bb_lower'] = bb_data['lower']
            indicators['bb_width'] = bb_data['width']
            indicators['bb_position'] = bb_data['position']
            
            # Moving averages
            ma_windows = config.get('ma_windows', [20, 50])
            ma_data = self.calculate_moving_averages(data['close'], ma_windows)
            indicators = pd.concat([indicators, pd.DataFrame(ma_data)], axis = 1)
            
            # Momentum
            momentum_windows = config.get('momentum_windows', [5, 10, 20])
            momentum_data = self.calculate_momentum_indicators(data['close'], momentum_windows)
            indicators = pd.concat([indicators, pd.DataFrame(momentum_data)], axis = 1)
            
            # Volatility
            vol_windows = config.get('volatility_windows', [5, 10, 20])
            vol_data = self.calculate_volatility_indicators(data['close'], vol_windows)
            indicators = pd.concat([indicators, pd.DataFrame(vol_data)], axis = 1)
        
        # OHLC-based indicators
        if all(col in data.columns for col in ['high', 'low', 'close']):
            # ATR
            indicators['atr'] = self.calculate_atr(data['high'], data['low'], data['close'])
            
            # ADX
            indicators['adx'] = self.calculate_adx(data['high'], data['low'], data['close'])
            
            # Price position
            pos_windows = config.get('position_windows', [10, 20, 50])
            pos_data = self.calculate_price_position_indicators(data['high'], data['low'], data['close'], pos_windows)
            indicators = pd.concat([indicators, pd.DataFrame(pos_data)], axis = 1)
        
        # Volume-based indicators
        if 'volume' in data.columns and 'close' in data.columns:
            vol_windows = config.get('volume_windows', [5, 10, 20])
            volume_data = self.calculate_volume_indicators(data['volume'], data['close'], vol_windows)
            indicators = pd.concat([indicators, pd.DataFrame(volume_data)], axis = 1)
        
        # Clean up indicators
        indicators = indicators.fillna(method='forward').fillna(0)
        
        return indicators

# Global instance for easy access
_global_indicators = TechnicalIndicators()

def get_technical_indicators(config: Optional[Dict] = None) -> TechnicalIndicators:
    """Get global technical indicators instance."""
    if config:
        _global_indicators.config.update(config)
    return _global_indicators

# Convenience functions for backward compatibility
def calculate_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    """Calculate RSI using global instance."""
    return _global_indicators.calculate_rsi(prices, window)

def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
    """Calculate MACD using global instance."""
    return _global_indicators.calculate_macd(prices, fast, slow, signal)

def calculate_bollinger_bands(prices: pd.Series, window: int = 20, num_std: float = 2) -> Dict[str, pd.Series]:
    """Calculate Bollinger Bands using global instance."""
    return _global_indicators.calculate_bollinger_bands(prices, window, num_std)

def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Calculate ATR using global instance."""
    return _global_indicators.calculate_atr(high, low, close, window)

def calculate_adx(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Calculate ADX using global instance."""
    return _global_indicators.calculate_adx(high, low, close, window)