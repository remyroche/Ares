"""Technical indicators and market microstructure features.

This module provides various technical analysis indicators and
market microstructure features for financial time series.
"""

from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
import numpy as np
from numba import njit, prange
import talib

from src.utils.logger import system_logger
from src.core.decorators import handles_errors


class TechnicalIndicatorCalculator:
    """Calculates various technical indicators efficiently."""
    
    def __init__(self):
        self.logger = system_logger.getChild("TechnicalIndicatorCalculator")
        
    @staticmethod
    @njit
    def _ema_numba(values: np.ndarray, span: int) -> np.ndarray:
        """Numba-optimized exponential moving average."""
        alpha = 2.0 / (span + 1)
        result = np.empty_like(values)
        result[0] = values[0]
        
        for i in range(1, len(values)):
            result[i] = alpha * values[i] + (1 - alpha) * result[i-1]
            
        return result
    
    @staticmethod
    @njit
    def _rsi_numba(values: np.ndarray, period: int) -> np.ndarray:
        """Numba-optimized RSI calculation."""
        deltas = np.diff(values)
        seed = deltas[:period+1]
        up = seed[seed >= 0].sum() / period
        down = -seed[seed < 0].sum() / period
        rs = up / down if down != 0 else 0
        rsi = np.empty(len(values))
        rsi[:period] = np.nan
        rsi[period] = 100 - 100 / (1 + rs)
        
        for i in range(period + 1, len(values)):
            delta = deltas[i-1]
            if delta > 0:
                upval = delta
                downval = 0
            else:
                upval = 0
                downval = -delta
                
            up = (up * (period - 1) + upval) / period
            down = (down * (period - 1) + downval) / period
            rs = up / down if down != 0 else 0
            rsi[i] = 100 - 100 / (1 + rs)
            
        return rsi
    
    def calculate_price_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate price-based technical features.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with price features
        """
        features = pd.DataFrame(index=data.index)
        
        # Price ratios
        features['high_low_ratio'] = data['high'] / (data['low'] + 1e-10)
        features['close_open_ratio'] = data['close'] / (data['open'] + 1e-10)
        
        # Price position within bar
        hl_range = data['high'] - data['low'] + 1e-10
        features['close_position'] = (data['close'] - data['low']) / hl_range
        features['open_position'] = (data['open'] - data['low']) / hl_range
        
        # Returns
        features['returns'] = data['close'].pct_change()
        features['log_returns'] = np.log(data['close'] / data['close'].shift(1))
        
        # Moving averages
        for period in [5, 10, 20, 50]:
            features[f'sma_{period}'] = data['close'].rolling(period).mean()
            features[f'ema_{period}'] = self._ema_numba(
                data['close'].values, period
            )
            
        # Price relative to moving averages
        for period in [5, 10, 20, 50]:
            ma_col = f'sma_{period}'
            if ma_col in features:
                features[f'close_to_sma_{period}'] = (
                    data['close'] / features[ma_col] - 1
                )
        
        return features
    
    def calculate_momentum_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate momentum-based features.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with momentum features
        """
        features = pd.DataFrame(index=data.index)
        
        # RSI
        for period in [6, 14, 28]:
            features[f'rsi_{period}'] = self._rsi_numba(
                data['close'].values, period
            )
        
        # MACD
        if len(data) >= 26:
            macd, signal, hist = talib.MACD(
                data['close'].values,
                fastperiod=12,
                slowperiod=26,
                signalperiod=9
            )
            features['macd'] = macd
            features['macd_signal'] = signal
            features['macd_hist'] = hist
        
        # Stochastic Oscillator
        if len(data) >= 14:
            slowk, slowd = talib.STOCH(
                data['high'].values,
                data['low'].values,
                data['close'].values,
                fastk_period=14,
                slowk_period=3,
                slowd_period=3
            )
            features['stoch_k'] = slowk
            features['stoch_d'] = slowd
        
        # Rate of Change
        for period in [5, 10, 20]:
            features[f'roc_{period}'] = (
                (data['close'] - data['close'].shift(period)) / 
                data['close'].shift(period)
            ) * 100
        
        return features
    
    def calculate_volatility_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate volatility-based features.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with volatility features
        """
        features = pd.DataFrame(index=data.index)
        
        # Historical volatility
        returns = data['close'].pct_change()
        for period in [5, 10, 20, 50]:
            features[f'volatility_{period}'] = returns.rolling(period).std()
        
        # Average True Range
        if len(data) >= 14:
            features['atr_14'] = talib.ATR(
                data['high'].values,
                data['low'].values,
                data['close'].values,
                timeperiod=14
            )
        
        # Bollinger Bands
        for period in [20, 50]:
            sma = data['close'].rolling(period).mean()
            std = data['close'].rolling(period).std()
            
            features[f'bb_upper_{period}'] = sma + 2 * std
            features[f'bb_lower_{period}'] = sma - 2 * std
            features[f'bb_width_{period}'] = 4 * std
            features[f'bb_position_{period}'] = (
                (data['close'] - features[f'bb_lower_{period}']) / 
                (features[f'bb_width_{period}'] + 1e-10)
            )
        
        # Keltner Channels
        if 'atr_14' in features:
            ema_20 = features.get('ema_20', data['close'].ewm(span=20).mean())
            features['kc_upper'] = ema_20 + 2 * features['atr_14']
            features['kc_lower'] = ema_20 - 2 * features['atr_14']
            features['kc_position'] = (
                (data['close'] - features['kc_lower']) / 
                ((features['kc_upper'] - features['kc_lower']) + 1e-10)
            )
        
        return features
    
    def calculate_volume_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate volume-based features.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with volume features
        """
        features = pd.DataFrame(index=data.index)
        
        # Volume moving averages
        for period in [5, 10, 20]:
            features[f'volume_sma_{period}'] = data['volume'].rolling(period).mean()
            features[f'volume_ratio_{period}'] = (
                data['volume'] / features[f'volume_sma_{period}']
            )
        
        # On Balance Volume
        obv = (np.sign(data['close'].diff()) * data['volume']).cumsum()
        features['obv'] = obv
        features['obv_ema_5'] = obv.ewm(span=5).mean()
        features['obv_ema_20'] = obv.ewm(span=20).mean()
        
        # Volume Price Trend
        features['vpt'] = (
            (data['close'].pct_change() * data['volume']).cumsum()
        )
        
        # Money Flow Index
        if len(data) >= 14:
            features['mfi_14'] = talib.MFI(
                data['high'].values,
                data['low'].values,
                data['close'].values,
                data['volume'].values,
                timeperiod=14
            )
        
        # Accumulation/Distribution Line
        clv = ((data['close'] - data['low']) - (data['high'] - data['close'])) / \
              ((data['high'] - data['low']) + 1e-10)
        features['ad_line'] = (clv * data['volume']).cumsum()
        
        return features
    
    def calculate_all_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicator features.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with all features
        """
        self.logger.info("Calculating technical indicators...")
        
        # Calculate each feature group
        price_features = self.calculate_price_features(data)
        momentum_features = self.calculate_momentum_features(data)
        volatility_features = self.calculate_volatility_features(data)
        volume_features = self.calculate_volume_features(data)
        
        # Combine all features
        all_features = pd.concat([
            price_features,
            momentum_features,
            volatility_features,
            volume_features
        ], axis=1)
        
        # Fill NaN values
        all_features = all_features.fillna(method='ffill').fillna(0)
        
        self.logger.info(f"Calculated {len(all_features.columns)} technical indicators")
        
        return all_features