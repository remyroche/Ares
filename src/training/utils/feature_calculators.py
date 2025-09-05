"""
Feature calculation utilities for matrix optimization.

This module contains all feature calculation methods extracted from the main optimizer
to reduce complexity and improve maintainability.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional


class FeatureCalculator:
    """Utility class for calculating various technical indicators."""
    
    @staticmethod
    def calculate_rsi(prices: pd.Series, period: int) -> pd.Series:
        """Calculate RSI with specific period."""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - 100 / (1 + rs)

    @staticmethod
    def calculate_sma(prices: pd.Series, period: int) -> pd.Series:
        """Calculate SMA with specific period."""
        return prices.rolling(window=period).mean()

    @staticmethod
    def calculate_ema(prices: pd.Series, period: int) -> pd.Series:
        """Calculate EMA with specific period."""
        return prices.ewm(span=period).mean()

    @staticmethod
    def calculate_bollinger_position(data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Bollinger Bands position with specific period."""
        sma = data['close'].rolling(window=period).mean()
        std = data['close'].rolling(window=period).std()
        upper = sma + 2 * std
        lower = sma - 2 * std
        return (data['close'] - lower) / (upper - lower)

    @staticmethod
    def calculate_atr(data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate ATR with specific period."""
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift())
        low_close = np.abs(data['low'] - data['close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return true_range.rolling(window=period).mean()

    @staticmethod
    def calculate_stochastic_k(data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Stochastic %K with specific period."""
        lowest_low = data['low'].rolling(window=period).min()
        highest_high = data['high'].rolling(window=period).max()
        return 100 * ((data['close'] - lowest_low) / (highest_high - lowest_low))

    @staticmethod
    def calculate_stochastic_d(data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Stochastic %D with specific period."""
        k = FeatureCalculator.calculate_stochastic_k(data, period)
        return k.rolling(window=3).mean()

    @staticmethod
    def calculate_adx(data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate ADX with specific period."""
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift())
        low_close = np.abs(data['low'] - data['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        dm_plus = (data['high'] - data['high'].shift()).where(data['high'] - data['high'].shift() > data['low'].shift() - data['low'], 0)
        dm_minus = (data['low'].shift() - data['low']).where(data['low'].shift() - data['low'] > data['high'] - data['high'].shift(), 0)
        di_plus = 100 * (dm_plus.rolling(window=period).mean() / atr)
        di_minus = 100 * (dm_minus.rolling(window=period).mean() / atr)
        dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
        return dx.rolling(window=period).mean()

    @staticmethod
    def calculate_cci(data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate CCI with specific period."""
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        sma = typical_price.rolling(window=period).mean()
        mad = typical_price.rolling(window=period).apply(lambda x: np.mean(np.abs(x - x.mean())))
        return (typical_price - sma) / (0.015 * mad)

    @staticmethod
    def calculate_williams_r(data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Williams %R with specific period."""
        highest_high = data['high'].rolling(window=period).max()
        lowest_low = data['low'].rolling(window=period).min()
        return -100 * ((highest_high - data['close']) / (highest_high - lowest_low))

    @staticmethod
    def calculate_mfi(data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Money Flow Index with specific period."""
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        money_flow = typical_price * data['volume']
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(window=period).sum()
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(window=period).sum()
        return 100 - 100 / (1 + positive_flow / negative_flow)

    @staticmethod
    def calculate_roc(prices: pd.Series, period: int) -> pd.Series:
        """Calculate Rate of Change with specific period."""
        return (prices - prices.shift(period)) / prices.shift(period) * 100

    @staticmethod
    def calculate_mom(prices: pd.Series, period: int) -> pd.Series:
        """Calculate Momentum with specific period."""
        return prices - prices.shift(period)

    @staticmethod
    def calculate_tsi(prices: pd.Series, period: int) -> pd.Series:
        """Calculate True Strength Index with specific period."""
        price_change = prices.diff()
        abs_price_change = abs(price_change)
        smoothed_change = price_change.ewm(span=period).mean()
        smoothed_abs_change = abs_price_change.ewm(span=period).mean()
        return 100 * (smoothed_change / smoothed_abs_change)

    @staticmethod
    def calculate_uo(data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Ultimate Oscillator with specific period."""
        tr = pd.concat([data['high'] - data['low'], abs(data['high'] - data['close'].shift(1)), abs(data['low'] - data['close'].shift(1))], axis=1).max(axis=1)
        bp = data['close'] - pd.concat([data['low'], data['close'].shift(1)], axis=1).min(axis=1)
        avg7 = bp.rolling(window=7).sum() / tr.rolling(window=7).sum()
        avg14 = bp.rolling(window=14).sum() / tr.rolling(window=14).sum()
        avg28 = bp.rolling(window=28).sum() / tr.rolling(window=28).sum()
        return 100 * (4 * avg7 + 2 * avg14 + avg28) / (4 + 2 + 1)

    @staticmethod
    def calculate_ao(data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Awesome Oscillator with specific period."""
        median_price = (data['high'] + data['low']) / 2
        return median_price.rolling(window=5).mean() - median_price.rolling(window=34).mean()

    @staticmethod
    def calculate_cmf(data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Chaikin Money Flow with specific period."""
        mfm = (data['close'] - data['low'] - (data['high'] - data['close'])) / (data['high'] - data['low'])
        mfm = mfm.replace([np.inf, -np.inf], 0)
        mfv = mfm * data['volume']
        return mfv.rolling(window=period).sum() / data['volume'].rolling(window=period).sum()

    @staticmethod
    def calculate_vwap(data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Volume Weighted Average Price with specific period."""
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        return (typical_price * data['volume']).rolling(window=period).sum() / data['volume'].rolling(window=period).sum()

    @staticmethod
    def calculate_obv(data: pd.DataFrame) -> pd.Series:
        """Calculate On Balance Volume."""
        obv = pd.Series(index=data.index, dtype=float)
        obv.iloc[0] = data['volume'].iloc[0]
        for i in range(1, len(data)):
            if data['close'].iloc[i] > data['close'].iloc[i - 1]:
                obv.iloc[i] = obv.iloc[i - 1] + data['volume'].iloc[i]
            elif data['close'].iloc[i] < data['close'].iloc[i - 1]:
                obv.iloc[i] = obv.iloc[i - 1] - data['volume'].iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i - 1]
        return obv

    @staticmethod
    def calculate_ad(data: pd.DataFrame) -> pd.Series:
        """Calculate Accumulation/Distribution Line."""
        clv = (data['close'] - data['low'] - (data['high'] - data['close'])) / (data['high'] - data['low'])
        clv = clv.replace([np.inf, -np.inf], 0)
        return (clv * data['volume']).cumsum()

    @staticmethod
    def calculate_volume_price_trend(data: pd.DataFrame) -> pd.Series:
        """Calculate Volume Price Trend."""
        price_change = data['close'].pct_change()
        return (price_change * data['volume']).cumsum()

    @staticmethod
    def calculate_volume_price_oscillator(data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Volume Price Oscillator with specific period."""
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        vwap = (typical_price * data['volume']).rolling(window=period).sum() / data['volume'].rolling(window=period).sum()
        return (typical_price - vwap) / vwap * 100

    @staticmethod
    def calculate_vwap_momentum(data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate VWAP momentum with specific period."""
        vwap = FeatureCalculator.calculate_vwap(data, period)
        return vwap / vwap.shift(period) - 1

    @staticmethod
    def calculate_vwap_returns(data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate VWAP returns with specific period."""
        vwap = FeatureCalculator.calculate_vwap(data, period)
        return vwap.pct_change()

    @staticmethod
    def calculate_price_vwap_ratio(data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate price to VWAP ratio with specific period."""
        vwap = FeatureCalculator.calculate_vwap(data, period)
        return data['close'] / vwap

    @staticmethod
    def calculate_price_vwap_deviation(data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate price to VWAP deviation with specific period."""
        vwap = FeatureCalculator.calculate_vwap(data, period)
        return (data['close'] - vwap) / vwap

    @staticmethod
    def calculate_price_vwap_spread(data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate price to VWAP spread with specific period."""
        vwap = FeatureCalculator.calculate_vwap(data, period)
        return data['close'] - vwap


class FeatureCalculatorRegistry:
    """Registry for feature calculation methods."""
    
    _calculators = {
        'RSI': FeatureCalculator.calculate_rsi,
        'MACD_fast': FeatureCalculator.calculate_ema,
        'MACD_slow': FeatureCalculator.calculate_ema,
        'Bollinger_Bands': FeatureCalculator.calculate_bollinger_position,
        'SMA_short': FeatureCalculator.calculate_sma,
        'SMA_long': FeatureCalculator.calculate_sma,
        'EMA_short': FeatureCalculator.calculate_ema,
        'EMA_long': FeatureCalculator.calculate_ema,
        'ATR': FeatureCalculator.calculate_atr,
        'Stochastic_k': FeatureCalculator.calculate_stochastic_k,
        'Stochastic_d': FeatureCalculator.calculate_stochastic_d,
        'ADX': FeatureCalculator.calculate_adx,
        'CCI': FeatureCalculator.calculate_cci,
        'Williams_R': FeatureCalculator.calculate_williams_r,
        'MFI': FeatureCalculator.calculate_mfi,
        'ROC': FeatureCalculator.calculate_roc,
        'MOM': FeatureCalculator.calculate_mom,
        'TSI': FeatureCalculator.calculate_tsi,
        'UO': FeatureCalculator.calculate_uo,
        'AO': FeatureCalculator.calculate_ao,
        'CMF': FeatureCalculator.calculate_cmf,
        'VWAP': FeatureCalculator.calculate_vwap,
        'OBV': FeatureCalculator.calculate_obv,
        'AD': FeatureCalculator.calculate_ad,
        'Chaikin_Money_Flow': FeatureCalculator.calculate_cmf,
        'Money_Flow_Index': FeatureCalculator.calculate_mfi,
        'Volume_Price_Trend': FeatureCalculator.calculate_volume_price_trend,
        'Accumulation_Distribution': FeatureCalculator.calculate_ad,
        'On_Balance_Volume': FeatureCalculator.calculate_obv,
        'Volume_Weighted_Average_Price': FeatureCalculator.calculate_vwap,
        'Volume_Price_Oscillator': FeatureCalculator.calculate_volume_price_oscillator,
        'VWAP_Momentum': FeatureCalculator.calculate_vwap_momentum,
        'VWAP_Returns': FeatureCalculator.calculate_vwap_returns,
        'Price_VWAP_Ratio': FeatureCalculator.calculate_price_vwap_ratio,
        'Price_VWAP_Deviation': FeatureCalculator.calculate_price_vwap_deviation,
        'Price_VWAP_Spread': FeatureCalculator.calculate_price_vwap_spread,
    }

    @classmethod
    def calculate_feature(cls, data: pd.DataFrame, feature_name: str, period: int) -> Optional[pd.Series]:
        """Calculate feature using the appropriate calculator."""
        calculator = cls._calculators.get(feature_name)
        if calculator is None:
            return None
        
        try:
            if feature_name in ['RSI', 'ROC', 'MOM', 'TSI']:
                return calculator(data['close'], period)
            elif feature_name in ['MACD_fast', 'MACD_slow', 'SMA_short', 'SMA_long', 'EMA_short', 'EMA_long']:
                return calculator(data['close'], period)
            elif feature_name == 'OBV':
                return calculator(data)
            elif feature_name == 'AD':
                return calculator(data)
            elif feature_name == 'Volume_Price_Trend':
                return calculator(data)
            elif feature_name == 'Accumulation_Distribution':
                return calculator(data)
            elif feature_name == 'On_Balance_Volume':
                return calculator(data)
            else:
                return calculator(data, period)
        except Exception:
            return None