"""
Standardized Feature Calculator for Regime Detection.

This module provides standardized feature calculation utilities that are
consistent across both NAS and TAS regime detection systems. It implements
the same feature calculation logic used in hmm_regime_discovery.py.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from src.utils.logger import system_logger


class StandardizedFeatureCalculator:
    """
    Standardized feature calculator for regime detection systems.

    This class provides consistent feature calculation across both NAS and TAS
    systems, using the same standardized feature set as hmm_regime_discovery.py.
    """

    def __init__(self):
        """Initialize the standardized feature calculator."""
        self.logger = system_logger.getChild('StandardizedFeatureCalculator')

    def calculate_all_features(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all standardized features for regime detection.

        This method calculates the same standardized features used in
        hmm_regime_discovery.py to ensure consistency.

        Args:
            market_data: Market data with OHLCV columns

        Returns:
            DataFrame with all standardized features
        """
        try:
            self.logger.info(f"📊 Calculating standardized features for {len(market_data)} data points")

            # Create copy to avoid modifying original data
            data = market_data.copy()

            # Ensure required columns exist
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in data.columns]
            if missing_cols:
                self.logger.warning(f"⚠️ Missing columns: {missing_cols}, creating fallbacks")
                for col in missing_cols:
                    if col == 'volume':
                        data[col] = 1000  # Default volume
                    else:
                        # Use close price as fallback for OHLC
                        data[col] = data.get('close', 1000)

            # Calculate features in order of dependencies
            features = pd.DataFrame(index=data.index)

            # 1. Price momentum features (12 and 20 periods)
            features['momentum_12'] = self._calculate_momentum(data['close'], 12)
            features['momentum_20'] = self._calculate_momentum(data['close'], 20)

            # 2. Volatility features (12 and 20 periods)
            features['volatility_12'] = self._calculate_volatility(data, 12)
            features['volatility_20'] = self._calculate_volatility(data, 20)

            # 3. Volume features
            features['volume_ratio_192m'] = self._calculate_volume_ratio(data, 192)

            # 4. Trend features
            features['trend_score'] = self._calculate_trend_score(data)

            # 5. Additional technical features
            features['price_change'] = data['close'].pct_change()
            features['volume_change'] = data['volume'].pct_change()
            features['hl_spread'] = (data['high'] - data['low']) / data['close']

            # 6. Rolling statistics
            features['price_std_20'] = data['close'].rolling(20).std()
            features['volume_std_20'] = data['volume'].rolling(20).std()

            # 7. Market regime indicators
            features['regime_indicator'] = self._calculate_regime_indicator(data)
            features['market_structure'] = self._calculate_market_structure(data)

            self.logger.info(f"✅ Calculated {len(features.columns)} standardized features")
            return features

        except Exception as e:
            self.logger.error(f"❌ Feature calculation failed: {e}")
            return pd.DataFrame(index=market_data.index)

    def get_primary_features(self) -> Dict[str, List[str]]:
        """
        Get primary features grouped by dimension for regime detection.

        Returns:
            Dictionary mapping dimensions to feature lists
        """
        return {
            'momentum': ['momentum_12', 'momentum_20'],
            'volatility': ['volatility_12', 'volatility_20'],
            'volume': ['volume_ratio_192m'],
            'trend': ['trend_score']
        }

    def _calculate_momentum(self, prices: pd.Series, period: int) -> pd.Series:
        """
        Calculate momentum over specified period.

        Args:
            prices: Price series
            period: Lookback period

        Returns:
            Momentum series
        """
        try:
            # Simple momentum: (current_price - price_n_periods_ago) / price_n_periods_ago
            shifted = prices.shift(period)
            momentum = (prices - shifted) / shifted

            # Handle NaN values
            momentum = momentum.fillna(0.0)

            return momentum

        except Exception as e:
            self.logger.warning(f"⚠️ Momentum calculation failed: {e}")
            return pd.Series(0.0, index=prices.index)

    def _calculate_volatility(self, data: pd.DataFrame, period: int) -> pd.Series:
        """
        Calculate volatility over specified period.

        Args:
            data: Market data DataFrame
            period: Lookback period

        Returns:
            Volatility series
        """
        try:
            # Calculate true range
            tr = pd.DataFrame({
                'hl': data['high'] - data['low'],
                'hc': (data['high'] - data['close'].shift(1)).abs(),
                'lc': (data['low'] - data['close'].shift(1)).abs()
            }).max(axis=1)

            # Average true range
            atr = tr.rolling(period).mean()

            # Volatility as ATR normalized by price
            volatility = atr / data['close']

            # Handle NaN values
            volatility = volatility.fillna(0.01)  # Small positive fallback

            return volatility

        except Exception as e:
            self.logger.warning(f"⚠️ Volatility calculation failed: {e}")
            return pd.Series(0.01, index=data.index)

    def _calculate_volume_ratio(self, data: pd.DataFrame, period: int) -> pd.Series:
        """
        Calculate volume ratio relative to moving average.

        Args:
            data: Market data DataFrame
            period: Lookback period

        Returns:
            Volume ratio series
        """
        try:
            # Volume moving average
            volume_ma = data['volume'].rolling(period).mean()

            # Volume ratio: current_volume / volume_ma
            volume_ratio = data['volume'] / volume_ma

            # Handle division by zero and NaN
            volume_ratio = volume_ratio.replace([np.inf, -np.inf], 1.0)
            volume_ratio = volume_ratio.fillna(1.0)

            return volume_ratio

        except Exception as e:
            self.logger.warning(f"⚠️ Volume ratio calculation failed: {e}")
            return pd.Series(1.0, index=data.index)

    def _calculate_trend_score(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate trend score using EMA and ADX components.

        Args:
            data: Market data DataFrame

        Returns:
            Trend score series
        """
        try:
            # EMA20 as trend filter
            ema20 = data['close'].ewm(span=20).mean()

            # Direction: 1 for uptrend, -1 for downtrend, 0 for neutral
            direction = pd.Series(0.0, index=data.index)

            # Uptrend: price above EMA20 and EMA20 rising
            uptrend = (data['close'] > ema20) & (ema20 > ema20.shift(1))
            direction[uptrend] = 1.0

            # Downtrend: price below EMA20 and EMA20 falling
            downtrend = (data['close'] < ema20) & (ema20 < ema20.shift(1))
            direction[downtrend] = -1.0

            # Trend strength: distance from EMA20 normalized
            price_distance = (data['close'] - ema20) / data['close']
            strength = price_distance.abs().clip(0, 0.1) * 10  # Scale to 0-1 range

            # Combined trend score: direction * strength
            trend_score = direction * strength

            # Handle NaN values
            trend_score = trend_score.fillna(0.0)

            return trend_score

        except Exception as e:
            self.logger.warning(f"⚠️ Trend score calculation failed: {e}")
            return pd.Series(0.0, index=data.index)

    def _calculate_regime_indicator(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate market regime indicator.

        Args:
            data: Market data DataFrame

        Returns:
            Regime indicator series
        """
        try:
            # Simple regime indicator based on price and volume behavior
            # Positive for bullish regime, negative for bearish
            price_trend = data['close'].pct_change(5).rolling(5).mean()
            volume_trend = data['volume'].pct_change(5).rolling(5).mean()

            # Regime indicator: combination of price and volume trends
            regime_indicator = (price_trend * 0.7 + volume_trend * 0.3)

            # Normalize to [-1, 1] range
            regime_indicator = regime_indicator.clip(-1, 1)

            # Handle NaN values
            regime_indicator = regime_indicator.fillna(0.0)

            return regime_indicator

        except Exception as e:
            self.logger.warning(f"⚠️ Regime indicator calculation failed: {e}")
            return pd.Series(0.0, index=data.index)

    def _calculate_market_structure(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate market structure indicator.

        Args:
            data: Market data DataFrame

        Returns:
            Market structure series
        """
        try:
            # Market structure based on recent price action
            # 1 for structured market, 0 for chaotic/noisy market
            price_range = (data['high'] - data['low']) / data['close']
            volume_consistency = 1.0 / (1.0 + data['volume'].pct_change().abs())

            # Market structure: inverse of price range * volume consistency
            market_structure = (1.0 - price_range.clip(0, 0.1)) * volume_consistency

            # Normalize to [0, 1] range
            market_structure = market_structure.clip(0, 1)

            # Handle NaN values
            market_structure = market_structure.fillna(0.5)

            return market_structure

        except Exception as e:
            self.logger.warning(f"⚠️ Market structure calculation failed: {e}")
            return pd.Series(0.5, index=data.index)