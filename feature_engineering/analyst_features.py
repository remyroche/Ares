"""
Analyst Features - Feature Bank for Financial Analysis

This module provides a comprehensive feature bank for analyst-level features
that can be used across different timeframes and market conditions.

Features are organized into the following categories:
1. Cross-timeframe momentum (CRITICAL)
2. Volatility structure
3. Volume patterns
4. Microstructure
5. Order flow (if available)
6. Regime features
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
from scipy.stats import linregress

logger = logging.getLogger(__name__)


class AnalystFeatureBank:
    """
    Comprehensive feature bank for analyst-level financial features.

    This class provides all the features specified in the analyst_features dictionary,
    organized by category and with proper error handling and validation.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the analyst feature bank.

        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)

        # Feature cache for performance
        self.feature_cache: Dict[str, pd.DataFrame] = {}

        # Timeframe configurations
        self.timeframes = {
            '5m': {'lookback': 20, 'description': 'Short-term momentum'},
            '15m': {'lookback': 20, 'description': 'Medium-term momentum'},
            '1h': {'lookback': 20, 'description': 'Long-term momentum'}
        }

        logger.info("✅ AnalystFeatureBank initialized")

    def generate_analyst_features(
        self,
        data: pd.DataFrame,
        agg_trades: Optional[pd.DataFrame] = None,
        futures_data: Optional[pd.DataFrame] = None,
        regime_data: Optional[pd.DataFrame] = None,
        include_regime_features: bool = True,
        include_order_flow: bool = False
    ) -> pd.DataFrame:
        """
        Generate all analyst features from the feature bank.

        Args:
            data: OHLCV data with columns ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            agg_trades: Aggregated trades data (optional)
            futures_data: Futures data (optional)
            regime_data: Regime classification data (optional)
            include_regime_features: Whether to include regime-based features
            include_order_flow: Whether to include order flow features

        Returns:
            DataFrame with all generated analyst features
        """
        try:
            logger.info("🎯 Starting analyst feature generation...")

            # Initialize features DataFrame
            features_df = pd.DataFrame(index=data.index)

            # 1. Cross-timeframe momentum (CRITICAL)
            logger.info("📊 Generating cross-timeframe momentum features...")
            features_df.update(self._generate_cross_timeframe_momentum(data))

            # 2. Volatility structure
            logger.info("📊 Generating volatility structure features...")
            features_df.update(self._generate_volatility_structure(data, regime_data))

            # 3. Volume patterns
            logger.info("📊 Generating volume pattern features...")
            features_df.update(self._generate_volume_patterns(data))

            # 4. Microstructure
            logger.info("📊 Generating microstructure features...")
            features_df.update(self._generate_microstructure_features(data, agg_trades))

            # 5. Order flow (if available and requested)
            if include_order_flow and agg_trades is not None:
                logger.info("📊 Generating order flow features...")
                features_df.update(self._generate_order_flow_features(agg_trades))

            # 6. Regime features
            if include_regime_features and regime_data is not None:
                logger.info("📊 Generating regime features...")
                features_df.update(self._generate_regime_features(regime_data))

            # Fill any NaN values with 0 (neutral values)
            features_df = features_df.fillna(0.0)

            logger.info(f"✅ Generated {len(features_df.columns)} analyst features")
            return features_df

        except Exception as e:
            logger.error(f"❌ Error generating analyst features: {e}")
            # Return empty DataFrame as fallback
            return pd.DataFrame(index=data.index)

    def _generate_cross_timeframe_momentum(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate cross-timeframe momentum features."""
        features = {}

        try:
            # Calculate returns for different timeframes
            returns_5m = data['close'].pct_change(20)  # 20 periods for 5m timeframe
            returns_15m = data['close'].pct_change(60)  # 60 periods for 15m timeframe
            returns_1h = data['close'].pct_change(240)  # 240 periods for 1h timeframe

            # Cross-timeframe momentum features
            features['momentum_5m'] = returns_5m.rolling(20).mean()
            features['momentum_15m'] = returns_15m.rolling(20).mean()
            features['momentum_1h'] = returns_1h.rolling(20).mean()

            # Momentum alignment (all same sign)
            mom_5m = features['momentum_5m']
            mom_15m = features['momentum_15m']
            mom_1h = features['momentum_1h']

            # Use sign function to determine direction
            features['momentum_alignment'] = (
                (np.sign(mom_5m) == np.sign(mom_15m)) &
                (np.sign(mom_15m) == np.sign(mom_1h))
            ).astype(int)

            # Cross-timeframe momentum divergence
            features['momentum_divergence'] = (
                np.abs(mom_5m - mom_15m) +
                np.abs(mom_15m - mom_1h) +
                np.abs(mom_1h - mom_5m)
            ) / 3

            # Momentum acceleration (rate of change of momentum)
            features['momentum_acceleration_5m'] = mom_5m.diff()
            features['momentum_acceleration_15m'] = mom_15m.diff()
            features['momentum_acceleration_1h'] = mom_1h.diff()

        except Exception as e:
            logger.warning(f"Error generating cross-timeframe momentum: {e}")

        return pd.DataFrame(features, index=data.index)

    def _generate_volatility_structure(self, data: pd.DataFrame, regime_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Generate volatility structure features."""
        features = {}

        try:
            # Calculate volatility for different timeframes
            returns = data['close'].pct_change()

            # Rolling volatility windows
            vol_5m = returns.rolling(20).std()   # 20 periods ~5m volatility
            vol_15m = returns.rolling(60).std()  # 60 periods ~15m volatility
            vol_1h = returns.rolling(240).std()  # 240 periods ~1h volatility

            # Volatility ratios
            features['vol_ratio_5m_15m'] = vol_5m / vol_15m
            features['vol_ratio_15m_1h'] = vol_15m / vol_1h
            features['vol_ratio_5m_1h'] = vol_5m / vol_1h

            # Current volatility relative to regime average
            if regime_data is not None and 'regime' in regime_data.columns:
                current_vol = returns.rolling(20).std()

                # Calculate regime-specific volatility averages
                regime_vol_avgs = {}
                for regime in regime_data['regime'].unique():
                    regime_mask = regime_data['regime'] == regime
                    regime_vol_avgs[regime] = current_vol[regime_mask].mean()

                # Current regime deviation
                current_regime = regime_data['regime'].iloc[-1] if len(regime_data) > 0 else None
                if current_regime is not None and current_regime in regime_vol_avgs:
                    regime_avg_vol = regime_vol_avgs[current_regime]
                    features['vol_regime_deviation'] = current_vol.iloc[-1] / regime_avg_vol if regime_avg_vol > 0 else 1.0
                else:
                    features['vol_regime_deviation'] = 1.0

            # Volatility regime indicators
            features['volatility_regime'] = pd.qcut(
                returns.rolling(100).std(),
                q=3,
                labels=['low', 'medium', 'high']
            ).astype(str)

            # Volatility momentum (rate of change)
            features['volatility_momentum'] = returns.rolling(20).std().pct_change(20)

            # Volatility acceleration
            features['volatility_acceleration'] = features['volatility_momentum'].diff()

        except Exception as e:
            logger.warning(f"Error generating volatility structure: {e}")

        return pd.DataFrame(features, index=data.index)

    def _generate_volume_patterns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate volume pattern features."""
        features = {}

        try:
            volume = data['volume']

            # Volume pressure (buy vs sell pressure proxy)
            # Using price movement direction as proxy for buy/sell pressure
            price_change = data['close'].pct_change()
            volume_up = volume.where(price_change > 0, 0)
            volume_down = volume.where(price_change < 0, 0)

            features['volume_pressure'] = (volume_up - volume_down) / volume.replace(0, 1)

            # Volume trend using linear regression
            def volume_trend(x):
                if len(x) < 10:
                    return 0.0
                try:
                    slope, _, _, _, _ = linregress(range(len(x)), x.values)
                    return slope
                except:
                    return 0.0

            features['volume_trend'] = volume.rolling(20).apply(volume_trend)

            # Volume momentum
            features['volume_momentum'] = volume.pct_change(20)

            # Volume acceleration
            features['volume_acceleration'] = features['volume_momentum'].diff()

            # Volume relative to average
            volume_ma = volume.rolling(50).mean()
            features['volume_ratio'] = volume / volume_ma.replace(0, 1)

            # Volume volatility
            features['volume_volatility'] = volume.rolling(20).std() / volume_ma.replace(0, 1)

            # Volume concentration (high volume periods)
            volume_percentile = volume.rolling(100).rank(pct=True)
            features['volume_concentration'] = (volume_percentile > 0.8).astype(int)

        except Exception as e:
            logger.warning(f"Error generating volume patterns: {e}")

        return pd.DataFrame(features, index=data.index)

    def _generate_microstructure_features(self, data: pd.DataFrame, agg_trades: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Generate microstructure features."""
        features = {}

        try:
            # Spread calculation (using high-low as proxy)
            spread = (data['high'] - data['low']) / data['close']

            # ATR for normalization (using simplified ATR calculation)
            high_low = data['high'] - data['low']
            high_close = np.abs(data['high'] - data['close'].shift(1))
            low_close = np.abs(data['low'] - data['close'].shift(1))
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = true_range.rolling(14).mean()

            features['spread_normalized'] = spread / atr.replace(0, 1)

            # Tick imbalance (using price changes as proxy for upticks/downticks)
            price_changes = data['close'].diff()

            # Count upticks vs downticks in rolling window
            upticks = (price_changes > 0).rolling(100).sum()
            downticks = (price_changes < 0).rolling(100).sum()
            total_ticks = upticks + downticks

            features['tick_imbalance'] = (upticks - downticks) / total_ticks.replace(0, 1)

            # Price impact (sensitivity to volume)
            returns = data['close'].pct_change()
            features['price_impact'] = returns / volume.replace(0, 1)

            # Market efficiency (returns per unit of volatility)
            volatility = returns.rolling(20).std()
            features['market_efficiency'] = returns.abs() / volatility.replace(0, 1)

            # Microstructure noise (high-frequency price movements)
            features['microstructure_noise'] = data['close'].diff().abs().rolling(10).mean()

        except Exception as e:
            logger.warning(f"Error generating microstructure features: {e}")

        return pd.DataFrame(features, index=data.index)

    def _generate_order_flow_features(self, agg_trades: pd.DataFrame) -> pd.DataFrame:
        """Generate order flow features."""
        features = {}

        try:
            if 'buy_volume' in agg_trades.columns and 'sell_volume' in agg_trades.columns:
                buy_volume = agg_trades['buy_volume']
                sell_volume = agg_trades['sell_volume']
                total_volume = buy_volume + sell_volume

                features['bid_ask_imbalance'] = (buy_volume - sell_volume) / total_volume.replace(0, 1)

            # Market order flow (if available)
            if 'market_buys' in agg_trades.columns and 'market_sells' in agg_trades.columns:
                market_buys = agg_trades['market_buys']
                market_sells = agg_trades['market_sells']

                features['market_order_flow'] = market_buys - market_sells

                # Order flow momentum
                features['order_flow_momentum'] = features['market_order_flow'].pct_change(20)

            # Order book imbalance (if available)
            if 'bid_size' in agg_trades.columns and 'ask_size' in agg_trades.columns:
                bid_size = agg_trades['bid_size']
                ask_size = agg_trades['ask_size']

                features['order_book_imbalance'] = (bid_size - ask_size) / (bid_size + ask_size).replace(0, 1)

        except Exception as e:
            logger.warning(f"Error generating order flow features: {e}")

        return pd.DataFrame(features, index=agg_trades.index)

    def _generate_regime_features(self, regime_data: pd.DataFrame) -> pd.DataFrame:
        """Generate regime-based features."""
        features = {}

        try:
            if 'regime' in regime_data.columns:
                regime = regime_data['regime']

                # Regime probabilities (if available) or use current regime
                current_regime = regime.iloc[-1] if len(regime) > 0 else None

                # For now, use simple regime classification
                if current_regime == 'trending':
                    features['regime_prob_trending'] = 1.0
                    features['regime_prob_choppy'] = 0.0
                elif current_regime == 'choppy':
                    features['regime_prob_trending'] = 0.0
                    features['regime_prob_choppy'] = 1.0
                else:
                    features['regime_prob_trending'] = 0.5
                    features['regime_prob_choppy'] = 0.5

                # Regime stability (1 - entropy of regime distribution)
                regime_counts = regime.value_counts()
                total_regimes = len(regime_counts)
                if total_regimes > 0:
                    # Shannon entropy calculation
                    regime_probs = regime_counts / len(regime)
                    entropy = -np.sum(regime_probs * np.log2(regime_probs.replace(0, 1)))
                    max_entropy = np.log2(total_regimes) if total_regimes > 1 else 1
                    features['regime_stability'] = 1 - (entropy / max_entropy)
                else:
                    features['regime_stability'] = 0.5

                # Regime transition frequency
                regime_changes = regime != regime.shift(1)
                features['regime_transition_rate'] = regime_changes.rolling(50).mean()

                # Regime persistence (how long current regime has lasted)
                features['regime_persistence'] = (~regime_changes).cumsum()

                # Regime confidence (if available)
                if 'regime_confidence' in regime_data.columns:
                    features['regime_confidence'] = regime_data['regime_confidence']

        except Exception as e:
            logger.warning(f"Error generating regime features: {e}")

        return pd.DataFrame(features, index=regime_data.index)

    def get_analyst_features_dict(self) -> Dict[str, Any]:
        """
        Return the analyst_features dictionary as specified in the requirements.

        Returns:
            Dictionary with all analyst features as specified
        """
        # This would be a static definition of the features
        # For now, return a summary of available features
        return {
            'cross_timeframe_momentum': {
                'momentum_5m': 'returns.rolling(20).mean()',
                'momentum_15m': 'returns_15m.rolling(20).mean()',
                'momentum_1h': 'returns_1h.rolling(20).mean()',
                'momentum_alignment': 'sign(mom_5m) == sign(mom_15m) == sign(mom_1h)'
            },
            'volatility_structure': {
                'vol_ratio_5m_15m': 'vol_5m / vol_15m',
                'vol_regime_deviation': 'current_vol / regime_avg_vol'
            },
            'volume_patterns': {
                'volume_pressure': '(buy_volume - sell_volume) / total_volume',
                'volume_trend': 'volume.rolling(20).apply(lambda x: linregress(range(20), x).slope)'
            },
            'microstructure': {
                'spread_normalized': 'spread / atr',
                'tick_imbalance': '(upticks - downticks) / total_ticks'
            },
            'order_flow': {
                'bid_ask_imbalance': '(bid_size - ask_size) / (bid_size + ask_size)',
                'market_order_flow': 'market_buys - market_sells'
            },
            'regime_features': {
                'regime_prob_trending': 'probability of trending regime',
                'regime_prob_choppy': 'probability of choppy regime',
                'regime_stability': '1 - regime_entropy'
            }
        }

    def get_feature_names(self) -> List[str]:
        """Get list of all available feature names."""
        return [
            'momentum_5m', 'momentum_15m', 'momentum_1h', 'momentum_alignment',
            'momentum_divergence', 'momentum_acceleration_5m', 'momentum_acceleration_15m', 'momentum_acceleration_1h',
            'vol_ratio_5m_15m', 'vol_ratio_15m_1h', 'vol_ratio_5m_1h', 'vol_regime_deviation',
            'volatility_regime', 'volatility_momentum', 'volatility_acceleration',
            'volume_pressure', 'volume_trend', 'volume_momentum', 'volume_acceleration',
            'volume_ratio', 'volume_volatility', 'volume_concentration',
            'spread_normalized', 'tick_imbalance', 'price_impact', 'market_efficiency', 'microstructure_noise',
            'bid_ask_imbalance', 'market_order_flow', 'order_flow_momentum', 'order_book_imbalance',
            'regime_prob_trending', 'regime_prob_choppy', 'regime_stability',
            'regime_transition_rate', 'regime_persistence', 'regime_confidence'
        ]


# Global instance for convenience
_global_analyst_feature_bank: Optional[AnalystFeatureBank] = None


def get_analyst_feature_bank() -> AnalystFeatureBank:
    """
    Get the global analyst feature bank instance.

    Returns:
        Global analyst feature bank instance
    """
    global _global_analyst_feature_bank

    if _global_analyst_feature_bank is None:
        _global_analyst_feature_bank = AnalystFeatureBank()

    return _global_analyst_feature_bank


def set_analyst_feature_bank(bank: AnalystFeatureBank) -> None:
    """
    Set the global analyst feature bank instance.

    Args:
        bank: Analyst feature bank instance
    """
    global _global_analyst_feature_bank
    _global_analyst_feature_bank = bank