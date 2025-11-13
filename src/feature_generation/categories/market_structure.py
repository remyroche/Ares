"""
Market Structure Features

This module generates regime-specific market structure features:
- Support and resistance levels within regimes
- Swing structure (higher highs, lower lows)
- Price efficiency and complexity
- Volume patterns within regimes
- Fractal dimension
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..core.feature_generator import FeatureGenerator, FeatureConfig, FeatureCategory


@dataclass
class MarketStructureConfig:
    """Configuration for market structure features."""
    name: str = "market_structure"
    category: str = "MARKET_STRUCTURE"
    description: str = "Regime-specific market structure and price patterns"

    # Windows for structure analysis
    swing_window: int = 5
    structure_window: int = 20
    fractal_window: int = 10

    # Thresholds
    swing_threshold: float = 0.001  # Minimum price change to count as swing

    min_periods: int = 5


class MarketStructureGenerator(FeatureGenerator):
    """
    Generates market structure features within regime contexts.

    Features include:
    - Higher highs and lower lows counts
    - Swing structure indicators
    - Fractal dimension
    - Price efficiency metrics
    - Volume patterns within regimes
    """

    def __init__(self, config: Optional[MarketStructureConfig] = None):
        self.config = config or MarketStructureConfig()

        feature_config = FeatureConfig(
            name=self.config.name,
            category=FeatureCategory.REGIME,
            description=self.config.description,
            required_columns=["high", "low", "close", "volume"],
            default_lookback=self.config.structure_window,
            min_lookback=self.config.min_periods,
            max_lookback=100
        )
        super().__init__(feature_config)

    def generate_features(
        self,
        data: pd.DataFrame,
        regime_labels: Optional[pd.Series] = None,
        **kwargs
    ) -> Dict[str, np.ndarray]:
        """
        Generate market structure features.

        Args:
            data: Market data DataFrame with high, low, close, volume
            regime_labels: Regime labels (0, 1, 2, ...)

        Returns:
            Dictionary of feature name -> feature array
        """
        features = {}
        n_samples = len(data)

        # Validate required columns
        required_cols = ['high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in data.columns:
                return self._generate_empty_features(n_samples)

        if regime_labels is None:
            return self._generate_empty_features(n_samples)

        # Ensure regime_labels is aligned with data
        if len(regime_labels) != n_samples:
            regime_labels = regime_labels.reindex(data.index, method='ffill')

        # 1. Price Structure Features
        features.update(self._generate_price_structure_features(data, regime_labels))

        # 2. Volume Structure Features
        features.update(self._generate_volume_structure_features(data, regime_labels))

        return features

    def _generate_price_structure_features(
        self,
        data: pd.DataFrame,
        regime_labels: pd.Series
    ) -> Dict[str, np.ndarray]:
        """Generate price structure features within regimes."""
        features = {}
        n_samples = len(data)

        high = data['high'].values
        low = data['low'].values
        close = data['close'].values
        regime_array = regime_labels.values

        # Identify swing highs and lows
        swing_highs = self._identify_swing_points(high, mode='high')
        swing_lows = self._identify_swing_points(low, mode='low')

        # Higher highs and lower lows within current regime
        higher_highs_count = np.zeros(n_samples)
        lower_lows_count = np.zeros(n_samples)

        for i in range(self.config.swing_window, n_samples):
            if np.isnan(regime_array[i]):
                continue

            # Find regime start
            regime_start = i
            for j in range(i - 1, max(0, i - 100), -1):
                if np.isnan(regime_array[j]) or regime_array[j] != regime_array[i]:
                    regime_start = j + 1
                    break

            # Count higher highs in current regime
            regime_swing_highs = [idx for idx in swing_highs if regime_start <= idx <= i]
            hh_count = 0
            for k in range(1, len(regime_swing_highs)):
                if high[regime_swing_highs[k]] > high[regime_swing_highs[k-1]]:
                    hh_count += 1
            higher_highs_count[i] = hh_count

            # Count lower lows in current regime
            regime_swing_lows = [idx for idx in swing_lows if regime_start <= idx <= i]
            ll_count = 0
            for k in range(1, len(regime_swing_lows)):
                if low[regime_swing_lows[k]] < low[regime_swing_lows[k-1]]:
                    ll_count += 1
            lower_lows_count[i] = ll_count

        features['higher_highs_count_regime'] = higher_highs_count
        features['lower_lows_count_regime'] = lower_lows_count

        # Swing structure (1 = uptrend, -1 = downtrend, 0 = ranging)
        swing_structure = np.zeros(n_samples)
        for i in range(n_samples):
            hh = higher_highs_count[i]
            ll = lower_lows_count[i]

            if hh > ll and hh >= 2:
                swing_structure[i] = 1  # Uptrend
            elif ll > hh and ll >= 2:
                swing_structure[i] = -1  # Downtrend
            # else: 0 (ranging)

        features['swing_structure_regime'] = swing_structure

        # Fractal dimension (measure of complexity)
        fractal_dim = self._calculate_fractal_dimension(close, self.config.fractal_window)
        features['fractal_dimension_regime'] = fractal_dim

        # Price range normalized by ATR within regime
        atr = self._calculate_atr(data, window=14)
        price_range_normalized = np.full(n_samples, np.nan)

        for i in range(n_samples):
            if np.isnan(regime_array[i]) or atr[i] == 0 or np.isnan(atr[i]):
                continue

            # Find regime start
            regime_start = i
            for j in range(i - 1, max(0, i - 100), -1):
                if np.isnan(regime_array[j]) or regime_array[j] != regime_array[i]:
                    regime_start = j + 1
                    break

            # Calculate range within regime
            regime_high = np.max(high[regime_start:i+1])
            regime_low = np.min(low[regime_start:i+1])
            price_range = regime_high - regime_low

            price_range_normalized[i] = price_range / atr[i]

        features['price_range_regime_normalized'] = price_range_normalized

        # Price efficiency (net displacement / total path length)
        price_efficiency = np.full(n_samples, np.nan)

        for i in range(n_samples):
            if np.isnan(regime_array[i]):
                continue

            # Find regime start
            regime_start = i
            for j in range(i - 1, max(0, i - 100), -1):
                if np.isnan(regime_array[j]) or regime_array[j] != regime_array[i]:
                    regime_start = j + 1
                    break

            if i - regime_start < 2:
                continue

            # Net displacement
            net_displacement = abs(close[i] - close[regime_start])

            # Total path length (sum of absolute price changes)
            total_path = np.sum(np.abs(np.diff(close[regime_start:i+1])))

            if total_path > 0:
                price_efficiency[i] = net_displacement / total_path

        features['price_efficiency_regime'] = price_efficiency

        return features

    def _generate_volume_structure_features(
        self,
        data: pd.DataFrame,
        regime_labels: pd.Series
    ) -> Dict[str, np.ndarray]:
        """Generate volume structure features within regimes."""
        features = {}
        n_samples = len(data)

        volume = data['volume'].values
        regime_array = regime_labels.values

        # Volume profile std within regime
        volume_std = np.full(n_samples, np.nan)
        volume_trend = np.full(n_samples, np.nan)
        volume_spike_count = np.zeros(n_samples)

        for i in range(n_samples):
            if np.isnan(regime_array[i]):
                continue

            # Find regime start
            regime_start = i
            for j in range(i - 1, max(0, i - 100), -1):
                if np.isnan(regime_array[j]) or regime_array[j] != regime_array[i]:
                    regime_start = j + 1
                    break

            regime_volume = volume[regime_start:i+1]

            if len(regime_volume) >= self.config.min_periods:
                # Volume volatility
                volume_std[i] = np.std(regime_volume)

                # Volume trend (linear regression slope)
                x = np.arange(len(regime_volume))
                if len(x) >= 2:
                    slope = np.polyfit(x, regime_volume, 1)[0]
                    volume_trend[i] = slope

                # Volume spikes (count of volumes > 2 std above mean)
                vol_mean = np.mean(regime_volume)
                vol_std_val = np.std(regime_volume)
                if vol_std_val > 0:
                    spike_threshold = vol_mean + 2 * vol_std_val
                    volume_spike_count[i] = np.sum(regime_volume > spike_threshold)

        features['volume_profile_regime_std'] = volume_std
        features['volume_trend_regime'] = volume_trend
        features['volume_spike_count_regime'] = volume_spike_count

        # Volume exhaustion (declining volume near end of regime)
        volume_exhaustion = np.zeros(n_samples)

        for i in range(self.config.swing_window, n_samples):
            if np.isnan(regime_array[i]):
                continue

            # Recent volume trend
            recent_volume = volume[i - self.config.swing_window:i+1]
            if len(recent_volume) >= 3:
                x = np.arange(len(recent_volume))
                slope = np.polyfit(x, recent_volume, 1)[0]

                # Negative slope indicates exhaustion
                if slope < 0:
                    volume_exhaustion[i] = 1

        features['volume_exhaustion_regime'] = volume_exhaustion

        return features

    def _identify_swing_points(self, prices: np.ndarray, mode: str = 'high') -> List[int]:
        """
        Identify swing highs or lows.

        Args:
            prices: Price array
            mode: 'high' for swing highs, 'low' for swing lows

        Returns:
            List of indices where swing points occur
        """
        swing_points = []
        window = self.config.swing_window

        for i in range(window, len(prices) - window):
            window_prices = prices[i - window:i + window + 1]

            if mode == 'high':
                if prices[i] == np.max(window_prices):
                    swing_points.append(i)
            else:  # mode == 'low'
                if prices[i] == np.min(window_prices):
                    swing_points.append(i)

        return swing_points

    def _calculate_fractal_dimension(self, prices: np.ndarray, window: int) -> np.ndarray:
        """
        Calculate Higuchi fractal dimension.

        Higher values indicate more complex, random behavior.
        Lower values indicate smoother, trending behavior.
        """
        n_samples = len(prices)
        fractal_dim = np.full(n_samples, np.nan)

        for i in range(window, n_samples):
            window_prices = prices[i - window:i+1]

            if np.all(np.isnan(window_prices)):
                continue

            # Simplified fractal dimension using box-counting
            # Normalize prices to [0, 1]
            p_min = np.nanmin(window_prices)
            p_max = np.nanmax(window_prices)

            if p_max - p_min == 0:
                fractal_dim[i] = 1.0
                continue

            normalized = (window_prices - p_min) / (p_max - p_min)

            # Calculate path length at different scales
            scales = [1, 2, 4]
            lengths = []

            for scale in scales:
                # Downsample
                downsampled = normalized[::scale]
                if len(downsampled) < 2:
                    continue

                # Calculate path length
                path_length = np.sum(np.abs(np.diff(downsampled)))
                lengths.append(path_length)

            if len(lengths) >= 2:
                # Estimate dimension from slope
                log_scales = np.log(scales[:len(lengths)])
                log_lengths = np.log(np.array(lengths) + 1e-10)

                # Dimension is approximately 2 - slope
                slope = np.polyfit(log_scales, log_lengths, 1)[0]
                fractal_dim[i] = 2.0 - slope

                # Clamp to reasonable range [1, 2]
                fractal_dim[i] = np.clip(fractal_dim[i], 1.0, 2.0)

        return fractal_dim

    def _calculate_atr(self, data: pd.DataFrame, window: int = 14) -> np.ndarray:
        """Calculate Average True Range."""
        high = data['high'].values
        low = data['low'].values
        close = data['close'].values

        # True range
        tr = np.maximum(
            high - low,
            np.maximum(
                np.abs(high - np.roll(close, 1)),
                np.abs(low - np.roll(close, 1))
            )
        )
        tr[0] = high[0] - low[0]  # First value

        # ATR as rolling mean of TR
        atr = np.full(len(tr), np.nan)
        for i in range(window - 1, len(tr)):
            atr[i] = np.mean(tr[max(0, i - window + 1):i + 1])

        return atr

    def _generate_empty_features(self, n_samples: int) -> Dict[str, np.ndarray]:
        """Generate empty feature arrays when data is not available."""
        feature_names = [
            'higher_highs_count_regime',
            'lower_lows_count_regime',
            'swing_structure_regime',
            'fractal_dimension_regime',
            'price_range_regime_normalized',
            'price_efficiency_regime',
            'volume_profile_regime_std',
            'volume_trend_regime',
            'volume_spike_count_regime',
            'volume_exhaustion_regime'
        ]

        return {name: np.full(n_samples, np.nan) for name in feature_names}

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Return a representative single feature as a Series for FeatureBank integration."""
        all_features = self.generate_features(data, **kwargs)
        preferred_name = 'fractal_dimension_regime'
        arr = all_features.get(preferred_name)
        if arr is None and all_features:
            # Fallback to first available feature
            preferred_name, arr = next(iter(all_features.items()))
        if isinstance(arr, pd.Series):
            return arr.rename(preferred_name)
        return pd.Series(arr if arr is not None else np.full(len(data), np.nan), index=data.index, name=preferred_name)


def create_market_structure_generators(
    config: Optional[MarketStructureConfig] = None
) -> List[FeatureGenerator]:
    """
    Factory function to create market structure feature generators.

    Args:
        config: Configuration for the generators

    Returns:
        List of feature generators
    """
    return [MarketStructureGenerator(config)]
