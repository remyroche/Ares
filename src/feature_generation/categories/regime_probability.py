"""
Regime Probability Features

This module generates features directly from HMM regime probabilities:
- Individual regime probabilities
- Probability gaps and confidence metrics
- Probability dynamics (trends, volatility)
- Probability patterns and crossovers
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..core.feature_generator import FeatureGenerator
from ..core.feature_config import FeatureConfig, FeatureCategory


@dataclass
class RegimeProbabilityConfig:
    """Configuration for regime probability features."""
    name: str = "regime_probability"
    category: str = "REGIME_PROBABILITY"
    description: str = "Features derived from HMM regime probabilities"

    # Windows for dynamics
    short_window: int = 5
    medium_window: int = 10

    # Threshold for probability crossovers
    crossover_threshold: float = 0.2

    min_periods: int = 3


class RegimeProbabilityGenerator(FeatureGenerator):
    """
    Generates features from HMM regime probabilities.

    Features include:
    - Individual regime probabilities
    - Max, second max, and probability gaps
    - Probability concentration metrics
    - Probability trends and volatility
    - Momentum and acceleration
    - Divergence and crossover patterns
    """

    def __init__(self, config: Optional[RegimeProbabilityConfig] = None):
        self.config = config or RegimeProbabilityConfig()

        feature_config = FeatureConfig(
            name=self.config.name,
            category=FeatureCategory.REGIME,
            description=self.config.description,
            required_columns=["close"],
            default_lookback=self.config.medium_window,
            min_lookback=self.config.min_periods,
            max_lookback=50
        )
        super().__init__(feature_config)

    def generate_features(
        self,
        data: pd.DataFrame,
        regime_probabilities: Optional[np.ndarray] = None,
        **kwargs
    ) -> Dict[str, np.ndarray]:
        """
        Generate regime probability features.

        Args:
            data: Market data DataFrame
            regime_probabilities: HMM regime probabilities (n_samples, n_regimes)

        Returns:
            Dictionary of feature name -> feature array
        """
        features = {}
        n_samples = len(data)

        if regime_probabilities is None:
            return self._generate_empty_features(n_samples)

        # Validate shape
        if regime_probabilities.ndim != 2 or regime_probabilities.shape[0] != n_samples:
            return self._generate_empty_features(n_samples)

        n_regimes = regime_probabilities.shape[1]

        if n_regimes == 0:
            return self._generate_empty_features(n_samples)

        # 1. Direct Probability Features
        features.update(
            self._generate_direct_probability_features(regime_probabilities, n_regimes)
        )

        # 2. Probability Dynamics
        features.update(
            self._generate_probability_dynamics(regime_probabilities, features)
        )

        # 3. Probability Patterns
        features.update(
            self._generate_probability_patterns(regime_probabilities, features)
        )

        return features

    def _generate_direct_probability_features(
        self,
        regime_probabilities: np.ndarray,
        n_regimes: int
    ) -> Dict[str, np.ndarray]:
        """Generate direct features from regime probabilities."""
        features = {}
        n_samples = regime_probabilities.shape[0]

        # Individual regime probabilities
        for regime_idx in range(n_regimes):
            features[f'regime_prob_{regime_idx}'] = regime_probabilities[:, regime_idx]

        # Max probability (most likely regime)
        max_prob = np.max(regime_probabilities, axis=1)
        features['regime_prob_max'] = max_prob

        # Second max probability
        second_max_prob = np.full(n_samples, np.nan)
        for i in range(n_samples):
            sorted_probs = np.sort(regime_probabilities[i, :])
            if len(sorted_probs) >= 2:
                second_max_prob[i] = sorted_probs[-2]

        features['regime_prob_second_max'] = second_max_prob

        # Probability gap (confidence measure)
        prob_gap = max_prob - second_max_prob
        features['regime_prob_gap'] = prob_gap

        return features

    def _generate_probability_dynamics(
        self,
        regime_probabilities: np.ndarray,
        direct_features: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Generate dynamic probability features."""
        features = {}
        n_samples = regime_probabilities.shape[0]

        max_prob = direct_features['regime_prob_max']

        # Probability entropy (uncertainty across all regimes)
        prob_entropy = np.full(n_samples, np.nan)
        for i in range(n_samples):
            probs = regime_probabilities[i, :]
            # Add small epsilon to avoid log(0)
            prob_entropy[i] = -np.sum(probs * np.log(probs + 1e-10))

        features['regime_prob_entropy'] = prob_entropy

        # Probability concentration (Gini coefficient)
        prob_concentration = np.full(n_samples, np.nan)
        for i in range(n_samples):
            probs = np.sort(regime_probabilities[i, :])
            n = len(probs)
            if n > 1:
                index = np.arange(1, n + 1)
                gini = (2 * np.sum(index * probs)) / (n * np.sum(probs)) - (n + 1) / n
                prob_concentration[i] = gini

        features['regime_prob_concentration'] = prob_concentration

        # Probability trend (change in max prob over window)
        prob_trend = np.full(n_samples, np.nan)
        for i in range(self.config.short_window, n_samples):
            window_probs = max_prob[i - self.config.short_window:i+1]
            if not np.all(np.isnan(window_probs)):
                # Linear regression slope
                x = np.arange(len(window_probs))
                valid = ~np.isnan(window_probs)
                if np.sum(valid) >= 3:
                    slope = np.polyfit(x[valid], window_probs[valid], 1)[0]
                    prob_trend[i] = slope

        features['regime_prob_trend_5'] = prob_trend

        # Probability volatility (std of max prob)
        prob_volatility = self._rolling_std(max_prob, self.config.short_window)
        features['regime_prob_volatility_5'] = prob_volatility

        # Probability acceleration (change in trend)
        prob_acceleration = np.full(n_samples, np.nan)
        prob_acceleration[1:] = np.diff(prob_trend)
        features['regime_prob_acceleration'] = prob_acceleration

        # Probability momentum (rate of change)
        prob_momentum = np.full(n_samples, np.nan)
        for i in range(self.config.short_window, n_samples):
            if not np.isnan(max_prob[i]) and not np.isnan(max_prob[i - self.config.short_window]):
                prob_momentum[i] = max_prob[i] - max_prob[i - self.config.short_window]

        features['regime_prob_momentum'] = prob_momentum

        return features

    def _generate_probability_patterns(
        self,
        regime_probabilities: np.ndarray,
        all_features: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Generate probability pattern features."""
        features = {}
        n_samples = regime_probabilities.shape[0]

        max_prob = all_features['regime_prob_max']

        # Probability divergence (current vs smoothed)
        smoothed_prob = self._rolling_mean(max_prob, self.config.short_window)
        prob_divergence = max_prob - smoothed_prob
        features['regime_prob_divergence'] = prob_divergence

        # Probability crossover count (regime ranking changes)
        crossover_count = np.zeros(n_samples)

        for i in range(1, n_samples):
            # Get regime ranks (which regime has highest prob)
            prev_ranks = np.argsort(regime_probabilities[i-1, :])[::-1]
            curr_ranks = np.argsort(regime_probabilities[i, :])[::-1]

            # Count rank changes in top regimes
            if prev_ranks[0] != curr_ranks[0]:
                crossover_count[i] = 1

        # Rolling sum of crossovers
        crossover_rolling = self._rolling_sum(crossover_count, self.config.short_window)
        features['regime_prob_crossover_count_5'] = crossover_rolling

        # Probability stability score (inverse of volatility)
        prob_volatility = all_features['regime_prob_volatility_5']
        prob_stability = np.full(n_samples, np.nan)
        for i in range(n_samples):
            if not np.isnan(prob_volatility[i]) and prob_volatility[i] > 0:
                prob_stability[i] = 1.0 / (prob_volatility[i] + 1e-6)

        features['regime_prob_stability_score'] = prob_stability

        # Confidence trend (increasing/decreasing certainty)
        prob_entropy = all_features['regime_prob_entropy']
        confidence_trend = np.full(n_samples, np.nan)

        for i in range(self.config.short_window, n_samples):
            window_entropy = prob_entropy[i - self.config.short_window:i+1]
            if not np.all(np.isnan(window_entropy)):
                x = np.arange(len(window_entropy))
                valid = ~np.isnan(window_entropy)
                if np.sum(valid) >= 3:
                    slope = np.polyfit(x[valid], window_entropy[valid], 1)[0]
                    # Negative slope = increasing confidence (decreasing entropy)
                    confidence_trend[i] = -slope

        features['regime_prob_confidence_trend'] = confidence_trend

        return features

    def _rolling_sum(self, arr: np.ndarray, window: int) -> np.ndarray:
        """Fast rolling sum with NaN handling."""
        result = np.full_like(arr, np.nan, dtype=float)
        for i in range(window - 1, len(arr)):
            window_data = arr[max(0, i - window + 1):i + 1]
            if not np.all(np.isnan(window_data)):
                result[i] = np.nansum(window_data)
        return result

    def _rolling_mean(self, arr: np.ndarray, window: int) -> np.ndarray:
        """Fast rolling mean with NaN handling."""
        result = np.full_like(arr, np.nan, dtype=float)
        for i in range(window - 1, len(arr)):
            window_data = arr[max(0, i - window + 1):i + 1]
            if not np.all(np.isnan(window_data)):
                result[i] = np.nanmean(window_data)
        return result

    def _rolling_std(self, arr: np.ndarray, window: int) -> np.ndarray:
        """Fast rolling std with NaN handling."""
        result = np.full_like(arr, np.nan, dtype=float)
        for i in range(window - 1, len(arr)):
            window_data = arr[max(0, i - window + 1):i + 1]
            valid_data = window_data[~np.isnan(window_data)]
            if len(valid_data) >= 2:
                result[i] = np.std(valid_data)
        return result

    def _generate_empty_features(self, n_samples: int) -> Dict[str, np.ndarray]:
        """Generate empty feature arrays when probability data is not available."""
        feature_names = [
            'regime_prob_0',
            'regime_prob_1',
            'regime_prob_max',
            'regime_prob_second_max',
            'regime_prob_gap',
            'regime_prob_entropy',
            'regime_prob_concentration',
            'regime_prob_trend_5',
            'regime_prob_volatility_5',
            'regime_prob_acceleration',
            'regime_prob_momentum',
            'regime_prob_divergence',
            'regime_prob_crossover_count_5',
            'regime_prob_stability_score',
            'regime_prob_confidence_trend'
        ]

        return {name: np.full(n_samples, np.nan) for name in feature_names}


def create_regime_probability_generators(
    config: Optional[RegimeProbabilityConfig] = None
) -> List[FeatureGenerator]:
    """
    Factory function to create regime probability feature generators.

    Args:
        config: Configuration for the generators

    Returns:
        List of feature generators
    """
    return [RegimeProbabilityGenerator(config)]
