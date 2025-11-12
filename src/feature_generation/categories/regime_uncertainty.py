"""
Regime Uncertainty Features

This module generates features quantifying regime classification uncertainty:
- Classification entropy and confusion
- Ambiguity indices
- Certainty trends
- Decision boundary distances
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import entropy

from ..core.feature_generator import FeatureGenerator
from ..core.feature_config import FeatureConfig, FeatureCategory


@dataclass
class RegimeUncertaintyConfig:
    """Configuration for regime uncertainty features."""
    name: str = "regime_uncertainty"
    category: str = "REGIME_UNCERTAINTY"
    description: str = "Regime classification uncertainty and confidence metrics"

    # Thresholds
    ambiguity_threshold: float = 0.2  # Prob > this counts as ambiguous
    decision_boundary: float = 0.5    # Threshold for binary decisions

    # Windows
    trend_window: int = 10

    min_periods: int = 3


class RegimeUncertaintyGenerator(FeatureGenerator):
    """
    Generates regime uncertainty features.

    Features include:
    - Classification entropy
    - Confusion scores
    - Ambiguity indices
    - Certainty trends
    - Decision boundary distances
    """

    def __init__(self, config: Optional[RegimeUncertaintyConfig] = None):
        self.config = config or RegimeUncertaintyConfig()

        feature_config = FeatureConfig(
            name=self.config.name,
            category=FeatureCategory.REGIME,
            description=self.config.description,
            required_columns=["close"],
            default_lookback=self.config.trend_window,
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
        Generate regime uncertainty features.

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

        # 1. Uncertainty Metrics
        features.update(
            self._generate_uncertainty_metrics(regime_probabilities)
        )

        return features

    def _generate_uncertainty_metrics(
        self,
        regime_probabilities: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Generate uncertainty-related features."""
        features = {}
        n_samples = regime_probabilities.shape[0]

        # Classification entropy (Shannon entropy)
        classification_entropy = np.full(n_samples, np.nan)
        for i in range(n_samples):
            probs = regime_probabilities[i, :]
            # Add small epsilon to avoid log(0)
            classification_entropy[i] = entropy(probs + 1e-10)

        features['regime_classification_entropy'] = classification_entropy

        # Confusion score (1 - max_prob)
        max_probs = np.max(regime_probabilities, axis=1)
        confusion_score = 1.0 - max_probs
        features['regime_confusion_score'] = confusion_score

        # Ambiguity index (number of regimes with prob > threshold)
        ambiguity_index = np.sum(
            regime_probabilities > self.config.ambiguity_threshold,
            axis=1
        ).astype(float)
        features['regime_ambiguity_index'] = ambiguity_index

        # Certainty trend (change in entropy over time)
        certainty_trend = np.full(n_samples, np.nan)

        for i in range(self.config.trend_window, n_samples):
            window_entropy = classification_entropy[i - self.config.trend_window:i+1]
            if not np.all(np.isnan(window_entropy)):
                # Linear regression slope
                x = np.arange(len(window_entropy))
                valid = ~np.isnan(window_entropy)
                if np.sum(valid) >= 3:
                    slope = np.polyfit(x[valid], window_entropy[valid], 1)[0]
                    # Negative slope = increasing certainty (decreasing entropy)
                    certainty_trend[i] = -slope

        features['regime_certainty_trend'] = certainty_trend

        # Decision boundary distance
        # For each sample, compute distance from 50/50 decision boundary
        decision_boundary_dist = np.full(n_samples, np.nan)

        for i in range(n_samples):
            probs = regime_probabilities[i, :]
            sorted_probs = np.sort(probs)[::-1]

            if len(sorted_probs) >= 2:
                # Distance between top two probabilities
                # 0 = at boundary (50/50), 1 = far from boundary (100/0)
                top_prob = sorted_probs[0]
                second_prob = sorted_probs[1]

                # Distance from equal probability
                decision_boundary_dist[i] = abs(top_prob - second_prob)

        features['regime_decision_boundary_dist'] = decision_boundary_dist

        # Additional uncertainty metrics

        # Normalized entropy (entropy / max_entropy)
        n_regimes = regime_probabilities.shape[1]
        max_entropy = np.log(n_regimes) if n_regimes > 1 else 1.0
        normalized_entropy = classification_entropy / max_entropy
        features['regime_normalized_entropy'] = normalized_entropy

        # Confidence ratio (max_prob / second_max_prob)
        confidence_ratio = np.full(n_samples, np.nan)
        for i in range(n_samples):
            sorted_probs = np.sort(regime_probabilities[i, :])[::-1]
            if len(sorted_probs) >= 2 and sorted_probs[1] > 0:
                confidence_ratio[i] = sorted_probs[0] / sorted_probs[1]

        features['regime_confidence_ratio'] = confidence_ratio

        # Effective number of regimes (exp(entropy))
        effective_n_regimes = np.exp(classification_entropy)
        features['regime_effective_n_regimes'] = effective_n_regimes

        # Probability spread (std of probabilities)
        prob_spread = np.std(regime_probabilities, axis=1)
        features['regime_probability_spread'] = prob_spread

        # Dominant regime stability (how stable is the max probability)
        max_prob_volatility = self._rolling_std(max_probs, self.config.trend_window)
        features['regime_dominant_stability'] = max_prob_volatility

        # Uncertainty change rate (rate of change in entropy)
        uncertainty_change = np.full(n_samples, np.nan)
        uncertainty_change[1:] = np.diff(classification_entropy)
        features['regime_uncertainty_change_rate'] = uncertainty_change

        return features

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
            'regime_classification_entropy',
            'regime_confusion_score',
            'regime_ambiguity_index',
            'regime_certainty_trend',
            'regime_decision_boundary_dist',
            'regime_normalized_entropy',
            'regime_confidence_ratio',
            'regime_effective_n_regimes',
            'regime_probability_spread',
            'regime_dominant_stability',
            'regime_uncertainty_change_rate'
        ]

        return {name: np.full(n_samples, np.nan) for name in feature_names}


def create_regime_uncertainty_generators(
    config: Optional[RegimeUncertaintyConfig] = None
) -> List[FeatureGenerator]:
    """
    Factory function to create regime uncertainty feature generators.

    Args:
        config: Configuration for the generators

    Returns:
        List of feature generators
    """
    return [RegimeUncertaintyGenerator(config)]
