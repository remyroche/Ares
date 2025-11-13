"""
Regime Transition Features

This module generates features related to regime transitions, including:
- Transition probabilities from HMM models
- Historical transition patterns
- Transition dynamics and stability metrics
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import entropy

from ..core.feature_generator import FeatureGenerator, FeatureConfig, FeatureCategory


@dataclass
class RegimeTransitionConfig:
    """Configuration for regime transition features."""
    name: str = "regime_transitions"
    category: str = "REGIME_TRANSITIONS"
    description: str = "Regime transition probability and pattern features"

    # Lookback windows for different metrics
    short_window: int = 5
    medium_window: int = 10
    long_window: int = 20

    # Minimum periods required
    min_periods: int = 5


class RegimeTransitionGenerator(FeatureGenerator):
    """
    Generates regime transition features from HMM outputs.

    Features include:
    - Transition probabilities to each regime
    - Self-transition (persistence) probability
    - Transition entropy and uncertainty
    - Historical switch counts and rates
    - Transition volatility and trends
    """

    def __init__(self, config: Optional[RegimeTransitionConfig] = None):
        self.config = config or RegimeTransitionConfig()

        feature_config = FeatureConfig(
            name=self.config.name,
            category=FeatureCategory.REGIME,
            description=self.config.description,
            required_columns=["close"],  # Base requirement
            default_lookback=self.config.long_window,
            min_lookback=self.config.min_periods,
            max_lookback=100
        )
        super().__init__(feature_config)

    def generate_features(
        self,
        data: pd.DataFrame,
        regime_labels: Optional[pd.Series] = None,
        regime_probabilities: Optional[np.ndarray] = None,
        transition_matrix: Optional[np.ndarray] = None,
        **kwargs
    ) -> Dict[str, np.ndarray]:
        """
        Generate regime transition features.

        Args:
            data: Market data DataFrame
            regime_labels: Regime labels (0, 1, 2, ...)
            regime_probabilities: HMM regime probabilities (n_samples, n_regimes)
            transition_matrix: HMM transition matrix (n_regimes, n_regimes)

        Returns:
            Dictionary of feature name -> feature array
        """
        features = {}
        n_samples = len(data)

        # Validate inputs
        if regime_labels is None or regime_probabilities is None:
            # Return empty features if regime data not available
            return self._generate_empty_features(n_samples)

        # Ensure regime_labels is aligned with data
        if len(regime_labels) != n_samples:
            regime_labels = regime_labels.reindex(data.index, method='ffill')

        # Get number of regimes
        n_regimes = regime_probabilities.shape[1] if regime_probabilities.ndim > 1 else 0

        if n_regimes == 0:
            return self._generate_empty_features(n_samples)

        # 1. Transition Probability Features (from transition matrix)
        if transition_matrix is not None:
            features.update(
                self._generate_transition_probability_features(
                    regime_labels, transition_matrix, n_regimes
                )
            )

        # 2. Historical Transition Patterns
        features.update(
            self._generate_transition_pattern_features(regime_labels)
        )

        # 3. Transition Dynamics
        features.update(
            self._generate_transition_dynamics_features(
                regime_labels, regime_probabilities
            )
        )

        return features

    def _generate_transition_probability_features(
        self,
        regime_labels: pd.Series,
        transition_matrix: np.ndarray,
        n_regimes: int
    ) -> Dict[str, np.ndarray]:
        """Generate features from HMM transition matrix."""
        features = {}
        n_samples = len(regime_labels)

        # Initialize arrays
        regime_array = regime_labels.values

        # For each sample, get transition probabilities from current regime
        for target_regime in range(n_regimes):
            trans_probs = np.full(n_samples, np.nan)
            for i in range(n_samples):
                if not np.isnan(regime_array[i]):
                    current_regime = int(regime_array[i])
                    if current_regime < len(transition_matrix):
                        trans_probs[i] = transition_matrix[current_regime, target_regime]

            features[f'regime_transition_prob_to_{target_regime}'] = trans_probs

        # Self-transition probability (stay in current regime)
        self_trans_prob = np.full(n_samples, np.nan)
        for i in range(n_samples):
            if not np.isnan(regime_array[i]):
                current_regime = int(regime_array[i])
                if current_regime < len(transition_matrix):
                    self_trans_prob[i] = transition_matrix[current_regime, current_regime]

        features['regime_self_transition_prob'] = self_trans_prob

        # Max transition probability (highest likelihood of moving to any regime)
        max_trans_prob = np.full(n_samples, np.nan)
        for i in range(n_samples):
            if not np.isnan(regime_array[i]):
                current_regime = int(regime_array[i])
                if current_regime < len(transition_matrix):
                    max_trans_prob[i] = np.max(transition_matrix[current_regime, :])

        features['regime_max_transition_prob'] = max_trans_prob

        # Transition entropy (uncertainty in next regime)
        trans_entropy = np.full(n_samples, np.nan)
        for i in range(n_samples):
            if not np.isnan(regime_array[i]):
                current_regime = int(regime_array[i])
                if current_regime < len(transition_matrix):
                    probs = transition_matrix[current_regime, :]
                    # Add small epsilon to avoid log(0)
                    trans_entropy[i] = entropy(probs + 1e-10)

        features['regime_transition_entropy'] = trans_entropy

        return features

    def _generate_transition_pattern_features(
        self,
        regime_labels: pd.Series
    ) -> Dict[str, np.ndarray]:
        """Generate features from historical transition patterns."""
        features = {}
        n_samples = len(regime_labels)

        regime_array = regime_labels.values

        # Detect regime switches (1 where regime changed, 0 otherwise)
        regime_switches = np.zeros(n_samples)
        for i in range(1, n_samples):
            if not np.isnan(regime_array[i]) and not np.isnan(regime_array[i-1]):
                if regime_array[i] != regime_array[i-1]:
                    regime_switches[i] = 1

        # Switch counts over different windows
        for window in [self.config.short_window, self.config.medium_window, self.config.long_window]:
            switch_count = self._rolling_sum(regime_switches, window)
            features[f'regime_switch_count_{window}'] = switch_count

            # Switch rate (normalized by window size)
            features[f'regime_switch_rate_{window}'] = switch_count / window

        # Switch acceleration (change in switch rate)
        switch_rate_short = features[f'regime_switch_rate_{self.config.short_window}']
        switch_rate_medium = features[f'regime_switch_rate_{self.config.medium_window}']
        features['regime_switch_acceleration'] = switch_rate_short - switch_rate_medium

        # Last switch distance (periods since last regime change)
        last_switch_dist = np.full(n_samples, np.nan)
        periods_since_switch = 0
        for i in range(n_samples):
            if regime_switches[i] == 1:
                periods_since_switch = 0
            else:
                periods_since_switch += 1
            last_switch_dist[i] = periods_since_switch

        features['regime_last_switch_distance'] = last_switch_dist

        return features

    def _generate_transition_dynamics_features(
        self,
        regime_labels: pd.Series,
        regime_probabilities: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Generate dynamic transition features."""
        features = {}
        n_samples = len(regime_labels)

        regime_array = regime_labels.values

        # Get current regime probability (probability of being in assigned regime)
        current_regime_prob = np.full(n_samples, np.nan)
        for i in range(n_samples):
            if not np.isnan(regime_array[i]):
                current_regime = int(regime_array[i])
                if current_regime < regime_probabilities.shape[1]:
                    current_regime_prob[i] = regime_probabilities[i, current_regime]

        # Transition volatility (std of regime probabilities over time)
        trans_volatility = self._rolling_std(current_regime_prob, self.config.short_window)
        features['regime_transition_volatility_5'] = trans_volatility

        # Transition trend (increasing/decreasing likelihood)
        trans_trend = np.full(n_samples, np.nan)
        for i in range(self.config.short_window, n_samples):
            window_probs = current_regime_prob[i-self.config.short_window:i]
            if not np.all(np.isnan(window_probs)):
                # Linear regression slope
                x = np.arange(len(window_probs))
                valid = ~np.isnan(window_probs)
                if np.sum(valid) >= 3:
                    slope = np.polyfit(x[valid], window_probs[valid], 1)[0]
                    trans_trend[i] = slope

        features['regime_transition_trend'] = trans_trend

        # Boundary proximity (how close to regime boundary, 0-1)
        # 0 = far from boundary (high confidence), 1 = at boundary (uncertain)
        boundary_proximity = 1.0 - current_regime_prob
        features['regime_boundary_proximity'] = boundary_proximity

        # Stability score (inverse of entropy)
        stability = current_regime_prob  # High prob = high stability
        features['regime_stability_score'] = stability

        # Flip-flop count (A→B→A pattern)
        flip_flop_count = np.zeros(n_samples)
        for i in range(2, n_samples):
            if (not np.isnan(regime_array[i]) and
                not np.isnan(regime_array[i-1]) and
                not np.isnan(regime_array[i-2])):
                if regime_array[i] == regime_array[i-2] and regime_array[i] != regime_array[i-1]:
                    flip_flop_count[i] = 1

        flip_flop_rolling = self._rolling_sum(flip_flop_count, self.config.medium_window)
        features['regime_flip_flop_count_10'] = flip_flop_rolling

        # Directional bias (tendency to move to higher/lower regime IDs)
        directional_bias = np.full(n_samples, np.nan)
        for i in range(1, n_samples):
            if not np.isnan(regime_array[i]) and not np.isnan(regime_array[i-1]):
                directional_bias[i] = regime_array[i] - regime_array[i-1]

        # Rolling average of directional bias
        directional_bias_avg = self._rolling_mean(directional_bias, self.config.medium_window)
        features['regime_directional_bias'] = directional_bias_avg

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
        """Generate empty feature arrays when regime data is not available."""
        features = {}

        # Transition probability features
        feature_names = [
            'regime_transition_prob_to_0',
            'regime_transition_prob_to_1',
            'regime_self_transition_prob',
            'regime_max_transition_prob',
            'regime_transition_entropy',
            'regime_switch_count_5',
            'regime_switch_count_10',
            'regime_switch_count_20',
            'regime_switch_rate_5',
            'regime_switch_rate_10',
            'regime_switch_rate_20',
            'regime_switch_acceleration',
            'regime_last_switch_distance',
            'regime_transition_volatility_5',
            'regime_transition_trend',
            'regime_boundary_proximity',
            'regime_stability_score',
            'regime_flip_flop_count_10',
            'regime_directional_bias'
        ]

        for name in feature_names:
            features[name] = np.full(n_samples, np.nan)

        return features

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Return a representative single feature as a Series for FeatureBank integration."""
        all_features = self.generate_features(data, **kwargs)
        preferred_name = 'regime_transition_volatility_5'
        arr = all_features.get(preferred_name)
        if arr is None and all_features:
            preferred_name, arr = next(iter(all_features.items()))
        if isinstance(arr, pd.Series):
            return arr.rename(preferred_name)
        return pd.Series(arr if arr is not None else np.full(len(data), np.nan), index=data.index, name=preferred_name)


def create_regime_transition_generators(
    config: Optional[RegimeTransitionConfig] = None
) -> List[FeatureGenerator]:
    """
    Factory function to create regime transition feature generators.

    Args:
        config: Configuration for the generators

    Returns:
        List of feature generators
    """
    return [RegimeTransitionGenerator(config)]
