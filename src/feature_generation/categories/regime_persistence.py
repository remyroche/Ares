"""
Regime Persistence Features

This module generates features related to regime duration and persistence:
- Duration metrics (current, previous, ratios)
- Persistence scores and survival probabilities
- Exhaustion and premature indicators
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..core.feature_generator import FeatureGenerator
from ..core.feature_config import FeatureConfig, FeatureCategory


@dataclass
class RegimePersistenceConfig:
    """Configuration for regime persistence features."""
    name: str = "regime_persistence"
    category: str = "REGIME_PERSISTENCE"
    description: str = "Regime duration and persistence metrics"

    # Windows for duration statistics
    duration_history_window: int = 10
    percentile_history_window: int = 50

    # Thresholds for exhaustion/premature detection
    exhaustion_threshold: float = 2.0  # duration > mean * threshold
    premature_threshold: float = 0.5   # duration < mean * threshold

    min_periods: int = 5


class RegimePersistenceGenerator(FeatureGenerator):
    """
    Generates regime persistence and duration features.

    Features include:
    - Current and previous regime durations
    - Duration ratios and percentiles
    - Average/max/min durations over history
    - Persistence scores and survival probabilities
    - Exhaustion and premature indicators
    - Regime age metrics
    """

    def __init__(self, config: Optional[RegimePersistenceConfig] = None):
        self.config = config or RegimePersistenceConfig()

        feature_config = FeatureConfig(
            name=self.config.name,
            category=FeatureCategory.REGIME,
            description=self.config.description,
            required_columns=["close"],
            default_lookback=self.config.percentile_history_window,
            min_lookback=self.config.min_periods,
            max_lookback=200
        )
        super().__init__(feature_config)

    def generate_features(
        self,
        data: pd.DataFrame,
        regime_labels: Optional[pd.Series] = None,
        **kwargs
    ) -> Dict[str, np.ndarray]:
        """
        Generate regime persistence features.

        Args:
            data: Market data DataFrame
            regime_labels: Regime labels (0, 1, 2, ...)

        Returns:
            Dictionary of feature name -> feature array
        """
        features = {}
        n_samples = len(data)

        if regime_labels is None:
            return self._generate_empty_features(n_samples)

        # Ensure regime_labels is aligned with data
        if len(regime_labels) != n_samples:
            regime_labels = regime_labels.reindex(data.index, method='ffill')

        regime_array = regime_labels.values

        # 1. Duration Features
        features.update(self._generate_duration_features(regime_array))

        # 2. Persistence Metrics
        features.update(self._generate_persistence_metrics(regime_array, features))

        return features

    def _generate_duration_features(self, regime_array: np.ndarray) -> Dict[str, np.ndarray]:
        """Generate regime duration features."""
        features = {}
        n_samples = len(regime_array)

        # Track current regime duration
        current_duration = np.zeros(n_samples)
        duration = 0
        for i in range(n_samples):
            if i == 0 or np.isnan(regime_array[i]):
                duration = 1
            elif regime_array[i] == regime_array[i-1]:
                duration += 1
            else:
                duration = 1
            current_duration[i] = duration

        features['regime_duration_current'] = current_duration

        # Previous regime duration
        previous_duration = np.full(n_samples, np.nan)
        last_regime_duration = np.nan
        for i in range(1, n_samples):
            if not np.isnan(regime_array[i]) and not np.isnan(regime_array[i-1]):
                if regime_array[i] != regime_array[i-1]:
                    # Regime just switched
                    last_regime_duration = current_duration[i-1]
            previous_duration[i] = last_regime_duration

        features['regime_duration_previous'] = previous_duration

        # Duration ratio (current / previous)
        duration_ratio = np.full(n_samples, np.nan)
        for i in range(n_samples):
            if not np.isnan(previous_duration[i]) and previous_duration[i] > 0:
                duration_ratio[i] = current_duration[i] / previous_duration[i]

        features['regime_duration_ratio'] = duration_ratio

        # Duration percentile (current duration vs historical distribution)
        duration_percentile = np.full(n_samples, np.nan)
        for i in range(self.config.percentile_history_window, n_samples):
            window_durations = current_duration[max(0, i - self.config.percentile_history_window):i]
            if len(window_durations) > 0:
                # Compute percentile rank
                current_dur = current_duration[i]
                percentile = np.sum(window_durations <= current_dur) / len(window_durations)
                duration_percentile[i] = percentile

        features['regime_duration_percentile'] = duration_percentile

        # Collect regime end durations for statistics
        regime_end_durations = self._extract_regime_durations(regime_array)

        # Average duration over last N regimes
        avg_duration = np.full(n_samples, np.nan)
        max_duration = np.full(n_samples, np.nan)
        min_duration = np.full(n_samples, np.nan)

        for i in range(n_samples):
            # Get completed regime durations up to this point
            completed_durations = [d for d in regime_end_durations if d['end_idx'] < i]

            if len(completed_durations) > 0:
                recent_durs = [d['duration'] for d in completed_durations[-self.config.duration_history_window:]]

                if len(recent_durs) >= self.config.min_periods:
                    avg_duration[i] = np.mean(recent_durs)
                    max_duration[i] = np.max(recent_durs)
                    min_duration[i] = np.min(recent_durs)

        features['regime_avg_duration_5'] = avg_duration
        features['regime_max_duration_10'] = max_duration
        features['regime_min_duration_10'] = min_duration

        return features

    def _generate_persistence_metrics(
        self,
        regime_array: np.ndarray,
        duration_features: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Generate persistence-related metrics."""
        features = {}
        n_samples = len(regime_array)

        current_duration = duration_features['regime_duration_current']
        avg_duration = duration_features['regime_avg_duration_5']

        # Persistence score (current / expected duration)
        persistence_score = np.full(n_samples, np.nan)
        for i in range(n_samples):
            if not np.isnan(avg_duration[i]) and avg_duration[i] > 0:
                persistence_score[i] = current_duration[i] / avg_duration[i]

        features['regime_persistence_score'] = persistence_score

        # Exhaustion indicator (duration >> average)
        exhaustion = np.zeros(n_samples)
        for i in range(n_samples):
            if not np.isnan(avg_duration[i]):
                if current_duration[i] > avg_duration[i] * self.config.exhaustion_threshold:
                    exhaustion[i] = 1

        features['regime_exhaustion_indicator'] = exhaustion

        # Premature indicator (duration << average)
        premature = np.zeros(n_samples)
        for i in range(n_samples):
            if not np.isnan(avg_duration[i]) and avg_duration[i] > 0:
                if current_duration[i] < avg_duration[i] * self.config.premature_threshold:
                    premature[i] = 1

        features['regime_premature_indicator'] = premature

        # Half-life (expected remaining duration based on exponential decay)
        # Simplified: half_life = avg_duration - current_duration
        half_life = np.full(n_samples, np.nan)
        for i in range(n_samples):
            if not np.isnan(avg_duration[i]):
                remaining = max(0, avg_duration[i] - current_duration[i])
                half_life[i] = remaining

        features['regime_half_life'] = half_life

        # Survival probability (probability regime continues next period)
        # Using exponential survival: P(T > t) = exp(-t / mean_duration)
        survival_prob = np.full(n_samples, np.nan)
        for i in range(n_samples):
            if not np.isnan(avg_duration[i]) and avg_duration[i] > 0:
                t = current_duration[i]
                mean_dur = avg_duration[i]
                # Probability of surviving one more period given current duration
                survival_prob[i] = np.exp(-(t + 1) / mean_dur) / np.exp(-t / mean_dur)

        features['regime_survival_probability'] = survival_prob

        # Age normalized (current duration / max observed duration)
        max_duration = duration_features['regime_max_duration_10']
        age_normalized = np.full(n_samples, np.nan)
        for i in range(n_samples):
            if not np.isnan(max_duration[i]) and max_duration[i] > 0:
                age_normalized[i] = current_duration[i] / max_duration[i]

        features['regime_age_normalized'] = age_normalized

        return features

    def _extract_regime_durations(self, regime_array: np.ndarray) -> List[Dict]:
        """
        Extract durations of completed regimes.

        Returns:
            List of dicts with 'regime', 'duration', 'start_idx', 'end_idx'
        """
        durations = []
        current_regime = None
        start_idx = 0
        duration = 0

        for i in range(len(regime_array)):
            if np.isnan(regime_array[i]):
                continue

            regime = int(regime_array[i])

            if current_regime is None:
                # First regime
                current_regime = regime
                start_idx = i
                duration = 1
            elif regime == current_regime:
                # Same regime continues
                duration += 1
            else:
                # Regime changed - save previous regime duration
                durations.append({
                    'regime': current_regime,
                    'duration': duration,
                    'start_idx': start_idx,
                    'end_idx': i - 1
                })
                # Start new regime
                current_regime = regime
                start_idx = i
                duration = 1

        # Add final regime (ongoing)
        if current_regime is not None:
            durations.append({
                'regime': current_regime,
                'duration': duration,
                'start_idx': start_idx,
                'end_idx': len(regime_array) - 1
            })

        return durations

    def _generate_empty_features(self, n_samples: int) -> Dict[str, np.ndarray]:
        """Generate empty feature arrays when regime data is not available."""
        feature_names = [
            'regime_duration_current',
            'regime_duration_previous',
            'regime_duration_ratio',
            'regime_duration_percentile',
            'regime_avg_duration_5',
            'regime_max_duration_10',
            'regime_min_duration_10',
            'regime_persistence_score',
            'regime_exhaustion_indicator',
            'regime_premature_indicator',
            'regime_half_life',
            'regime_survival_probability',
            'regime_age_normalized'
        ]

        return {name: np.full(n_samples, np.nan) for name in feature_names}


def create_regime_persistence_generators(
    config: Optional[RegimePersistenceConfig] = None
) -> List[FeatureGenerator]:
    """
    Factory function to create regime persistence feature generators.

    Args:
        config: Configuration for the generators

    Returns:
        List of feature generators
    """
    return [RegimePersistenceGenerator(config)]
