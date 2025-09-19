"""
HMM Performance Metrics Feature Generator

This module creates features from HMM performance metrics that can be used in ML models
for regime prediction, model quality assessment, and meta-learning applications.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union, Tuple
import logging

from ..core.feature_generator import (
    FeatureGenerator, 
    FeatureConfig, 
    FeatureCategory,
    VectorizedFeatureGenerator
)

logger = logging.getLogger(__name__)


class HMMPerformanceMetricsFeatureGenerator(VectorizedFeatureGenerator):
    """
    Feature generator that converts HMM performance metrics into ML-ready features.
    
    This generator takes HMM performance metrics and creates features that can be used
    for downstream ML models, including regime prediction, model quality assessment,
    and ensemble weighting.
    """
    
    def __init__(self, lookback_window: int = 20):
        """
        Initialize HMM Performance Metrics Feature Generator.
        
        Args:
            lookback_window: Window size for rolling metrics calculations
        """
        config = FeatureConfig(
            name="hmm_performance_metrics",
            category=FeatureCategory.HMM_REGIME,
            description="Features derived from HMM model performance metrics",
            required_columns=["close"],  # Minimal requirement
            optional_columns=["high", "low", "volume", "regime_labels", "regime_probabilities"],
            default_lookback=lookback_window,
            min_lookback=5,
            max_lookback=100
        )
        super().__init__(config)
        self.lookback_window = lookback_window
        self._cached_metrics = {}
        
    def generate_features(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Generate features from HMM performance metrics.
        
        Args:
            data: Market data DataFrame
            **kwargs: Additional parameters including:
                - hmm_performance_metrics: Dict of HMM performance metrics
                - regime_labels: Array of regime labels
                - regime_probabilities: Array of regime probabilities
                
        Returns:
            DataFrame with HMM performance-based features
        """
        try:
            # Extract HMM metrics from kwargs
            hmm_metrics = kwargs.get('hmm_performance_metrics', {})
            regime_labels = kwargs.get('regime_labels', None)
            regime_probabilities = kwargs.get('regime_probabilities', None)
            
            if not hmm_metrics:
                logger.warning("No HMM performance metrics provided, generating basic features")
                return self._generate_basic_features(data, regime_labels, regime_probabilities)
            
            # Generate comprehensive features from metrics
            features = self._generate_metrics_features(data, hmm_metrics, regime_labels, regime_probabilities)
            
            return features
            
        except Exception as e:
            logger.error(f"Failed to generate HMM performance metrics features: {e}")
            return pd.DataFrame(index=data.index)
    
    def _generate_metrics_features(
        self, 
        data: pd.DataFrame, 
        hmm_metrics: Dict[str, float],
        regime_labels: Optional[np.ndarray] = None,
        regime_probabilities: Optional[np.ndarray] = None
    ) -> pd.DataFrame:
        """Generate features from HMM performance metrics."""
        
        features = pd.DataFrame(index=data.index)
        
        # 1. Static Metrics Features (broadcast to all rows)
        static_metrics = [
            'regime_stability', 'regime_balance', 'regime_entropy', 'regime_gini_coefficient',
            'avg_confidence', 'min_confidence', 'max_confidence', 'confidence_std',
            'avg_regime_duration', 'min_regime_duration', 'max_regime_duration',
            'regime_duration_std', 'regime_duration_cv', 'avg_regime_persistence',
            'min_regime_persistence', 'max_regime_persistence', 'n_regimes_detected',
            'regime_coverage', 'avg_uncertainty', 'uncertainty_std',
            'regime_separation_ratio', 'avg_regime_distance', 'min_regime_distance',
            'max_regime_distance', 'transition_rate'
        ]
        
        for metric in static_metrics:
            if metric in hmm_metrics:
                features[f'hmm_{metric}'] = hmm_metrics[metric]
        
        # 2. Dynamic Features from Regime Labels and Probabilities
        if regime_labels is not None and len(regime_labels) == len(data):
            features = pd.concat([features, self._generate_regime_based_features(
                data, regime_labels, regime_probabilities
            )], axis=1)
        
        # 3. Rolling Metrics Features
        if regime_labels is not None and regime_probabilities is not None:
            features = pd.concat([features, self._generate_rolling_metrics_features(
                data, regime_labels, regime_probabilities
            )], axis=1)
        
        # 4. Interaction Features
        features = pd.concat([features, self._generate_interaction_features(
            features, hmm_metrics
        )], axis=1)
        
        return features
    
    def _generate_regime_based_features(
        self, 
        data: pd.DataFrame, 
        regime_labels: np.ndarray,
        regime_probabilities: Optional[np.ndarray] = None
    ) -> pd.DataFrame:
        """Generate features based on regime labels and probabilities."""
        
        features = pd.DataFrame(index=data.index)
        
        # Current regime
        features['hmm_current_regime'] = regime_labels
        
        # Regime changes
        regime_changes = np.diff(regime_labels, prepend=regime_labels[0])
        features['hmm_regime_changed'] = (regime_changes != 0).astype(int)
        
        # Time since regime change
        time_since_change = np.zeros_like(regime_labels)
        current_duration = 0
        for i in range(len(regime_labels)):
            if i == 0 or regime_changes[i] != 0:
                current_duration = 1
            else:
                current_duration += 1
            time_since_change[i] = current_duration
        
        features['hmm_time_since_regime_change'] = time_since_change
        
        # Regime persistence (how long current regime has lasted)
        features['hmm_regime_persistence'] = time_since_change
        
        if regime_probabilities is not None:
            # Current regime probability
            max_probs = np.max(regime_probabilities, axis=1)
            features['hmm_regime_confidence'] = max_probs
            
            # Uncertainty (1 - max probability)
            features['hmm_regime_uncertainty'] = 1 - max_probs
            
            # Probability spread (difference between top 2 probabilities)
            if regime_probabilities.shape[1] > 1:
                sorted_probs = np.sort(regime_probabilities, axis=1)
                prob_spread = sorted_probs[:, -1] - sorted_probs[:, -2]
                features['hmm_probability_spread'] = prob_spread
            
            # Individual regime probabilities
            for i in range(regime_probabilities.shape[1]):
                features[f'hmm_regime_{i}_prob'] = regime_probabilities[:, i]
        
        return features
    
    def _generate_rolling_metrics_features(
        self, 
        data: pd.DataFrame, 
        regime_labels: np.ndarray,
        regime_probabilities: np.ndarray
    ) -> pd.DataFrame:
        """Generate rolling window metrics features."""
        
        features = pd.DataFrame(index=data.index)
        
        # Convert to pandas Series for rolling operations
        regime_series = pd.Series(regime_labels, index=data.index)
        confidence_series = pd.Series(np.max(regime_probabilities, axis=1), index=data.index)
        
        # Rolling regime stability
        def rolling_stability(window_labels):
            if len(window_labels) <= 1:
                return 1.0
            changes = np.sum(np.diff(window_labels) != 0)
            return 1 - (changes / (len(window_labels) - 1))
        
        features['hmm_rolling_stability'] = regime_series.rolling(
            window=self.lookback_window, min_periods=2
        ).apply(rolling_stability)
        
        # Rolling confidence statistics
        features['hmm_rolling_avg_confidence'] = confidence_series.rolling(
            window=self.lookback_window, min_periods=1
        ).mean()
        
        features['hmm_rolling_confidence_std'] = confidence_series.rolling(
            window=self.lookback_window, min_periods=2
        ).std().fillna(0)
        
        # Rolling regime diversity (number of unique regimes in window)
        features['hmm_rolling_regime_diversity'] = regime_series.rolling(
            window=self.lookback_window, min_periods=1
        ).apply(lambda x: len(np.unique(x)))
        
        # Rolling transition rate
        def rolling_transition_rate(window_labels):
            if len(window_labels) <= 1:
                return 0.0
            changes = np.sum(np.diff(window_labels) != 0)
            return changes / len(window_labels)
        
        features['hmm_rolling_transition_rate'] = regime_series.rolling(
            window=self.lookback_window, min_periods=2
        ).apply(rolling_transition_rate)
        
        return features
    
    def _generate_interaction_features(
        self, 
        features: pd.DataFrame, 
        hmm_metrics: Dict[str, float]
    ) -> pd.DataFrame:
        """Generate interaction features between metrics and market data."""
        
        interaction_features = pd.DataFrame(index=features.index)
        
        # Interaction between confidence and stability
        if 'hmm_regime_confidence' in features.columns and 'regime_stability' in hmm_metrics:
            interaction_features['hmm_confidence_stability_product'] = (
                features['hmm_regime_confidence'] * hmm_metrics['regime_stability']
            )
        
        # Regime quality score (combination of multiple metrics)
        quality_components = []
        
        if 'regime_stability' in hmm_metrics:
            quality_components.append(hmm_metrics['regime_stability'])
        if 'regime_balance' in hmm_metrics:
            quality_components.append(hmm_metrics['regime_balance'])
        if 'avg_confidence' in hmm_metrics:
            quality_components.append(hmm_metrics['avg_confidence'])
        
        if quality_components:
            regime_quality = np.mean(quality_components)
            interaction_features['hmm_regime_quality_score'] = regime_quality
        
        # Model reliability indicator
        reliability_components = []
        if 'regime_coverage' in hmm_metrics:
            reliability_components.append(hmm_metrics['regime_coverage'])
        if 'regime_separation_ratio' in hmm_metrics:
            reliability_components.append(hmm_metrics['regime_separation_ratio'])
        
        if reliability_components:
            model_reliability = np.mean(reliability_components)
            interaction_features['hmm_model_reliability'] = model_reliability
        
        return interaction_features
    
    def _generate_basic_features(
        self, 
        data: pd.DataFrame,
        regime_labels: Optional[np.ndarray] = None,
        regime_probabilities: Optional[np.ndarray] = None
    ) -> pd.DataFrame:
        """Generate basic features when no HMM metrics are available."""
        
        features = pd.DataFrame(index=data.index)
        
        if regime_labels is not None:
            features['hmm_current_regime'] = regime_labels
            
            # Basic regime change detection
            regime_changes = np.diff(regime_labels, prepend=regime_labels[0])
            features['hmm_regime_changed'] = (regime_changes != 0).astype(int)
        
        if regime_probabilities is not None:
            max_probs = np.max(regime_probabilities, axis=1)
            features['hmm_regime_confidence'] = max_probs
            features['hmm_regime_uncertainty'] = 1 - max_probs
        
        # Default values for missing metrics
        features['hmm_regime_stability'] = 0.5  # Neutral value
        features['hmm_regime_balance'] = 0.5    # Neutral value
        features['hmm_model_reliability'] = 0.5  # Neutral value
        
        return features
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names that this generator produces."""
        base_features = [
            'hmm_regime_stability', 'hmm_regime_balance', 'hmm_regime_entropy',
            'hmm_regime_gini_coefficient', 'hmm_avg_confidence', 'hmm_min_confidence',
            'hmm_max_confidence', 'hmm_confidence_std', 'hmm_avg_regime_duration',
            'hmm_min_regime_duration', 'hmm_max_regime_duration', 'hmm_regime_duration_std',
            'hmm_regime_duration_cv', 'hmm_avg_regime_persistence', 'hmm_min_regime_persistence',
            'hmm_max_regime_persistence', 'hmm_n_regimes_detected', 'hmm_regime_coverage',
            'hmm_avg_uncertainty', 'hmm_uncertainty_std', 'hmm_regime_separation_ratio',
            'hmm_avg_regime_distance', 'hmm_min_regime_distance', 'hmm_max_regime_distance',
            'hmm_transition_rate', 'hmm_current_regime', 'hmm_regime_changed',
            'hmm_time_since_regime_change', 'hmm_regime_persistence', 'hmm_regime_confidence',
            'hmm_regime_uncertainty', 'hmm_probability_spread', 'hmm_rolling_stability',
            'hmm_rolling_avg_confidence', 'hmm_rolling_confidence_std', 'hmm_rolling_regime_diversity',
            'hmm_rolling_transition_rate', 'hmm_confidence_stability_product',
            'hmm_regime_quality_score', 'hmm_model_reliability'
        ]
        
        return base_features


def create_hmm_performance_features_from_result(
    data: pd.DataFrame, 
    hmm_result: Any,
    lookback_window: int = 20
) -> pd.DataFrame:
    """
    Convenience function to create HMM performance features from HMM clustering result.
    
    Args:
        data: Market data DataFrame
        hmm_result: HMMClusteringResult object
        lookback_window: Window size for rolling calculations
        
    Returns:
        DataFrame with HMM performance-based features
    """
    generator = HMMPerformanceMetricsFeatureGenerator(lookback_window=lookback_window)
    
    features = generator.generate_features(
        data,
        hmm_performance_metrics=hmm_result.performance_metrics,
        regime_labels=hmm_result.regime_labels,
        regime_probabilities=hmm_result.regime_probabilities
    )
    
    return features


def integrate_hmm_metrics_with_features(
    base_features: pd.DataFrame,
    hmm_metrics: Dict[str, float],
    regime_labels: Optional[np.ndarray] = None,
    regime_probabilities: Optional[np.ndarray] = None,
    lookback_window: int = 20
) -> pd.DataFrame:
    """
    Integrate HMM performance metrics with existing feature DataFrame.
    
    Args:
        base_features: Existing features DataFrame
        hmm_metrics: HMM performance metrics dictionary
        regime_labels: Array of regime labels
        regime_probabilities: Array of regime probabilities
        lookback_window: Window size for rolling calculations
        
    Returns:
        Combined DataFrame with base features and HMM performance features
    """
    generator = HMMPerformanceMetricsFeatureGenerator(lookback_window=lookback_window)
    
    hmm_features = generator.generate_features(
        base_features,
        hmm_performance_metrics=hmm_metrics,
        regime_labels=regime_labels,
        regime_probabilities=regime_probabilities
    )
    
    # Combine features
    combined_features = pd.concat([base_features, hmm_features], axis=1)
    
    # Remove duplicate columns if any
    combined_features = combined_features.loc[:, ~combined_features.columns.duplicated()]
    
    return combined_features