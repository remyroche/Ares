# src/training/steps/combined_fractional_system.py

"""Combined Fractional System: Integration of fractional labeling and fractional differentiation.
Designed to work with existing HMM regime system without redundant regime tuning.
"""

import asyncio
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.logger import get_logger
from src.utils.error_handler import handle_errors
from src.utils.centralized_decorators import (
    validate_data_quality,
    validate_feature_engineering_with_lookahead_bias_detection,
)

# Import fractional components
from src.training.steps.step04_analyst_labeling_feature_engineering_components.fractional_triple_barrier_labeling import (
    FractionalTripleBarrierLabeling
)
from src.training.steps.fractional_differentiation import FractionalFeatureGenerator


class HMMFractionalIntegration:
    """Integrate fractional systems with existing HMM regime system."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize HMM integration component.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.regime_metrics = {}  # Track performance per regime
        self.feature_enhancement = self.config.get('feature_enhancement', True)
        self.quality_tracking = self.config.get('quality_tracking', True)
        self.logger = get_logger("HMMFractionalIntegration")

    def enhance_features(self, features: pd.DataFrame, hmm_regime: Optional[str] = None) -> pd.DataFrame:
        """Enhance features with HMM regime information.

        Args:
            features: Input features DataFrame
            hmm_regime: HMM regime label (optional)

        Returns:
            Enhanced features DataFrame
        """
        enhanced_features = features.copy()

        if self.feature_enhancement and hmm_regime:
            try:
                # Add regime-specific feature quality metrics
                regime_quality = self.calculate_regime_quality(features, hmm_regime)
                regime_stability = self.calculate_regime_stability(features, hmm_regime)

                enhanced_features[f'regime_{hmm_regime}_quality'] = regime_quality
                enhanced_features[f'regime_{hmm_regime}_stability'] = regime_stability

                # Track regime metrics
                if self.quality_tracking:
                    self.regime_metrics[hmm_regime] = {
                        'quality': regime_quality,
                        'stability': regime_stability,
                        'feature_count': len(features.columns),
                        'timestamp': pd.Timestamp.now()
                    }

                self.logger.info(f"Enhanced features for regime {hmm_regime}: quality={regime_quality:.3f}, stability={regime_stability:.3f}")

            except Exception as e:
                self.logger.warning(f"Failed to enhance features for regime {hmm_regime}: {e}")

        return enhanced_features

    def calculate_regime_quality(self, features: pd.DataFrame, regime: str) -> float:
        """Calculate feature quality for specific HMM regime.

        Args:
            features: Features DataFrame
            regime: HMM regime label

        Returns:
            Quality score (0-1)
        """
        try:
            # Calculate various quality metrics
            variance_scores = []
            correlation_scores = []
            information_scores = []

            for col in features.columns:
                if col.startswith('regime_'):
                    continue  # Skip existing regime features

                feature_series = features[col].dropna()
                if len(feature_series) == 0:
                    continue

                # Variance score (good features have reasonable variance)
                variance = feature_series.var()
                if 0.001 <= variance <= 0.1:
                    variance_score = 1.0
                else:
                    variance_score = max(0.0, 1.0 - abs(variance - 0.05) / 0.05)
                variance_scores.append(variance_score)

                # Correlation score (avoid perfect correlations)
                if len(features.columns) > 1:
                    correlations = []
                    for other_col in features.columns:
                        if other_col != col and not other_col.startswith('regime_'):
                            corr = abs(feature_series.corr(features[other_col].dropna()))
                            correlations.append(corr)

                    if correlations:
                        avg_correlation = np.mean(correlations)
                        correlation_score = max(0.0, 1.0 - avg_correlation)
                        correlation_scores.append(correlation_score)

                # Information score (entropy-like measure)
                non_zero_ratio = (feature_series != 0).sum() / len(feature_series)
                unique_ratio = feature_series.nunique() / len(feature_series)
                information_score = non_zero_ratio * unique_ratio
                information_scores.append(information_score)

            # Combine scores
            if variance_scores:
                avg_variance = np.mean(variance_scores)
            else:
                avg_variance = 0.5

            if correlation_scores:
                avg_correlation = np.mean(correlation_scores)
            else:
                avg_correlation = 0.5

            if information_scores:
                avg_information = np.mean(information_scores)
            else:
                avg_information = 0.5

            # Weighted combination
            quality_score = 0.4 * avg_variance + 0.3 * avg_correlation + 0.3 * avg_information

            return min(1.0, max(0.0, quality_score))

        except Exception as e:
            self.logger.warning(f"Error calculating regime quality: {e}")
            return 0.5

    def calculate_regime_stability(self, features: pd.DataFrame, regime: str) -> float:
        """Calculate feature stability for specific HMM regime.

        Args:
            features: Features DataFrame
            regime: HMM regime label

        Returns:
            Stability score (0-1)
        """
        try:
            stability_scores = []

            for col in features.columns:
                if col.startswith('regime_'):
                    continue  # Skip existing regime features

                feature_series = features[col].dropna()
                if len(feature_series) < 50:
                    continue

                # Calculate rolling variance stability
                rolling_var = feature_series.rolling(window=min(50, len(feature_series)//4), min_periods=10).var()

                if rolling_var.mean() > 0:
                    var_consistency = 1.0 - (rolling_var.std() / rolling_var.mean())
                    stability_score = max(0.0, var_consistency)
                else:
                    stability_score = 0.5

                stability_scores.append(stability_score)

            if stability_scores:
                return np.mean(stability_scores)
            else:
                return 0.5

        except Exception as e:
            self.logger.warning(f"Error calculating regime stability: {e}")
            return 0.5


class CombinedFractionalSystem:
    """Unified system combining fractional labeling and differentiation.

    Designed to work with existing HMM regime system without redundant regime tuning.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize combined fractional system.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}

        # Initialize components
        self.fractional_labeler = FractionalTripleBarrierLabeling(
            fractional_config=self.config.get('labeling', {})
        )

        self.fractional_feature_generator = FractionalFeatureGenerator(
            config=self.config.get('differentiation', {})
        )

        self.hmm_integration = HMMFractionalIntegration(
            config=self.config.get('hmm_integration', {})
        )

        # Performance tracking
        self.performance_history = []
        self.logger = get_logger("CombinedFractionalSystem")

        self.logger.info("✅ Combined Fractional System initialized successfully")

    @handle_errors("Combined fractional system processing")
    @validate_data_quality
    @validate_feature_engineering_with_lookahead_bias_detection
    async def process_data(
        self,
        price_data: pd.DataFrame,
        volume_data: pd.DataFrame,
        hmm_regime: Optional[str] = None
    ) -> Dict[str, Any]:
        """Process data through combined fractional system.

        Args:
            price_data: OHLCV price data
            volume_data: Volume data
            hmm_regime: HMM regime label (optional)

        Returns:
            Dictionary containing processed features, labels, and metrics
        """
        start_time = time.time()

        try:
            self.logger.info(f"🚀 Starting combined fractional processing (regime: {hmm_regime})")

            # 1. Generate fractional differentiation features (Step 6)
            self.logger.info("📊 Generating fractional differentiation features...")
            features = self.fractional_feature_generator.generate_features(price_data, volume_data)

            # 2. Apply fractional labeling
            self.logger.info("🏷️ Applying fractional labeling...")
            labels = self.fractional_labeler.apply_fractional_triple_barrier_labeling(
                price_data, regime_labels=hmm_regime
            )

            # 3. Enhance features with HMM regime information
            self.logger.info("🔧 Enhancing features with HMM regime information...")
            enhanced_features = self.hmm_integration.enhance_features(features, hmm_regime)

            # 4. Calculate performance metrics
            performance_metrics = self._calculate_performance_metrics(
                enhanced_features, labels, hmm_regime, time.time() - start_time
            )

            # 5. Track performance history
            self.performance_history.append({
                'timestamp': pd.Timestamp.now(),
                'regime': hmm_regime,
                'metrics': performance_metrics,
                'feature_count': len(enhanced_features.columns),
                'sample_count': len(enhanced_features)
            })

            self.logger.info(f"✅ Combined processing complete: {len(enhanced_features.columns)} features, {len(enhanced_features)} samples")

            return {
                'features': enhanced_features,
                'labels': labels,
                'hmm_regime': hmm_regime,
                'performance_metrics': performance_metrics,
                'processing_time': time.time() - start_time
            }

        except Exception as e:
            self.logger.error(f"❌ Combined processing failed: {e}")
            raise

    def _calculate_performance_metrics(
        self,
        features: pd.DataFrame,
        labels: pd.DataFrame,
        hmm_regime: Optional[str],
        processing_time: float
    ) -> Dict[str, Any]:
        """Calculate performance metrics for the combined system.

        Args:
            features: Enhanced features DataFrame
            labels: Fractional labels DataFrame
            hmm_regime: HMM regime label
            processing_time: Processing time in seconds

        Returns:
            Dictionary of performance metrics
        """
        try:
            metrics = {
                'processing_time': processing_time,
                'feature_count': len(features.columns),
                'sample_count': len(features),
                'regime': hmm_regime
            }

            # Feature quality metrics
            if not features.empty:
                # Calculate feature quality
                feature_qualities = []
                for col in features.columns:
                    if col.startswith('regime_'):
                        continue

                    feature_series = features[col].dropna()
                    if len(feature_series) > 0:
                        variance = feature_series.var()
                        non_zero_ratio = (feature_series != 0).sum() / len(feature_series)
                        quality_score = min(1.0, variance * 100) * non_zero_ratio
                        feature_qualities.append(quality_score)

                if feature_qualities:
                    metrics['feature_quality'] = np.mean(feature_qualities)
                    metrics['feature_quality_std'] = np.std(feature_qualities)
                else:
                    metrics['feature_quality'] = 0.0
                    metrics['feature_quality_std'] = 0.0

            # Label quality metrics
            if not labels.empty and 'fractional_label' in labels.columns:
                label_series = labels['fractional_label'].dropna()
                if len(label_series) > 0:
                    metrics['label_variance'] = label_series.var()
                    metrics['label_range'] = label_series.max() - label_series.min()
                    metrics['label_mean'] = label_series.mean()

                    # Label distribution metrics
                    positive_labels = (label_series > 0).sum()
                    negative_labels = (label_series < 0).sum()
                    neutral_labels = (label_series == 0).sum()
                    total_labels = len(label_series)

                    metrics['label_distribution'] = {
                        'positive_ratio': positive_labels / total_labels,
                        'negative_ratio': negative_labels / total_labels,
                        'neutral_ratio': neutral_labels / total_labels
                    }

            # HMM regime metrics
            if hmm_regime:
                regime_metrics = self.hmm_integration.get_regime_metrics()
                if hmm_regime in regime_metrics:
                    metrics['regime_quality'] = regime_metrics[hmm_regime]['quality']
                    metrics['regime_stability'] = regime_metrics[hmm_regime]['stability']

            return metrics

        except Exception as e:
            self.logger.warning(f"Error calculating performance metrics: {e}")
            return {
                'processing_time': processing_time,
                'feature_count': len(features.columns),
                'sample_count': len(features),
                'regime': hmm_regime,
                'error': str(e)
            }


# Configuration helper