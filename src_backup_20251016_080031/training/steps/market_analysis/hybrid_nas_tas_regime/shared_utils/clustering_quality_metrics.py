"""
Clustering Quality Metrics for Regime Detection

This module provides comprehensive clustering quality evaluation metrics
for assessing the quality of regime detection in market analysis.

Metrics included:
- Silhouette Score
- Calinski-Harabasz Score
- Davies-Bouldin Index
- Gap Statistic
- Additional supporting metrics
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
import warnings
from dataclasses import dataclass
from src.utils.tprint import (
    tprint, tprint_debug, tprint_info, tprint_warning, tprint_error, 
    tprint_success, tprint_progress, tprint_performance, tprint_timer
)
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


@dataclass
class ClusteringQualityConfig:
    """Configuration for clustering quality metrics."""

    # Core metrics to compute
    compute_silhouette: bool = True
    compute_calinski_harabasz: bool = True
    compute_davies_bouldin: bool = True
    compute_gap_statistic: bool = True

    # Gap statistic parameters
    gap_n_clusters_range: Tuple[int, int] = (2, 15)
    gap_n_bootstraps: int = 10
    gap_random_state: int = 42

    # Additional metrics
    compute_inertia: bool = True
    compute_separation_index: bool = True
    compute_cohesion_index: bool = True

    # Feature preprocessing
    enable_feature_scaling: bool = True
    feature_scaling_method: str = "standard"  # "standard", "robust", "minmax"

    # Validation options
    validate_input_data: bool = True
    handle_outliers: bool = True
    outlier_threshold: float = 3.0  # Standard deviations


@dataclass
class ClusteringQualityResult:
    """Result from clustering quality evaluation."""

    # Core clustering metrics
    silhouette_score: float = 0.0
    calinski_harabasz_score: float = 0.0
    davies_bouldin_index: float = 0.0
    gap_statistic: Dict[str, Any] = None

    # Additional metrics
    inertia: float = 0.0
    separation_index: float = 0.0
    cohesion_index: float = 0.0

    # Gap statistic details
    gap_optimal_clusters: int = 0
    gap_scores: List[float] = None
    gap_std_errors: List[float] = None

    # Economic significance aspects
    economic_significance_scores: Dict[str, float] = None
    regime_economic_profiles: Dict[str, Dict[str, Any]] = None
    trading_viability_scores: Dict[str, float] = None
    risk_adjusted_returns: Dict[str, float] = None

    # Metadata
    n_samples: int = 0
    n_features: int = 0
    n_clusters: int = 0
    evaluation_timestamp: str = ""
    computation_time: float = 0.0

    # Quality assessment
    overall_quality_score: float = 0.0
    quality_interpretation: str = ""

    # Economic quality assessment
    economic_quality_score: float = 0.0
    economic_interpretation: str = ""
    economically_significant_regimes: int = 0
    economically_viable_regimes: int = 0


class ClusteringQualityMetrics:
    """
    Comprehensive clustering quality evaluation for regime detection.

    This class provides multiple clustering quality metrics to assess
    the effectiveness of regime clustering in market analysis.
    """

    def __init__(self, config: ClusteringQualityConfig = None):
        """Initialize clustering quality metrics evaluator."""
        self.config = config or ClusteringQualityConfig()
        self.logger = logging.getLogger(self.__class__.__name__)

    def evaluate_clustering_quality(self,
                                 features: Union[pd.DataFrame, np.ndarray],
                                 cluster_labels: np.ndarray,
                                 cluster_centers: Optional[np.ndarray] = None,
                                 timestamps: Optional[np.ndarray] = None,
                                 market_data: Optional[np.ndarray] = None) -> ClusteringQualityResult:
        """
        Evaluate comprehensive clustering quality metrics.

        Args:
            features: Feature matrix (n_samples, n_features)
            cluster_labels: Cluster assignments for each sample
            cluster_centers: Optional cluster centers

        Returns:
            ClusteringQualityResult with comprehensive metrics
        """
        import time
        start_time = time.time()

        try:
            # Validate and preprocess input data
            features_clean, labels_clean = self._validate_and_preprocess_data(features, cluster_labels)

            # Calculate core metrics
            metrics_result = ClusteringQualityResult()

            # Basic dataset information
            metrics_result.n_samples = len(features_clean)
            metrics_result.n_features = features_clean.shape[1] if len(features_clean.shape) > 1 else 1
            metrics_result.n_clusters = len(np.unique(labels_clean))

            # Silhouette Score
            if self.config.compute_silhouette:
                metrics_result.silhouette_score = self._calculate_silhouette_score(features_clean, labels_clean)

            # Calinski-Harabasz Score
            if self.config.compute_calinski_harabasz:
                metrics_result.calinski_harabasz_score = self._calculate_calinski_harabasz_score(features_clean, labels_clean)

            # Davies-Bouldin Index
            if self.config.compute_davies_bouldin:
                metrics_result.davies_bouldin_index = self._calculate_davies_bouldin_index(features_clean, labels_clean)

            # Gap Statistic
            if self.config.compute_gap_statistic:
                gap_result = self._calculate_gap_statistic(features_clean, labels_clean)
                metrics_result.gap_statistic = gap_result
                metrics_result.gap_optimal_clusters = gap_result.get('optimal_clusters', 0)
                metrics_result.gap_scores = gap_result.get('gap_scores', [])
                metrics_result.gap_std_errors = gap_result.get('gap_std_errors', [])

            # Additional metrics
            if self.config.compute_inertia:
                metrics_result.inertia = self._calculate_inertia(features_clean, labels_clean, cluster_centers)

            if self.config.compute_separation_index:
                metrics_result.separation_index = self._calculate_separation_index(features_clean, labels_clean, cluster_centers)

            if self.config.compute_cohesion_index:
                metrics_result.cohesion_index = self._calculate_cohesion_index(features_clean, labels_clean, cluster_centers)

            # Economic distinctness evaluation
            economic_result = self._calculate_economic_significance(features_clean, labels_clean, timestamps, market_data)
            if economic_result:
                metrics_result.economic_significance_scores = economic_result.get('economic_scores', {})
                metrics_result.regime_economic_profiles = economic_result.get('regime_profiles', {})
                metrics_result.risk_adjusted_returns = economic_result.get('risk_adjusted_returns', {})

                # Calculate economic distinctness metrics
                economic_values = list(metrics_result.economic_significance_scores.values())
                if len(economic_values) > 1:
                    # Economic distinctness between regimes (variance-based)
                    economic_distinctness = np.var(economic_values) * 2
                    economic_distinctness = min(economic_distinctness, 1.0)

                    # Average economic consistency across regimes
                    avg_economic_score = np.mean(economic_values)
                else:
                    economic_distinctness = 0.0
                    avg_economic_score = economic_values[0] if economic_values else 0.0

                # Count regimes with high economic consistency (focus on within-regime similarity)
                consistent_regimes = sum(1 for score in metrics_result.economic_significance_scores.values() if score >= 0.6)
                highly_consistent_regimes = sum(1 for score in metrics_result.economic_significance_scores.values() if score >= 0.7)

                metrics_result.economically_significant_regimes = consistent_regimes
                metrics_result.economically_viable_regimes = highly_consistent_regimes

                # Calculate overall economic quality score (emphasizes distinctness and consistency)
                economic_quality_score = (economic_distinctness * 0.4) + (avg_economic_score * 0.6)
                metrics_result.economic_quality_score = economic_quality_score
                metrics_result.economic_interpretation = self._interpret_economic_score(economic_quality_score)

            # Overall quality assessment (now includes both clustering and economic quality)
            metrics_result.overall_quality_score = self._calculate_overall_quality_score(metrics_result)
            metrics_result.quality_interpretation = self._interpret_quality_score(metrics_result.overall_quality_score)

            metrics_result.evaluation_timestamp = pd.Timestamp.now().isoformat()
            metrics_result.computation_time = time.time() - start_time

            return metrics_result

        except Exception as e:
            self.logger.error(f"Clustering quality evaluation failed: {e}")
            return ClusteringQualityResult()

    def _validate_and_preprocess_data(self, features: Union[pd.DataFrame, np.ndarray],
                                    labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Validate and preprocess input data."""
        try:
            # Convert to numpy arrays
            if isinstance(features, pd.DataFrame):
                features_array = features.values
            else:
                features_array = features

            labels_array = np.asarray(labels)

            # Basic validation
            if features_array.shape[0] != labels_array.shape[0]:
                raise ValueError("Features and labels must have same number of samples")

            if len(np.unique(labels_array)) < 2:
                raise ValueError("Need at least 2 clusters for meaningful evaluation")

            # Handle outliers if enabled
            if self.config.handle_outliers:
                features_array = self._remove_outliers(features_array)

            # Feature scaling if enabled
            if self.config.enable_feature_scaling:
                features_array = self._scale_features(features_array)

            return features_array, labels_array

        except Exception as e:
            self.logger.error(f"Data validation/preprocessing failed: {e}")
            raise

    def _remove_outliers(self, features: np.ndarray) -> np.ndarray:
        """Remove outliers using z-score method."""
        try:
            # Calculate z-scores
            z_scores = np.abs((features - np.mean(features, axis=0)) / np.std(features, axis=0))

            # Remove rows with any feature having z-score > threshold
            outlier_mask = np.any(z_scores > self.config.outlier_threshold, axis=1)
            features_clean = features[~outlier_mask]

            if features_clean.shape[0] == 0:
                self.logger.warning("All samples removed as outliers, using original data")
                return features

            return features_clean

        except Exception as e:
            self.logger.warning(f"Outlier removal failed: {e}")
            return features

    def _scale_features(self, features: np.ndarray) -> np.ndarray:
        """Scale features using specified method."""
        try:
            if self.config.feature_scaling_method == "standard":
                scaler = StandardScaler()
                return scaler.fit_transform(features)
            elif self.config.feature_scaling_method == "robust":
                from sklearn.preprocessing import RobustScaler
                scaler = RobustScaler()
                return scaler.fit_transform(features)
            elif self.config.feature_scaling_method == "minmax":
                from sklearn.preprocessing import MinMaxScaler
                scaler = MinMaxScaler()
                return scaler.fit_transform(features)
            else:
                return features

        except Exception as e:
            self.logger.warning(f"Feature scaling failed: {e}")
            return features

    def _calculate_silhouette_score(self, features: np.ndarray, labels: np.ndarray) -> float:
        """Calculate Silhouette Score."""
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                score = silhouette_score(features, labels)

            self.logger.debug(f"Silhouette Score: {score:.4f}")
            return score

        except Exception as e:
            self.logger.warning(f"Silhouette Score calculation failed: {e}")
            return 0.0

    def _calculate_calinski_harabasz_score(self, features: np.ndarray, labels: np.ndarray) -> float:
        """Calculate Calinski-Harabasz Score."""
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                score = calinski_harabasz_score(features, labels)

            self.logger.debug(f"Calinski-Harabasz Score: {score:.4f}")
            return score

        except Exception as e:
            self.logger.warning(f"Calinski-Harabasz Score calculation failed: {e}")
            return 0.0

    def _calculate_davies_bouldin_index(self, features: np.ndarray, labels: np.ndarray) -> float:
        """Calculate Davies-Bouldin Index."""
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                score = davies_bouldin_score(features, labels)

            self.logger.debug(f"Davies-Bouldin Index: {score:.4f}")
            return score

        except Exception as e:
            self.logger.warning(f"Davies-Bouldin Index calculation failed: {e}")
            return 1.0  # Return worst possible score

    def _calculate_gap_statistic(self, features: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
        """Calculate Gap Statistic for optimal cluster number."""
        try:
            n_clusters_range = range(self.config.gap_n_clusters_range[0],
                                   self.config.gap_n_clusters_range[1] + 1)

            gap_scores = []
            std_errors = []
            reference_inertias = []

            # Calculate reference distribution
            for n_clusters in n_clusters_range:
                # Generate reference datasets
                reference_inertias_k = []

                for _ in range(self.config.gap_n_bootstraps):
                    # Generate random data with same distribution
                    random_data = self._generate_reference_data(features, n_clusters)
                    random_labels = self._generate_random_labels(len(features), n_clusters)

                    # Calculate inertia for random data
                    kmeans = KMeans(n_clusters=n_clusters, random_state=self.config.gap_random_state, n_init=10)
                    kmeans.fit(random_data)
                    reference_inertias_k.append(kmeans.inertia_)

                reference_inertias.append(np.mean(reference_inertias_k))

            # Calculate actual clustering inertias
            actual_inertias = []
            for n_clusters in n_clusters_range:
                kmeans = KMeans(n_clusters=n_clusters, random_state=self.config.gap_random_state, n_init=10)
                kmeans.fit(features)
                actual_inertias.append(kmeans.inertia_)

            # Calculate gap scores
            for i, n_clusters in enumerate(n_clusters_range):
                gap_score = np.log(reference_inertias[i]) - np.log(actual_inertias[i])
                gap_scores.append(gap_score)

                # Estimate standard error
                std_error = np.std(np.log(reference_inertias[i]) - np.log(actual_inertias[i]))
                std_errors.append(std_error)

            # Find optimal number of clusters
            optimal_clusters = n_clusters_range[np.argmax(gap_scores)]

            result = {
                'gap_scores': gap_scores,
                'std_errors': std_errors,
                'reference_inertias': reference_inertias,
                'actual_inertias': actual_inertias,
                'optimal_clusters': optimal_clusters,
                'n_clusters_range': list(n_clusters_range)
            }

            self.logger.debug(f"Gap Statistic - Optimal clusters: {optimal_clusters}")
            return result

        except Exception as e:
            self.logger.warning(f"Gap Statistic calculation failed: {e}")
            return {'optimal_clusters': len(np.unique(labels))}

    def _generate_reference_data(self, features: np.ndarray, n_clusters: int) -> np.ndarray:
        """Generate reference data for Gap Statistic."""
        try:
            # Simple approach: generate uniform random data within feature bounds
            n_samples, n_features = features.shape

            # Get bounds for each feature
            feature_mins = np.min(features, axis=0)
            feature_maxs = np.max(features, axis=0)

            # Generate uniform random data
            random_data = np.random.uniform(
                feature_mins,
                feature_maxs,
                size=(n_samples, n_features)
            )

            return random_data

        except Exception as e:
            self.logger.warning(f"Reference data generation failed: {e}")
            return features

    def _generate_random_labels(self, n_samples: int, n_clusters: int) -> np.ndarray:
        """Generate random cluster labels."""
        try:
            return np.random.randint(0, n_clusters, n_samples)
        except Exception:
            return np.zeros(n_samples, dtype=int)

    def _calculate_inertia(self, features: np.ndarray, labels: np.ndarray,
                          cluster_centers: Optional[np.ndarray] = None) -> float:
        """Calculate within-cluster sum of squares (inertia)."""
        try:
            if cluster_centers is None:
                # Calculate cluster centers
                unique_labels = np.unique(labels)
                centers = []

                for label in unique_labels:
                    mask = labels == label
                    if np.any(mask):
                        center = np.mean(features[mask], axis=0)
                        centers.append(center)

                cluster_centers = np.array(centers)

            # Calculate inertia
            inertia = 0.0
            unique_labels = np.unique(labels)

            for i, label in enumerate(unique_labels):
                if i >= len(cluster_centers):
                    break

                mask = labels == label
                if np.any(mask):
                    distances = np.sum((features[mask] - cluster_centers[i]) ** 2)
                    inertia += distances

            return inertia

        except Exception as e:
            self.logger.warning(f"Inertia calculation failed: {e}")
            return 0.0

    def _calculate_separation_index(self, features: np.ndarray, labels: np.ndarray,
                                  cluster_centers: Optional[np.ndarray] = None) -> float:
        """Calculate separation index (between-cluster distance)."""
        try:
            if cluster_centers is None:
                unique_labels = np.unique(labels)
                centers = []

                for label in unique_labels:
                    mask = labels == label
                    if np.any(mask):
                        center = np.mean(features[mask], axis=0)
                        centers.append(center)

                cluster_centers = np.array(centers)

            if len(cluster_centers) < 2:
                return 0.0

            # Calculate pairwise distances between cluster centers
            separation = 0.0
            n_pairs = 0

            for i in range(len(cluster_centers)):
                for j in range(i + 1, len(cluster_centers)):
                    distance = np.linalg.norm(cluster_centers[i] - cluster_centers[j])
                    separation += distance
                    n_pairs += 1

            return separation / n_pairs if n_pairs > 0 else 0.0

        except Exception as e:
            self.logger.warning(f"Separation index calculation failed: {e}")
            return 0.0

    def _calculate_economic_significance(self, features: np.ndarray, labels: np.ndarray,
                                       timestamps: Optional[np.ndarray] = None,
                                       market_data: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Calculate economic distinctness and within-regime consistency for each regime."""
        try:
            economic_scores = {}
            regime_profiles = {}
            risk_adjusted_returns = {}

            unique_labels = np.unique(labels)

            for label in unique_labels:
                regime_mask = labels == label
                if not np.any(regime_mask):
                    continue

                # Get regime data
                regime_features = features[regime_mask]
                regime_size = len(regime_features)

                if market_data is not None and market_data.shape[0] == features.shape[0]:
                    regime_prices = market_data[regime_mask, 3]  # Close prices
                else:
                    # Use first feature as proxy for price
                    regime_prices = regime_features[:, 0]

                # Calculate economic metrics for this regime
                profile = self._calculate_regime_economic_profile(regime_prices, regime_features, timestamps)
                # Focus on within-regime consistency rather than individual profitability
                economic_score = self._calculate_regime_economic_consistency(profile)
                risk_adjusted_return = self._calculate_risk_adjusted_return(regime_prices)

                economic_scores[f'regime_{label}'] = economic_score
                regime_profiles[f'regime_{label}'] = profile
                risk_adjusted_returns[f'regime_{label}'] = risk_adjusted_return

            return {
                'economic_scores': economic_scores,
                'regime_profiles': regime_profiles,
                'risk_adjusted_returns': risk_adjusted_returns
            }

        except Exception as e:
            self.logger.warning(f"Economic significance calculation failed: {e}")
            return {}

    def _calculate_regime_economic_profile(self, prices: np.ndarray, features: np.ndarray,
                                         timestamps: Optional[np.ndarray]) -> Dict[str, Any]:
        """Calculate comprehensive economic profile for a regime."""
        try:
            if len(prices) < 10:  # Need minimum samples
                return {}

            # Calculate returns
            returns = np.diff(prices) / prices[:-1]

            # Basic return metrics
            total_return = (prices[-1] - prices[0]) / prices[0] if prices[0] != 0 else 0.0
            annualized_return = total_return * (252 / len(prices)) if len(prices) > 1 else 0.0  # Assuming daily data

            # Risk metrics
            volatility = np.std(returns) * np.sqrt(252) if len(returns) > 0 else 0.0  # Annualized
            max_drawdown = self._calculate_max_drawdown(np.cumprod(1 + returns))

            # Sharpe ratio (assuming 0% risk-free rate)
            sharpe_ratio = annualized_return / volatility if volatility > 0 else 0.0

            # Sortino ratio (downside deviation)
            downside_returns = returns[returns < 0]
            downside_deviation = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else 0.0
            sortino_ratio = annualized_return / downside_deviation if downside_deviation > 0 else 0.0

            # Win rate and profit factor
            winning_trades = np.sum(returns > 0)
            losing_trades = np.sum(returns < 0)
            win_rate = winning_trades / len(returns) if len(returns) > 0 else 0.0

            avg_win = np.mean(returns[returns > 0]) if winning_trades > 0 else 0.0
            avg_loss = abs(np.mean(returns[returns < 0])) if losing_trades > 0 else 0.0
            profit_factor = (avg_win * winning_trades) / (avg_loss * losing_trades) if losing_trades > 0 else float('inf')

            return {
                'total_return': total_return,
                'annualized_return': annualized_return,
                'volatility': volatility,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'regime_size': len(prices),
                'avg_return': np.mean(returns),
                'return_std': np.std(returns)
            }

        except Exception as e:
            self.logger.warning(f"Regime economic profile calculation failed: {e}")
            return {}

    def _calculate_regime_economic_consistency(self, profile: Dict[str, Any]) -> float:
        """Calculate within-regime economic consistency score."""
        try:
            if not profile:
                return 0.0

            # Focus on within-regime consistency rather than profitability
            score_components = []

            # Return consistency (how stable are returns within the regime)
            returns = profile.get('avg_return', 0.0)
            return_std = profile.get('return_std', 1.0)
            return_consistency = 1.0 / (1.0 + return_std) if return_std > 0 else 1.0
            return_consistency = max(0.0, min(1.0, return_consistency))
            score_components.append(('return_consistency', return_consistency, 0.3))

            # Volatility stability (how consistent is volatility within the regime)
            volatility = profile.get('volatility', 0.0)
            # Lower relative volatility within regime is better
            vol_stability = 1.0 / (1.0 + volatility)
            score_components.append(('volatility_stability', vol_stability, 0.25))

            # Regime size factor (larger regimes tend to be more stable)
            regime_size = profile.get('regime_size', 0)
            size_factor = min(regime_size / 100.0, 1.0)  # Normalize to 0-1
            score_components.append(('regime_size_factor', size_factor, 0.2))

            # Return distribution normality (more normal = more consistent)
            # This is a simple proxy for consistency
            return_distribution_score = 0.5  # Neutral score for now
            score_components.append(('return_distribution', return_distribution_score, 0.15))

            # Sharpe ratio consistency (how stable is risk-adjusted performance)
            sharpe = profile.get('sharpe_ratio', 0.0)
            sharpe_consistency = min(abs(sharpe), 1.0)  # Absolute value for consistency
            score_components.append(('sharpe_consistency', sharpe_consistency, 0.1))

            # Calculate weighted score
            total_score = 0.0
            for component_name, component_score, weight in score_components:
                total_score += component_score * weight

            return min(total_score, 1.0)

        except Exception as e:
            self.logger.warning(f"Economic consistency score calculation failed: {e}")
            return 0.0

    def _calculate_regime_economic_score(self, profile: Dict[str, Any]) -> float:
        """Calculate overall economic significance score for a regime."""
        try:
            if not profile:
                return 0.0

            # Weighted economic score
            score_components = []

            # Sharpe ratio contribution (0-1 range, higher better)
            sharpe = profile.get('sharpe_ratio', 0.0)
            sharpe_score = min(max(sharpe / 2.0, 0.0), 1.0)  # Normalize to 0-1
            score_components.append(('sharpe', sharpe_score, 0.3))

            # Win rate contribution (0-1 range)
            win_rate = profile.get('win_rate', 0.0)
            win_rate_score = win_rate
            score_components.append(('win_rate', win_rate_score, 0.2))

            # Profit factor contribution (0-1 range, higher better)
            profit_factor = profile.get('profit_factor', 0.0)
            pf_score = min(profit_factor / 2.0, 1.0) if profit_factor > 0 else 0.0
            score_components.append(('profit_factor', pf_score, 0.2))

            # Drawdown penalty (lower drawdown is better)
            max_dd = profile.get('max_drawdown', 1.0)
            dd_penalty = 1.0 - max_dd  # Convert to 0-1 score (lower DD = higher score)
            score_components.append(('drawdown_penalty', dd_penalty, 0.2))

            # Volatility consideration (moderate volatility is good)
            volatility = profile.get('volatility', 0.0)
            # Optimal volatility around 0.15-0.25 for most markets
            vol_score = 1.0 - abs(volatility - 0.20) / 0.20
            vol_score = max(0.0, min(1.0, vol_score))
            score_components.append(('volatility_score', vol_score, 0.1))

            # Calculate weighted score
            total_score = 0.0
            for component_name, component_score, weight in score_components:
                total_score += component_score * weight

            return min(total_score, 1.0)

        except Exception as e:
            self.logger.warning(f"Economic score calculation failed: {e}")
            return 0.0

    def _calculate_risk_adjusted_return(self, prices: np.ndarray) -> float:
        """Calculate risk-adjusted return for a regime."""
        try:
            if len(prices) < 2:
                return 0.0

            returns = np.diff(prices) / prices[:-1]
            annualized_return = np.mean(returns) * 252  # Assuming daily data
            volatility = np.std(returns) * np.sqrt(252)

            return annualized_return / volatility if volatility > 0 else 0.0

        except Exception as e:
            self.logger.warning(f"Risk-adjusted return calculation failed: {e}")
            return 0.0

    def _calculate_cohesion_index(self, features: np.ndarray, labels: np.ndarray,
                                cluster_centers: Optional[np.ndarray] = None) -> float:
        """Calculate cohesion index (within-cluster compactness)."""
        try:
            if cluster_centers is None:
                unique_labels = np.unique(labels)
                centers = []

                for label in unique_labels:
                    mask = labels == label
                    if np.any(mask):
                        center = np.mean(features[mask], axis=0)
                        centers.append(center)

                cluster_centers = np.array(centers)

            # Calculate average within-cluster distance
            total_distance = 0.0
            total_samples = 0

            unique_labels = np.unique(labels)

            for i, label in enumerate(unique_labels):
                if i >= len(cluster_centers):
                    break

                mask = labels == label
                if np.any(mask):
                    cluster_features = features[mask]
                    cluster_center = cluster_centers[i]

                    distances = np.linalg.norm(cluster_features - cluster_center, axis=1)
                    total_distance += np.sum(distances)
                    total_samples += len(distances)

            return total_distance / total_samples if total_samples > 0 else 0.0

        except Exception as e:
            self.logger.warning(f"Cohesion index calculation failed: {e}")
            return 0.0

    def _calculate_overall_quality_score(self, result: ClusteringQualityResult) -> float:
        """Calculate overall clustering quality score including economic factors."""
        try:
            scores = []
            weights = []

            # Clustering quality metrics (35% weight)
            clustering_weight = 0.35

            # Silhouette Score (0-1, higher better)
            if result.silhouette_score > 0:
                scores.append(result.silhouette_score)
                weights.append(0.20 * clustering_weight)

            # Calinski-Harabasz Score (higher better)
            if result.calinski_harabasz_score > 0:
                # Normalize to 0-1 range (rough approximation)
                normalized_ch = min(result.calinski_harabasz_score / 1000, 1.0)
                scores.append(normalized_ch)
                weights.append(0.15 * clustering_weight)

            # Economic distinctness metrics (65% weight)
            economic_weight = 0.65

            # Economic distinctness between regimes
            if result.economic_significance_scores:
                # Measure economic separation between regimes
                economic_values = list(result.economic_significance_scores.values())
                if len(economic_values) > 1:
                    # Economic distinctness based on variance between regimes
                    economic_distinctness = np.var(economic_values) * 2  # Scale variance
                    economic_distinctness = min(economic_distinctness, 1.0)
                    scores.append(economic_distinctness)
                    weights.append(0.25 * economic_weight)

                    # Average economic significance across regimes
                    avg_economic_score = np.mean(economic_values)
                    scores.append(avg_economic_score)
                    weights.append(0.25 * economic_weight)

                    # Economic consistency within regimes (inverse of within-regime variance)
                    within_regime_consistency = 1.0 - np.mean([np.var([economic_values[i]]) for i in range(len(economic_values))])
                    scores.append(within_regime_consistency)
                    weights.append(0.15 * economic_weight)
                else:
                    # Single regime case
                    avg_economic_score = economic_values[0] if economic_values else 0.0
                    scores.append(avg_economic_score)
                    weights.append(0.65 * economic_weight)

            # Gap Statistic (if available)
            if result.gap_statistic and 'gap_scores' in result.gap_statistic:
                max_gap = max(result.gap_statistic['gap_scores']) if result.gap_statistic['gap_scores'] else 0
                scores.append(min(max_gap / 2, 1.0))  # Normalize roughly
                weights.append(0.10)

            # Additional structural metrics
            if result.separation_index > 0:
                normalized_separation = min(result.separation_index / 10, 1.0)
                scores.append(normalized_separation)
                weights.append(0.05)

            if result.cohesion_index > 0:
                normalized_cohesion = 1.0 / (1.0 + result.cohesion_index)
                scores.append(normalized_cohesion)
                weights.append(0.05)

            # Weighted average
            if scores and weights:
                total_weight = sum(weights)
                weighted_score = sum(score * weight for score, weight in zip(scores, weights)) / total_weight
                return min(weighted_score, 1.0)

            return 0.0

        except Exception as e:
            self.logger.warning(f"Overall quality score calculation failed: {e}")
            return 0.0

    def _interpret_quality_score(self, score: float) -> str:
        """Interpret overall quality score."""
        if score >= 0.8:
            return "Excellent"
        elif score >= 0.6:
            return "Good"
        elif score >= 0.4:
            return "Fair"
        elif score >= 0.2:
            return "Poor"
        else:
            return "Very Poor"

    def _interpret_economic_score(self, score: float) -> str:
        """Interpret economic distinctness score."""
        if score >= 0.8:
            return "Highly Distinct"
        elif score >= 0.6:
            return "Well Separated"
        elif score >= 0.4:
            return "Moderately Distinct"
        elif score >= 0.2:
            return "Weakly Distinct"
        else:
            return "Poorly Distinct"

    def _calculate_max_drawdown(self, cumulative_returns: np.ndarray) -> float:
        """Calculate maximum drawdown from cumulative returns."""
        try:
            if len(cumulative_returns) == 0:
                return 0.0

            peak = cumulative_returns[0]
            max_dd = 0.0

            for value in cumulative_returns:
                if value > peak:
                    peak = value
                dd = (peak - value) / (1 + peak) if peak != 0 else 0
                max_dd = max(max_dd, dd)

            return max_dd

        except Exception:
            return 0.0


def create_clustering_quality_evaluator(config: Optional[ClusteringQualityConfig] = None) -> ClusteringQualityMetrics:
    """Create clustering quality metrics evaluator."""
    return ClusteringQualityMetrics(config)


def quick_clustering_evaluation(features: Union[pd.DataFrame, np.ndarray],
                              cluster_labels: np.ndarray,
                              config: Optional[ClusteringQualityConfig] = None,
                              timestamps: Optional[np.ndarray] = None,
                              market_data: Optional[np.ndarray] = None) -> ClusteringQualityResult:
    """Quick clustering quality evaluation with default settings."""
    evaluator = create_clustering_quality_evaluator(config)
    return evaluator.evaluate_clustering_quality(features, cluster_labels, None, timestamps, market_data)
