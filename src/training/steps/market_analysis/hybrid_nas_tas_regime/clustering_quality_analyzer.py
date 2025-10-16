"""
Comprehensive clustering quality metrics analyzer for regime discovery.

This module provides statistical validation of clustering quality using
multiple metrics including Silhouette, Calinski-Harabasz, Davies-Bouldin,
and Gap statistic analysis.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform
import logging
from typing import Dict, List, Tuple, Optional, Any
import warnings

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class ClusteringQualityAnalyzer:
    """Comprehensive clustering quality analysis for regime discovery."""

    def __init__(self, random_state: int = 42):
        """Initialize the clustering quality analyzer.

        Args:
            random_state: Random state for reproducibility
        """
        self.random_state = random_state
        self.logger = logging.getLogger(__name__)

    def calculate_comprehensive_metrics(self,
                                      features: np.ndarray,
                                      labels: np.ndarray,
                                      regime_assignments: Optional[List] = None,
                                      temporal_alignment: str = 'recent') -> Dict[str, Any]:
        """Calculate comprehensive clustering quality metrics.

        Args:
            features: Feature matrix (n_samples, n_features)
            labels: Cluster assignments (n_samples,)
            regime_assignments: Optional regime assignments for comparison
            temporal_alignment: How to align features with labels ('recent', 'beginning', 'center')

        Returns:
            Dictionary containing all clustering quality metrics
        """
        try:
            self.logger.info("🔍 Calculating comprehensive clustering quality metrics...")

            # Ensure inputs are numpy arrays
            features = np.array(features)
            labels = np.array(labels)

            # Handle dimension mismatch between features and labels
            if features.shape[0] != labels.shape[0]:
                self.logger.warning(f"⚠️ Dimension mismatch: features has {features.shape[0]} samples, labels has {labels.shape[0]} samples")

                # If labels has fewer samples, align features to match labels temporally
                if labels.shape[0] < features.shape[0]:
                    # Align features with labels based on temporal_alignment strategy
                    features = self._align_features_with_labels(features, labels, temporal_alignment)
                    self.logger.info(f"✅ Aligned features to match labels: {features.shape[0]} samples")
                else:
                    # If labels has more samples, this is unexpected - handle gracefully
                    self.logger.warning("⚠️ Labels has more samples than features - this is unexpected")
                    # Truncate labels to match features
                    labels = labels[:features.shape[0]]
                    self.logger.info(f"✅ Truncated labels to match features: {labels.shape[0]} samples")

            # Standardize features for consistent metric calculations
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)

            # Calculate basic metrics
            metrics = self._calculate_basic_metrics(features_scaled, labels)

            # Calculate advanced metrics
            metrics.update(self._calculate_advanced_metrics(features_scaled, labels))

            # Calculate regime-specific metrics
            if regime_assignments is not None:
                metrics.update(self._calculate_regime_metrics(features_scaled, labels, regime_assignments))

            # Calculate stability metrics
            metrics.update(self._calculate_stability_metrics(features_scaled, labels))

            # Calculate overall quality assessment
            metrics.update(self._calculate_quality_assessment(metrics))

            self.logger.info("✅ Clustering quality metrics calculated successfully")
            return metrics

        except Exception as e:
            self.logger.error(f"❌ Clustering quality calculation failed: {e}")
            return self._get_default_metrics()

    def _calculate_basic_metrics(self, features: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
        """Calculate basic clustering quality metrics."""
        try:
            n_clusters = len(np.unique(labels))
            n_samples = len(labels)

            # Silhouette Score (-1 to 1, higher is better)
            silhouette = silhouette_score(features, labels) if n_clusters > 1 else -1.0

            # Calinski-Harabasz Score (higher is better)
            calinski_harabasz = calinski_harabasz_score(features, labels) if n_clusters > 1 else 0.0

            # Davies-Bouldin Index (lower is better)
            davies_bouldin = davies_bouldin_score(features, labels) if n_clusters > 1 else float('inf')

            return {
                'silhouette_score': silhouette,
                'calinski_harabasz_score': calinski_harabasz,
                'davies_bouldin_index': davies_bouldin,
                'n_clusters': n_clusters,
                'n_samples': n_samples
            }

        except Exception as e:
            self.logger.warning(f"⚠️ Basic metrics calculation failed: {e}")
            return {
                'silhouette_score': -1.0,
                'calinski_harabasz_score': 0.0,
                'davies_bouldin_index': float('inf'),
                'n_clusters': len(np.unique(labels)),
                'n_samples': len(labels)
            }

    def _calculate_advanced_metrics(self, features: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
        """Calculate advanced clustering quality metrics."""
        try:
            n_clusters = len(np.unique(labels))

            # Gap Statistic
            gap_statistic = self._calculate_gap_statistic(features, labels)

            # Inertia (within-cluster sum of squares)
            inertia = self._calculate_inertia(features, labels)

            # Cluster separation
            separation = self._calculate_cluster_separation(features, labels)

            # Cluster compactness
            compactness = self._calculate_cluster_compactness(features, labels)

            return {
                'gap_statistic': gap_statistic,
                'inertia': inertia,
                'cluster_separation': separation,
                'cluster_compactness': compactness
            }

        except Exception as e:
            self.logger.warning(f"⚠️ Advanced metrics calculation failed: {e}")
            return {
                'gap_statistic': 0.0,
                'inertia': float('inf'),
                'cluster_separation': 0.0,
                'cluster_compactness': float('inf')
            }

    def _calculate_regime_metrics(self, features: np.ndarray, labels: np.ndarray,
                                regime_assignments: List) -> Dict[str, Any]:
        """Calculate regime-specific metrics."""
        try:
            # Regime distribution analysis
            regime_distribution = self._analyze_regime_distribution(labels)

            # Regime balance
            regime_balance = self._calculate_regime_balance(labels)

            # Regime stability
            regime_stability = self._calculate_regime_stability(labels)

            return {
                'regime_distribution': regime_distribution,
                'regime_balance': regime_balance,
                'regime_stability': regime_stability
            }

        except Exception as e:
            self.logger.warning(f"⚠️ Regime metrics calculation failed: {e}")
            return {
                'regime_distribution': {},
                'regime_balance': 0.0,
                'regime_stability': 0.0
            }

    def _calculate_stability_metrics(self, features: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
        """Calculate clustering stability metrics."""
        try:
            # Cluster consistency
            consistency = self._calculate_cluster_consistency(features, labels)

            # Cluster robustness
            robustness = self._calculate_cluster_robustness(features, labels)

            return {
                'cluster_consistency': consistency,
                'cluster_robustness': robustness
            }

        except Exception as e:
            self.logger.warning(f"⚠️ Stability metrics calculation failed: {e}")
            return {
                'cluster_consistency': 0.0,
                'cluster_robustness': 0.0
            }

    def _calculate_gap_statistic(self, features: np.ndarray, labels: np.ndarray) -> float:
        """Calculate Gap statistic for optimal cluster number validation."""
        try:
            n_clusters = len(np.unique(labels))
            if n_clusters <= 1:
                return 0.0

            # Calculate within-cluster dispersion for actual data
            actual_dispersion = self._calculate_within_cluster_dispersion(features, labels)

            # Generate reference data and calculate dispersion
            n_refs = 10
            ref_dispersions = []

            for _ in range(n_refs):
                # Generate reference data with same distribution
                ref_data = self._generate_reference_data(features)
                ref_labels = self._cluster_reference_data(ref_data, n_clusters)
                ref_dispersion = self._calculate_within_cluster_dispersion(ref_data, ref_labels)
                ref_dispersions.append(ref_dispersion)

            # Calculate Gap statistic
            gap = np.log(np.mean(ref_dispersions)) - np.log(actual_dispersion)
            return gap

        except Exception as e:
            self.logger.warning(f"⚠️ Gap statistic calculation failed: {e}")
            return 0.0

    def _calculate_inertia(self, features: np.ndarray, labels: np.ndarray) -> float:
        """Calculate within-cluster sum of squares (inertia)."""
        try:
            inertia = 0.0
            for cluster_id in np.unique(labels):
                cluster_points = features[labels == cluster_id]
                if len(cluster_points) > 0:
                    centroid = np.mean(cluster_points, axis=0)
                    inertia += np.sum((cluster_points - centroid) ** 2)
            return inertia

        except Exception as e:
            self.logger.warning(f"⚠️ Inertia calculation failed: {e}")
            return float('inf')

    def _calculate_cluster_separation(self, features: np.ndarray, labels: np.ndarray) -> float:
        """Calculate cluster separation (between-cluster distance)."""
        try:
            n_clusters = len(np.unique(labels))
            if n_clusters <= 1:
                return 0.0

            centroids = []
            for cluster_id in np.unique(labels):
                cluster_points = features[labels == cluster_id]
                if len(cluster_points) > 0:
                    centroids.append(np.mean(cluster_points, axis=0))

            centroids = np.array(centroids)

            # Calculate average distance between centroids
            distances = pdist(centroids)
            return np.mean(distances)

        except Exception as e:
            self.logger.warning(f"⚠️ Cluster separation calculation failed: {e}")
            return 0.0

    def _calculate_cluster_compactness(self, features: np.ndarray, labels: np.ndarray) -> float:
        """Calculate cluster compactness (within-cluster distance)."""
        try:
            compactness = 0.0
            total_points = 0

            for cluster_id in np.unique(labels):
                cluster_points = features[labels == cluster_id]
                if len(cluster_points) > 1:
                    centroid = np.mean(cluster_points, axis=0)
                    distances = np.linalg.norm(cluster_points - centroid, axis=1)
                    compactness += np.sum(distances)
                    total_points += len(cluster_points)

            return compactness / total_points if total_points > 0 else float('inf')

        except Exception as e:
            self.logger.warning(f"⚠️ Cluster compactness calculation failed: {e}")
            return float('inf')

    def _analyze_regime_distribution(self, labels: np.ndarray) -> Dict[str, Any]:
        """Analyze regime distribution characteristics."""
        try:
            unique_labels, counts = np.unique(labels, return_counts=True)
            total_samples = len(labels)

            distribution = {}
            for label, count in zip(unique_labels, counts):
                distribution[f'regime_{label}'] = {
                    'count': int(count),
                    'percentage': float(count / total_samples * 100),
                    'label': int(label)
                }

            return distribution

        except Exception as e:
            self.logger.warning(f"⚠️ Regime distribution analysis failed: {e}")
            return {}

    def _calculate_regime_balance(self, labels: np.ndarray) -> float:
        """Calculate regime balance (how evenly distributed regimes are)."""
        try:
            unique_labels, counts = np.unique(labels, return_counts=True)
            if len(unique_labels) <= 1:
                return 1.0

            # Calculate coefficient of variation (lower is more balanced)
            mean_count = np.mean(counts)
            std_count = np.std(counts)
            cv = std_count / mean_count if mean_count > 0 else 1.0

            # Convert to balance score (0-1, higher is more balanced)
            balance = 1.0 / (1.0 + cv)
            return balance

        except Exception as e:
            self.logger.warning(f"⚠️ Regime balance calculation failed: {e}")
            return 0.0

    def _calculate_regime_stability(self, labels: np.ndarray) -> float:
        """Calculate regime stability (how consistent regime assignments are)."""
        try:
            # Calculate regime persistence (how often regimes stay the same)
            if len(labels) <= 1:
                return 1.0

            transitions = 0
            for i in range(1, len(labels)):
                if labels[i] != labels[i-1]:
                    transitions += 1

            stability = 1.0 - (transitions / (len(labels) - 1))
            return max(0.0, stability)

        except Exception as e:
            self.logger.warning(f"⚠️ Regime stability calculation failed: {e}")
            return 0.0

    def _calculate_cluster_consistency(self, features: np.ndarray, labels: np.ndarray) -> float:
        """Calculate cluster consistency across multiple runs."""
        try:
            # Run clustering multiple times with different random seeds
            n_runs = 5
            consistency_scores = []

            for seed in range(n_runs):
                kmeans = KMeans(n_clusters=len(np.unique(labels)),
                              random_state=seed, n_init=10)
                new_labels = kmeans.fit_predict(features)

                # Calculate consistency with original labels
                consistency = self._calculate_label_consistency(labels, new_labels)
                consistency_scores.append(consistency)

            return np.mean(consistency_scores)

        except Exception as e:
            self.logger.warning(f"⚠️ Cluster consistency calculation failed: {e}")
            return 0.0

    def _calculate_cluster_robustness(self, features: np.ndarray, labels: np.ndarray) -> float:
        """Calculate cluster robustness to noise."""
        try:
            # Add small amount of noise and see how labels change
            noise_level = 0.01
            noisy_features = features + np.random.normal(0, noise_level, features.shape)

            # Re-cluster with noisy data
            kmeans = KMeans(n_clusters=len(np.unique(labels)),
                          random_state=self.random_state, n_init=10)
            noisy_labels = kmeans.fit_predict(noisy_features)

            # Calculate robustness (how much labels changed)
            robustness = self._calculate_label_consistency(labels, noisy_labels)
            return robustness

        except Exception as e:
            self.logger.warning(f"⚠️ Cluster robustness calculation failed: {e}")
            return 0.0

    def _calculate_quality_assessment(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall quality assessment based on all metrics."""
        try:
            # Silhouette score assessment
            silhouette = metrics.get('silhouette_score', -1.0)
            if silhouette > 0.5:
                silhouette_quality = 'excellent'
            elif silhouette > 0.3:
                silhouette_quality = 'good'
            elif silhouette > 0.1:
                silhouette_quality = 'fair'
            else:
                silhouette_quality = 'poor'

            # Calinski-Harabasz assessment
            calinski = metrics.get('calinski_harabasz_score', 0.0)
            if calinski > 200:
                calinski_quality = 'excellent'
            elif calinski > 100:
                calinski_quality = 'good'
            elif calinski > 50:
                calinski_quality = 'fair'
            else:
                calinski_quality = 'poor'

            # Davies-Bouldin assessment
            davies = metrics.get('davies_bouldin_index', float('inf'))
            if davies < 1.0:
                davies_quality = 'excellent'
            elif davies < 2.0:
                davies_quality = 'good'
            elif davies < 3.0:
                davies_quality = 'fair'
            else:
                davies_quality = 'poor'

            # Overall quality score
            quality_score = self._calculate_overall_quality_score(metrics)

            return {
                'silhouette_quality': silhouette_quality,
                'calinski_quality': calinski_quality,
                'davies_quality': davies_quality,
                'overall_quality_score': quality_score,
                'quality_interpretation': self._interpret_quality_score(quality_score)
            }

        except Exception as e:
            self.logger.warning(f"⚠️ Quality assessment calculation failed: {e}")
            return {
                'silhouette_quality': 'unknown',
                'calinski_quality': 'unknown',
                'davies_quality': 'unknown',
                'overall_quality_score': 0.0,
                'quality_interpretation': 'Unable to assess quality'
            }

    def _calculate_overall_quality_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate overall quality score (0-1)."""
        try:
            # Normalize metrics to 0-1 scale
            silhouette = max(0, min(1, (metrics.get('silhouette_score', -1) + 1) / 2))
            calinski = min(1, metrics.get('calinski_harabasz_score', 0) / 200)
            davies = max(0, 1 - (metrics.get('davies_bouldin_index', 10) / 10))
            balance = metrics.get('regime_balance', 0)
            stability = metrics.get('regime_stability', 0)

            # Weighted average
            weights = [0.3, 0.2, 0.2, 0.15, 0.15]
            scores = [silhouette, calinski, davies, balance, stability]

            quality_score = sum(w * s for w, s in zip(weights, scores))
            return min(1.0, max(0.0, quality_score))

        except Exception as e:
            self.logger.warning(f"⚠️ Overall quality score calculation failed: {e}")
            return 0.0

    def _interpret_quality_score(self, score: float) -> str:
        """Interpret quality score."""
        if score >= 0.8:
            return 'Excellent clustering quality'
        elif score >= 0.6:
            return 'Good clustering quality'
        elif score >= 0.4:
            return 'Fair clustering quality'
        elif score >= 0.2:
            return 'Poor clustering quality'
        else:
            return 'Very poor clustering quality'

    def _calculate_within_cluster_dispersion(self, features: np.ndarray, labels: np.ndarray) -> float:
        """Calculate within-cluster dispersion."""
        try:
            dispersion = 0.0
            for cluster_id in np.unique(labels):
                cluster_points = features[labels == cluster_id]
                if len(cluster_points) > 0:
                    centroid = np.mean(cluster_points, axis=0)
                    dispersion += np.sum((cluster_points - centroid) ** 2)
            return dispersion

        except Exception as e:
            self.logger.warning(f"⚠️ Within-cluster dispersion calculation failed: {e}")
            return float('inf')

    def _generate_reference_data(self, features: np.ndarray) -> np.ndarray:
        """Generate reference data for Gap statistic."""
        try:
            # Generate uniform random data in the same range
            min_vals = np.min(features, axis=0)
            max_vals = np.max(features, axis=0)

            ref_data = np.random.uniform(min_vals, max_vals, features.shape)
            return ref_data

        except Exception as e:
            self.logger.warning(f"⚠️ Reference data generation failed: {e}")
            return features

    def _cluster_reference_data(self, ref_data: np.ndarray, n_clusters: int) -> np.ndarray:
        """Cluster reference data."""
        try:
            kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
            return kmeans.fit_predict(ref_data)

        except Exception as e:
            self.logger.warning(f"⚠️ Reference data clustering failed: {e}")
            return np.zeros(len(ref_data), dtype=int)

    def _calculate_label_consistency(self, labels1: np.ndarray, labels2: np.ndarray) -> float:
        """Calculate consistency between two label sets."""
        try:
            if len(labels1) != len(labels2):
                return 0.0

            # Calculate adjusted rand index
            from sklearn.metrics import adjusted_rand_score
            return adjusted_rand_score(labels1, labels2)

        except Exception as e:
            self.logger.warning(f"⚠️ Label consistency calculation failed: {e}")
            return 0.0

    def _align_features_with_labels(self, features: np.ndarray, labels: np.ndarray,
                                   temporal_alignment: str) -> np.ndarray:
        """Align features with labels based on temporal alignment strategy.

        Args:
            features: Feature matrix (n_samples, n_features)
            labels: Cluster assignments (n_samples,)
            temporal_alignment: Alignment strategy ('recent', 'beginning', 'center')

        Returns:
            Aligned features array
        """
        try:
            n_labels = len(labels)
            n_features = len(features)

            if temporal_alignment == 'recent':
                # Use the most recent samples (default for regime detection)
                offset = n_features - n_labels
                aligned_features = features[offset:]
                self.logger.info(f"📊 Using recent alignment: offset={offset}")

            elif temporal_alignment == 'beginning':
                # Use the first N samples
                aligned_features = features[:n_labels]
                self.logger.info(f"📊 Using beginning alignment")

            elif temporal_alignment == 'center':
                # Use the center N samples
                start_idx = (n_features - n_labels) // 2
                aligned_features = features[start_idx:start_idx + n_labels]
                self.logger.info(f"📊 Using center alignment: start_idx={start_idx}")

            else:
                # Default to recent alignment
                offset = n_features - n_labels
                aligned_features = features[offset:]
                self.logger.info(f"📊 Using default (recent) alignment: offset={offset}")

            return aligned_features

        except Exception as e:
            self.logger.warning(f"⚠️ Feature alignment failed: {e}")
            # Fallback to simple truncation
            return features[:n_labels]

    def _get_default_metrics(self) -> Dict[str, Any]:
        """Get default metrics when calculation fails."""
        return {
            'silhouette_score': -1.0,
            'calinski_harabasz_score': 0.0,
            'davies_bouldin_index': float('inf'),
            'gap_statistic': 0.0,
            'inertia': float('inf'),
            'cluster_separation': 0.0,
            'cluster_compactness': float('inf'),
            'regime_distribution': {},
            'regime_balance': 0.0,
            'regime_stability': 0.0,
            'cluster_consistency': 0.0,
            'cluster_robustness': 0.0,
            'silhouette_quality': 'unknown',
            'calinski_quality': 'unknown',
            'davies_quality': 'unknown',
            'overall_quality_score': 0.0,
            'quality_interpretation': 'Unable to assess quality',
            'n_clusters': 0,
            'n_samples': 0
        }
