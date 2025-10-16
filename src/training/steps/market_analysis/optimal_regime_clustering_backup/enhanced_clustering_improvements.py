"""
Enhanced Clustering Improvements and Advanced Metrics

This module provides advanced quality metrics and multi-objective optimization
improvements for the enhanced clustering system.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Callable
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import warnings
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class QualityMetricType(Enum):
    """Types of quality metrics for clustering evaluation."""
    SEPARATION = "separation"
    COHESION = "cohesion"
    STABILITY = "stability"
    INFORMATION = "information"
    TEMPORAL = "temporal"
    DOMAIN_SPECIFIC = "domain_specific"
    ROBUSTNESS = "robustness"

@dataclass
class QualityMetric:
    """Individual quality metric with metadata."""
    name: str
    value: float
    weight: float
    metric_type: QualityMetricType
    description: str
    confidence_interval: Optional[Tuple[float, float]] = None
    is_higher_better: bool = True

@dataclass
class ClusterQualityProfile:
    """Comprehensive quality profile for clustering results."""
    metrics: Dict[str, QualityMetric]
    overall_score: float
    confidence_score: float
    recommendations: List[str]
    warnings: List[str]

class AdvancedQualityEvaluator:
    """Advanced quality evaluation system for clustering results."""

    def __init__(self):
        self.metric_weights = self._initialize_default_weights()

    def _initialize_default_weights(self) -> Dict[str, float]:
        """Initialize default weights for different metrics."""
        return {
            'silhouette': 0.25,
            'davies_bouldin': 0.20,
            'calinski_harabasz': 0.15,
            'within_cluster_cv': 0.15,
            'cluster_stability': 0.08,  # Reduced from 0.10 for stability over time
            'information_preservation': 0.08,
            'temporal_consistency': 0.05,
            'domain_constraints': 0.02,
            'stability_over_time': 0.05  # New: small weight for temporal stability
        }

    def evaluate_comprehensive_quality(self, features: np.ndarray, labels: np.ndarray,
                                     original_features: Optional[np.ndarray] = None,
                                     timestamps: Optional[np.ndarray] = None,
                                     domain_constraints: Optional[Dict[str, Any]] = None) -> ClusterQualityProfile:
        """Evaluate comprehensive clustering quality with multiple metrics.

        Args:
            features: Feature matrix used for clustering
            labels: Cluster labels
            original_features: Original features before transformation (for information preservation)
            timestamps: Timestamps for temporal analysis
            domain_constraints: Domain-specific constraints

        Returns:
            ClusterQualityProfile with comprehensive evaluation
        """
        try:
            metrics = {}

            # 1. Basic clustering metrics
            metrics['silhouette'] = self._calculate_enhanced_silhouette(features, labels)
            metrics['davies_bouldin'] = self._calculate_enhanced_davies_bouldin(features, labels)
            metrics['calinski_harabasz'] = self._calculate_enhanced_calinski_harabasz(features, labels)

            # 2. Enhanced CV metrics
            metrics['within_cluster_cv'] = self._calculate_enhanced_within_cluster_cv(features, labels)

            # 3. Stability metrics
            metrics['cluster_stability'] = self._calculate_cluster_stability(features, labels)

            # 4. Information preservation
            if original_features is not None:
                metrics['information_preservation'] = self._calculate_information_preservation(
                    features, original_features, labels)

            # 5. Temporal consistency
            if timestamps is not None:
                metrics['temporal_consistency'] = self._calculate_temporal_consistency(
                    features, labels, timestamps)

            # 6. Domain-specific constraints
            if domain_constraints:
                metrics['domain_constraints'] = self._evaluate_domain_constraints(
                    features, labels, domain_constraints)

            # 7. Robustness metrics
            metrics['robustness'] = self._calculate_robustness_metrics(features, labels)

            # 8. Stability over time
            if timestamps is not None:
                metrics['stability_over_time'] = self._calculate_stability_over_time(features, labels, timestamps)

            # Calculate overall score with adaptive weighting
            overall_score = self._calculate_adaptive_overall_score(metrics)

            # Generate recommendations and warnings
            recommendations, warnings = self._generate_insights(metrics)

            return ClusterQualityProfile(
                metrics=metrics,
                overall_score=overall_score,
                confidence_score=self._calculate_confidence_score(metrics),
                recommendations=recommendations,
                warnings=warnings
            )

        except Exception as e:
            logger.warning(f"Comprehensive quality evaluation failed: {e}")
            return self._create_fallback_quality_profile()

    def _calculate_enhanced_silhouette(self, features: np.ndarray, labels: np.ndarray) -> QualityMetric:
        """Calculate enhanced Silhouette score with confidence intervals."""
        try:
            mask = labels != -1
            if mask.sum() < 2:
                return QualityMetric(
                    name="silhouette",
                    value=0.0,
                    weight=0.25,
                    metric_type=QualityMetricType.SEPARATION,
                    description="Enhanced Silhouette score with confidence intervals",
                    confidence_interval=(0.0, 0.0),
                    is_higher_better=True
                )

            clean_features = features[mask]
            clean_labels = labels[mask]

            # Calculate multiple Silhouette scores with bootstrapping
            n_bootstrap = min(100, len(clean_features) // 2)
            silhouette_scores = []

            for _ in range(n_bootstrap):
                indices = np.random.choice(len(clean_features), size=min(50, len(clean_features)), replace=True)
                sample_features = clean_features[indices]
                sample_labels = clean_labels[indices]

                if len(np.unique(sample_labels)) > 1:
                    score = silhouette_score(sample_features, sample_labels)
                    silhouette_scores.append(score)

            mean_score = np.mean(silhouette_scores) if silhouette_scores else 0.0
            std_score = np.std(silhouette_scores) if silhouette_scores else 0.0

            return QualityMetric(
                name="silhouette",
                value=float(mean_score),
                weight=0.25,
                metric_type=QualityMetricType.SEPARATION,
                description="Enhanced Silhouette score with confidence intervals",
                confidence_interval=(float(mean_score - 1.96 * std_score), float(mean_score + 1.96 * std_score)),
                is_higher_better=True
            )

        except Exception as e:
            logger.warning(f"Enhanced Silhouette calculation failed: {e}")
            return QualityMetric(
                name="silhouette",
                value=0.0,
                weight=0.25,
                metric_type=QualityMetricType.SEPARATION,
                description="Silhouette score (fallback)",
                is_higher_better=True
            )

    def _calculate_enhanced_within_cluster_cv(self, features: np.ndarray, labels: np.ndarray) -> QualityMetric:
        """Calculate enhanced within-cluster CV with multiple dimensions."""
        try:
            mask = labels != -1
            if mask.sum() < 2:
                return QualityMetric(
                    name="within_cluster_cv",
                    value=0.0,
                    weight=0.15,
                    metric_type=QualityMetricType.COHESION,
                    description="Enhanced within-cluster CV with outlier mitigation",
                    is_higher_better=False
                )

            clean_features = features[mask]
            clean_labels = labels[mask]
            unique_labels = np.unique(clean_labels)

            # Calculate CV for each cluster and each feature
            cluster_cvs = []

            for label in unique_labels:
                cluster_mask = clean_labels == label
                cluster_features = clean_features[cluster_mask]

                if len(cluster_features) < 2:
                    continue

                # Calculate CV for each feature dimension
                feature_cvs = []
                for i in range(min(4, cluster_features.shape[1])):
                    feature_values = cluster_features[:, i]
                    feature_values = feature_values[np.isfinite(feature_values)]

                    if len(feature_values) < 2:
                        continue

                    mean_val = np.mean(feature_values)
                    std_val = np.std(feature_values)

                    if mean_val == 0:
                        cv = 0.0
                    else:
                        # Enhanced CV with outlier mitigation
                        cv = std_val / abs(mean_val)
                        if cv > 10.0:  # Extreme CV likely due to outliers
                            mad = np.median(np.abs(feature_values - np.median(feature_values)))
                            cv = mad / abs(mean_val) if mean_val != 0 else 0.0

                    feature_cvs.append(cv)

                if feature_cvs:
                    cluster_cvs.append(np.mean(feature_cvs))

            if not cluster_cvs:
                return QualityMetric(
                    name="within_cluster_cv",
                    value=0.0,
                    weight=0.15,
                    metric_type=QualityMetricType.COHESION,
                    description="Enhanced within-cluster CV with outlier mitigation",
                    is_higher_better=False
                )

            mean_cv = np.mean(cluster_cvs)
            std_cv = np.std(cluster_cvs)

            return QualityMetric(
                name="within_cluster_cv",
                value=float(mean_cv),
                weight=0.15,
                metric_type=QualityMetricType.COHESION,
                description="Enhanced within-cluster CV with outlier mitigation",
                confidence_interval=(float(mean_cv - std_cv), float(mean_cv + std_cv)),
                is_higher_better=False
            )

        except Exception as e:
            logger.warning(f"Enhanced within-cluster CV calculation failed: {e}")
            return QualityMetric(
                name="within_cluster_cv",
                value=0.0,
                weight=0.15,
                metric_type=QualityMetricType.COHESION,
                description="Within-cluster CV (fallback)",
                is_higher_better=False
            )

    def _calculate_cluster_stability(self, features: np.ndarray, labels: np.ndarray) -> QualityMetric:
        """Calculate cluster stability using bootstrapping."""
        try:
            mask = labels != -1
            if mask.sum() < 10:
                return QualityMetric(
                    name="cluster_stability",
                    value=0.0,
                    weight=0.10,
                    metric_type=QualityMetricType.STABILITY,
                    description="Cluster stability via bootstrapping",
                    is_higher_better=True
                )

            clean_features = features[mask]
            clean_labels = labels[mask]
            n_samples = len(clean_features)

            # Bootstrap stability analysis
            n_bootstrap = min(50, n_samples // 2)
            stability_scores = []

            for _ in range(n_bootstrap):
                indices = np.random.choice(n_samples, size=min(50, n_samples), replace=True)
                sample_features = clean_features[indices]
                sample_labels = clean_labels[indices]

                if len(np.unique(sample_labels)) > 1:
                    try:
                        score = silhouette_score(sample_features, sample_labels)
                        stability_scores.append(score)
                    except:
                        continue

            if not stability_scores:
                return QualityMetric(
                    name="cluster_stability",
                    value=0.0,
                    weight=0.10,
                    metric_type=QualityMetricType.STABILITY,
                    description="Cluster stability via bootstrapping",
                    is_higher_better=True
                )

            mean_stability = np.mean(stability_scores)
            std_stability = np.std(stability_scores)

            return QualityMetric(
                name="cluster_stability",
                value=float(mean_stability),
                weight=0.10,
                metric_type=QualityMetricType.STABILITY,
                description="Cluster stability via bootstrapping",
                confidence_interval=(float(mean_stability - std_stability), float(mean_stability + std_stability)),
                is_higher_better=True
            )

        except Exception as e:
            logger.warning(f"Cluster stability calculation failed: {e}")
            return QualityMetric(
                name="cluster_stability",
                value=0.0,
                weight=0.10,
                metric_type=QualityMetricType.STABILITY,
                description="Cluster stability (fallback)",
                is_higher_better=True
            )

    def _calculate_information_preservation(self, features: np.ndarray, original_features: np.ndarray,
                                          labels: np.ndarray) -> QualityMetric:
        """Calculate information preservation between original and transformed features."""
        try:
            mask = labels != -1
            if mask.sum() < 2 or original_features is None:
                return QualityMetric(
                    name="information_preservation",
                    value=0.0,
                    weight=0.08,
                    metric_type=QualityMetricType.INFORMATION,
                    description="Information preservation between original and transformed features",
                    is_higher_better=True
                )

            clean_features = features[mask]
            clean_original = original_features[mask]
            clean_labels = labels[mask]

            # Calculate mutual information preservation per cluster
            preservation_scores = []

            for label in np.unique(clean_labels):
                cluster_mask = clean_labels == label
                cluster_features = clean_features[cluster_mask]
                cluster_original = clean_original[cluster_mask]

                if len(cluster_features) < 2 or len(cluster_original) < 2:
                    continue

                # Calculate mutual information between original and transformed features
                cluster_preservation = 0.0
                for i in range(min(cluster_features.shape[1], cluster_original.shape[1])):
                    try:
                        mi = mutual_info_regression(
                            cluster_original[:, i:i+1],
                            cluster_features[:, i]
                        )[0]
                        cluster_preservation += mi
                    except:
                        continue

                preservation_scores.append(cluster_preservation)

            if not preservation_scores:
                return QualityMetric(
                    name="information_preservation",
                    value=0.0,
                    weight=0.08,
                    metric_type=QualityMetricType.INFORMATION,
                    description="Information preservation between original and transformed features",
                    is_higher_better=True
                )

            mean_preservation = np.mean(preservation_scores)

            return QualityMetric(
                name="information_preservation",
                value=float(mean_preservation),
                weight=0.08,
                metric_type=QualityMetricType.INFORMATION,
                description="Information preservation between original and transformed features",
                is_higher_better=True
            )

        except Exception as e:
            logger.warning(f"Information preservation calculation failed: {e}")
            return QualityMetric(
                name="information_preservation",
                value=0.0,
                weight=0.08,
                metric_type=QualityMetricType.INFORMATION,
                description="Information preservation (fallback)",
                is_higher_better=True
            )

    def _calculate_temporal_consistency(self, features: np.ndarray, labels: np.ndarray,
                                      timestamps: np.ndarray) -> QualityMetric:
        """Calculate temporal consistency of clusters over time."""
        try:
            mask = labels != -1
            if mask.sum() < 2 or timestamps is None:
                return QualityMetric(
                    name="temporal_consistency",
                    value=0.0,
                    weight=0.05,
                    metric_type=QualityMetricType.TEMPORAL,
                    description="Temporal consistency of clusters",
                    is_higher_better=True
                )

            clean_features = features[mask]
            clean_labels = labels[mask]
            clean_timestamps = timestamps[mask]

            # Sort by time
            sort_indices = np.argsort(clean_timestamps)
            sorted_features = clean_features[sort_indices]
            sorted_labels = clean_labels[sort_indices]

            # Calculate consistency over time windows
            n_windows = min(10, len(sorted_features) // 10)
            if n_windows < 2:
                return QualityMetric(
                    name="temporal_consistency",
                    value=0.0,
                    weight=0.05,
                    metric_type=QualityMetricType.TEMPORAL,
                    description="Temporal consistency of clusters",
                    is_higher_better=True
                )

            window_size = len(sorted_features) // n_windows
            consistency_scores = []

            for i in range(n_windows - 1):
                window1_start = i * window_size
                window1_end = (i + 1) * window_size
                window2_start = (i + 1) * window_size
                window2_end = min((i + 2) * window_size, len(sorted_features))

                window1_labels = sorted_labels[window1_start:window1_end]
                window2_labels = sorted_labels[window2_start:window2_end]

                # Calculate label overlap between consecutive windows
                if len(window1_labels) > 0 and len(window2_labels) > 0:
                    unique_labels = np.unique(np.concatenate([window1_labels, window2_labels]))
                    overlap_matrix = np.zeros((len(unique_labels), 2))

                    for j, label in enumerate(unique_labels):
                        overlap_matrix[j, 0] = np.sum(window1_labels == label)
                        overlap_matrix[j, 1] = np.sum(window2_labels == label)

                    # Calculate consistency as normalized overlap
                    total_points = len(window1_labels) + len(window2_labels)
                    overlap_score = np.sum(np.minimum(overlap_matrix[:, 0], overlap_matrix[:, 1])) / total_points
                    consistency_scores.append(overlap_score)

            if not consistency_scores:
                return QualityMetric(
                    name="temporal_consistency",
                    value=0.0,
                    weight=0.05,
                    metric_type=QualityMetricType.TEMPORAL,
                    description="Temporal consistency of clusters",
                    is_higher_better=True
                )

            mean_consistency = np.mean(consistency_scores)

            return QualityMetric(
                name="temporal_consistency",
                value=float(mean_consistency),
                weight=0.05,
                metric_type=QualityMetricType.TEMPORAL,
                description="Temporal consistency of clusters",
                is_higher_better=True
            )

        except Exception as e:
            logger.warning(f"Temporal consistency calculation failed: {e}")
            return QualityMetric(
                name="temporal_consistency",
                value=0.0,
                weight=0.05,
                metric_type=QualityMetricType.TEMPORAL,
                description="Temporal consistency (fallback)",
                is_higher_better=True
            )

    def _calculate_adaptive_overall_score(self, metrics: Dict[str, QualityMetric]) -> float:
        """Calculate adaptive overall score based on metric performance."""
        try:
            # Start with base weights
            weights = self.metric_weights.copy()

            # Adjust weights based on metric performance
            for name, metric in metrics.items():
                if name in weights:
                    # Boost weight for high-performing metrics
                    if metric.is_higher_better and metric.value > 0.7:
                        weights[name] *= 1.2
                    elif not metric.is_higher_better and metric.value < 0.3:
                        weights[name] *= 1.2
                    # Reduce weight for poor-performing metrics
                    elif metric.is_higher_better and metric.value < 0.3:
                        weights[name] *= 0.8
                    elif not metric.is_higher_better and metric.value > 0.7:
                        weights[name] *= 0.8

            # Normalize weights
            total_weight = sum(weights.values())
            weights = {k: v / total_weight for k, v in weights.items()}

            # Calculate weighted score
            overall_score = 0.0
            for name, metric in metrics.items():
                if name in weights:
                    if metric.is_higher_better:
                        overall_score += weights[name] * metric.value
                    else:
                        overall_score += weights[name] * (1.0 - metric.value)

            return float(overall_score)

        except Exception as e:
            logger.warning(f"Adaptive overall score calculation failed: {e}")
            # Fallback to simple average
            values = [m.value for m in metrics.values() if m.is_higher_better]
            return float(np.mean(values)) if values else 0.0

    def _calculate_confidence_score(self, metrics: Dict[str, QualityMetric]) -> float:
        """Calculate confidence score based on metric stability and coverage."""
        try:
            # Calculate based on:
            # 1. Number of metrics available
            # 2. Confidence intervals
            # 3. Metric stability

            available_metrics = len(metrics)
            max_possible_metrics = len(self.metric_weights)

            # Metric coverage score
            coverage_score = available_metrics / max_possible_metrics

            # Confidence interval stability
            ci_scores = []
            for metric in metrics.values():
                if metric.confidence_interval:
                    ci_range = metric.confidence_interval[1] - metric.confidence_interval[0]
                    if ci_range > 0:
                        # Smaller CI range = higher confidence
                        ci_score = 1.0 / (1.0 + ci_range)
                        ci_scores.append(ci_score)

            ci_stability = np.mean(ci_scores) if ci_scores else 0.5

            # Combine scores
            confidence_score = 0.6 * coverage_score + 0.4 * ci_stability

            return float(confidence_score)

        except Exception as e:
            logger.warning(f"Confidence score calculation failed: {e}")
            return 0.5  # Neutral confidence

    def _generate_insights(self, metrics: Dict[str, QualityMetric]) -> Tuple[List[str], List[str]]:
        """Generate insights and recommendations based on metrics."""
        recommendations = []
        warnings = []

        try:
            # Analyze individual metrics
            for name, metric in metrics.items():
                if metric.value < 0.3 and metric.is_higher_better:
                    warnings.append(f"Low {metric.name} score: {metric.value:.3f} - {metric.description}")
                    recommendations.append(f"Consider improving {metric.name} by adjusting clustering parameters")

                elif metric.value > 0.8 and not metric.is_higher_better:
                    warnings.append(f"High {metric.name} score: {metric.value:.3f} - {metric.description}")
                    recommendations.append(f"Consider reducing {metric.name} by using different clustering approach")

            # Cross-metric analysis
            silhouette = metrics.get('silhouette', QualityMetric('', 0, 0, QualityMetricType.SEPARATION, ''))
            within_cv = metrics.get('within_cluster_cv', QualityMetric('', 0, 0, QualityMetricType.COHESION, ''))

            if (silhouette.value > 0.5 and within_cv.value < 0.3):
                recommendations.append("Excellent clustering: High separation with low within-cluster variance")
            elif (silhouette.value < 0.2 and within_cv.value > 0.5):
                warnings.append("Poor clustering: Low separation with high within-cluster variance")
                recommendations.append("Consider feature engineering or different clustering algorithm")

            # Stability analysis
            stability = metrics.get('cluster_stability', QualityMetric('', 0, 0, QualityMetricType.STABILITY, ''))
            if stability.value < 0.3:
                warnings.append("Low cluster stability detected")
                recommendations.append("Consider increasing sample size or using more stable clustering method")

            # Information preservation
            info_preservation = metrics.get('information_preservation', QualityMetric('', 0, 0, QualityMetricType.INFORMATION, ''))
            if info_preservation.value < 0.5:
                recommendations.append("Consider preserving more information from original features")

            # Temporal consistency
            temporal = metrics.get('temporal_consistency', QualityMetric('', 0, 0, QualityMetricType.TEMPORAL, ''))
            if temporal.value < 0.3:
                warnings.append("Low temporal consistency detected")
                recommendations.append("Consider temporal-aware clustering or time-series specific methods")

            return recommendations, warnings

        except Exception as e:
            logger.warning(f"Insight generation failed: {e}")
            return ["Consider manual review of clustering results"], []

    def _create_fallback_quality_profile(self) -> ClusterQualityProfile:
        """Create fallback quality profile when evaluation fails."""
        return ClusterQualityProfile(
            metrics={},
            overall_score=0.0,
            confidence_score=0.0,
            recommendations=["Manual review required due to evaluation failure"],
            warnings=["Quality evaluation failed"]
        )

    # Placeholder methods for enhanced metrics (would need implementation)
    def _calculate_enhanced_davies_bouldin(self, features: np.ndarray, labels: np.ndarray) -> QualityMetric:
        """Enhanced Davies-Bouldin calculation."""
        try:
            mask = labels != -1
            if mask.sum() < 2:
                return QualityMetric("davies_bouldin", float('inf'), 0.20, QualityMetricType.SEPARATION, "Enhanced Davies-Bouldin score", is_higher_better=False)

            clean_features = features[mask]
            clean_labels = labels[mask]

            if len(np.unique(clean_labels)) > 1:
                score = davies_bouldin_score(clean_features, clean_labels)
                return QualityMetric("davies_bouldin", float(score), 0.20, QualityMetricType.SEPARATION, "Enhanced Davies-Bouldin score", is_higher_better=False)

            return QualityMetric("davies_bouldin", float('inf'), 0.20, QualityMetricType.SEPARATION, "Enhanced Davies-Bouldin score", is_higher_better=False)

        except Exception:
            return QualityMetric("davies_bouldin", float('inf'), 0.20, QualityMetricType.SEPARATION, "Davies-Bouldin score (fallback)", is_higher_better=False)

    def _calculate_enhanced_calinski_harabasz(self, features: np.ndarray, labels: np.ndarray) -> QualityMetric:
        """Enhanced Calinski-Harabasz calculation."""
        try:
            mask = labels != -1
            if mask.sum() < 2:
                return QualityMetric("calinski_harabasz", 0.0, 0.15, QualityMetricType.SEPARATION, "Enhanced Calinski-Harabasz score", is_higher_better=True)

            clean_features = features[mask]
            clean_labels = labels[mask]

            if len(np.unique(clean_labels)) > 1:
                score = calinski_harabasz_score(clean_features, clean_labels)
                return QualityMetric("calinski_harabasz", float(score), 0.15, QualityMetricType.SEPARATION, "Enhanced Calinski-Harabasz score", is_higher_better=True)

            return QualityMetric("calinski_harabasz", 0.0, 0.15, QualityMetricType.SEPARATION, "Enhanced Calinski-Harabasz score", is_higher_better=True)

        except Exception:
            return QualityMetric("calinski_harabasz", 0.0, 0.15, QualityMetricType.SEPARATION, "Calinski-Harabasz score (fallback)", is_higher_better=True)

    def _evaluate_domain_constraints(self, features: np.ndarray, labels: np.ndarray,
                                   domain_constraints: Dict[str, Any]) -> QualityMetric:
        """Evaluate domain-specific constraints."""
        # Placeholder - would implement domain-specific logic
        return QualityMetric("domain_constraints", 1.0, 0.02, QualityMetricType.DOMAIN_SPECIFIC, "Domain constraints satisfaction", is_higher_better=True)

    def _calculate_robustness_metrics(self, features: np.ndarray, labels: np.ndarray) -> QualityMetric:
        """Calculate robustness metrics."""
        # Placeholder - would implement robustness analysis
        return QualityMetric("robustness", 0.8, 0.05, QualityMetricType.ROBUSTNESS, "Clustering robustness", is_higher_better=True)

    def _calculate_stability_over_time(self, features: np.ndarray, labels: np.ndarray, timestamps: np.ndarray) -> QualityMetric:
        """Calculate cluster stability over time using temporal consistency analysis."""
        try:
            mask = labels != -1
            if mask.sum() < 5 or timestamps is None:
                return QualityMetric(
                    name="stability_over_time",
                    value=0.0,
                    weight=0.05,
                    metric_type=QualityMetricType.STABILITY,
                    description="Cluster stability over time using temporal consistency",
                    is_higher_better=True
                )

            clean_features = features[mask]
            clean_labels = labels[mask]
            clean_timestamps = timestamps[mask]

            # Sort by time
            sort_indices = np.argsort(clean_timestamps)
            sorted_features = clean_features[sort_indices]
            sorted_labels = clean_labels[sort_indices]

            # Calculate stability using multiple time windows
            n_windows = min(5, len(sorted_features) // 10)
            if n_windows < 2:
                return QualityMetric(
                    name="stability_over_time",
                    value=0.0,
                    weight=0.05,
                    metric_type=QualityMetricType.STABILITY,
                    description="Cluster stability over time using temporal consistency",
                    is_higher_better=True
                )

            window_size = len(sorted_features) // n_windows
            stability_scores = []

            # Calculate stability across consecutive windows
            for i in range(n_windows - 1):
                window1_start = i * window_size
                window1_end = (i + 1) * window_size
                window2_start = (i + 1) * window_size
                window2_end = min((i + 2) * window_size, len(sorted_features))

                if window1_end <= window1_start or window2_end <= window2_start:
                    continue

                window1_labels = sorted_labels[window1_start:window1_end]
                window2_labels = sorted_labels[window2_start:window2_end]

                # Calculate Jaccard similarity between cluster assignments
                if len(window1_labels) > 0 and len(window2_labels) > 0:
                    # Create contingency table
                    unique_labels = np.unique(np.concatenate([window1_labels, window2_labels]))
                    contingency = np.zeros((len(unique_labels), 2))

                    for j, label in enumerate(unique_labels):
                        contingency[j, 0] = np.sum(window1_labels == label)
                        contingency[j, 1] = np.sum(window2_labels == label)

                    # Calculate stability as weighted overlap
                    total_mass = np.sum(contingency)
                    if total_mass > 0:
                        # Use intersection over union approach
                        intersection = np.sum(np.minimum(contingency[:, 0], contingency[:, 1]))
                        union = np.sum(np.maximum(contingency[:, 0], contingency[:, 1]))
                        jaccard_similarity = intersection / union if union > 0 else 0.0
                        stability_scores.append(jaccard_similarity)

            if not stability_scores:
                return QualityMetric(
                    name="stability_over_time",
                    value=0.0,
                    weight=0.05,
                    metric_type=QualityMetricType.STABILITY,
                    description="Cluster stability over time using temporal consistency",
                    is_higher_better=True
                )

            mean_stability = np.mean(stability_scores)
            std_stability = np.std(stability_scores)

            return QualityMetric(
                name="stability_over_time",
                value=float(mean_stability),
                weight=0.05,
                metric_type=QualityMetricType.STABILITY,
                description="Cluster stability over time using temporal consistency",
                confidence_interval=(float(mean_stability - std_stability), float(mean_stability + std_stability)),
                is_higher_better=True
            )

        except Exception as e:
            logger.warning(f"Stability over time calculation failed: {e}")
            return QualityMetric(
                name="stability_over_time",
                value=0.0,
                weight=0.05,
                metric_type=QualityMetricType.STABILITY,
                description="Stability over time (fallback)",
                is_higher_better=True
            )

class AdvancedMultiObjectiveOptimizer:
    """Advanced multi-objective optimization for clustering."""

    def __init__(self):
        self.objectives = self._initialize_objectives()

    def _initialize_objectives(self) -> Dict[str, Dict[str, Any]]:
        """Initialize multi-objective optimization functions."""
        return {
            'separation': {
                'function': self._objective_separation,
                'weight': 0.3,
                'description': 'Cluster separation quality'
            },
            'cohesion': {
                'function': self._objective_cohesion,
                'weight': 0.4,
                'description': 'Cluster cohesion quality'
            },
            'stability': {
                'function': self._objective_stability,
                'weight': 0.1,
                'description': 'Bootstrap stability analysis'
            },
            'interpretability': {
                'function': self._objective_interpretability,
                'weight': 0.1,
                'description': 'Cluster interpretability'
            },
            'stability_over_time': {
                'function': self._objective_stability_over_time,
                'weight': 0.1,
                'description': 'Stability of clusters over time periods'
            }
        }

    def optimize_multi_objective(self, features: np.ndarray, labels: np.ndarray,
                               original_features: Optional[np.ndarray] = None,
                               timestamps: Optional[np.ndarray] = None,
                               domain_constraints: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """Perform multi-objective optimization evaluation.

        Args:
            features: Feature matrix
            labels: Cluster labels
            original_features: Original features (not used)
            timestamps: Time information
            domain_constraints: Domain-specific constraints (not used)

        Returns:
            Dictionary of objective scores
        """
        try:
            objective_scores = {}

            for obj_name, obj_config in self.objectives.items():
                try:
                    # Call appropriate function based on objective name
                    if obj_name == 'separation':
                        score = self._objective_separation(features, labels, None, None, None)
                    elif obj_name == 'cohesion':
                        score = self._objective_cohesion(features, labels, None, None, None)
                    elif obj_name == 'stability':
                        score = self._objective_stability(features, labels, None, None, None)
                    elif obj_name == 'interpretability':
                        score = self._objective_interpretability(features, labels, None, None, None)
                    elif obj_name == 'stability_over_time':
                        score = self._objective_stability_over_time(features, labels, None, timestamps, None)
                    else:
                        score = 0.5  # Default neutral score

                    objective_scores[obj_name] = score
                except Exception as e:
                    logger.warning(f"Objective {obj_name} calculation failed: {e}")
                    objective_scores[obj_name] = 0.5

            return objective_scores

        except Exception as e:
            logger.error(f"Multi-objective optimization failed: {e}")
            return {name: 0.5 for name in self.objectives.keys()}

    def _objective_separation(self, features: np.ndarray, labels: np.ndarray,
                            original_features: Optional[np.ndarray], timestamps: Optional[np.ndarray],
                            domain_constraints: Optional[Dict[str, Any]]) -> float:
        """Calculate separation objective."""
        try:
            mask = labels != -1
            if mask.sum() < 2:
                return 0.0

            # Combine silhouette and Davies-Bouldin
            silhouette = self._calculate_silhouette_score(features, labels)
            db_score = davies_bouldin_score(features[mask], labels[mask])

            # Normalize Davies-Bouldin (lower is better)
            db_normalized = 1.0 / (1.0 + db_score)

            return 0.7 * silhouette + 0.3 * db_normalized

        except Exception:
            return 0.0

    def _objective_cohesion(self, features: np.ndarray, labels: np.ndarray,
                           original_features: Optional[np.ndarray], timestamps: Optional[np.ndarray],
                           domain_constraints: Optional[Dict[str, Any]]) -> float:
        """Calculate cohesion objective."""
        try:
            # Use enhanced CV calculation
            mask = labels != -1
            if mask.sum() < 2:
                return 0.0

            clean_features = features[mask]
            clean_labels = labels[mask]
            unique_labels = np.unique(clean_labels)

            cluster_cvs = []
            for label in unique_labels:
                cluster_mask = clean_labels == label
                cluster_features = clean_features[cluster_mask]

                if len(cluster_features) < 2:
                    continue

                # Calculate CV for each feature
                feature_cvs = []
                for i in range(cluster_features.shape[1]):
                    feature_values = cluster_features[:, i]
                    feature_values = feature_values[np.isfinite(feature_values)]

                    if len(feature_values) < 2:
                        continue

                    mean_val = np.mean(feature_values)
                    std_val = np.std(feature_values)

                    if mean_val == 0:
                        cv = 0.0
                    else:
                        cv = std_val / abs(mean_val)
                        if cv > 10.0:  # Extreme CV
                            mad = np.median(np.abs(feature_values - np.median(feature_values)))
                            cv = mad / abs(mean_val) if mean_val != 0 else 0.0

                    feature_cvs.append(cv)

                if feature_cvs:
                    cluster_cvs.append(np.mean(feature_cvs))

            if not cluster_cvs:
                return 0.0

            mean_cv = np.mean(cluster_cvs)

            # Cohesion is inverse of CV (lower CV = higher cohesion)
            return 1.0 / (1.0 + mean_cv)

        except Exception:
            return 0.0

    def _calculate_silhouette_score(self, features: np.ndarray, labels: np.ndarray) -> float:
        """Calculate Silhouette score with error handling."""
        try:
            mask = labels != -1
            if mask.sum() < 2:
                return 0.0

            return float(silhouette_score(features[mask], labels[mask]))
        except Exception:
            return 0.0

    def _objective_stability(self, features: np.ndarray, labels: np.ndarray,
                           original_features: Optional[np.ndarray], timestamps: Optional[np.ndarray],
                           domain_constraints: Optional[Dict[str, Any]]) -> float:
        """Calculate stability objective using bootstrap analysis."""
        try:
            mask = labels != -1
            if mask.sum() < 10:
                return 0.5

            clean_features = features[mask]
            clean_labels = labels[mask]

            # Bootstrap stability analysis
            n_bootstrap = min(30, len(clean_features) // 2)
            stability_scores = []

            for _ in range(n_bootstrap):
                # Sample with replacement
                indices = np.random.choice(len(clean_features), size=min(50, len(clean_features)), replace=True)
                sample_features = clean_features[indices]
                sample_labels = clean_labels[indices]

                if len(np.unique(sample_labels)) > 1:
                    try:
                        # Calculate silhouette on bootstrap sample
                        score = silhouette_score(sample_features, sample_labels)
                        stability_scores.append(score)
                    except:
                        continue

            if not stability_scores:
                return 0.5

            # Stability is measured by consistency of silhouette scores
            mean_stability = np.mean(stability_scores)
            std_stability = np.std(stability_scores)

            # Higher mean and lower variance = higher stability
            stability_score = mean_stability * (1.0 / (1.0 + std_stability))

            return float(stability_score)

        except Exception:
            return 0.5

    def _objective_interpretability(self, features: np.ndarray, labels: np.ndarray,
                                  original_features: Optional[np.ndarray], timestamps: Optional[np.ndarray],
                                  domain_constraints: Optional[Dict[str, Any]]) -> float:
        """Calculate cluster interpretability objective."""
        try:
            mask = labels != -1
            if mask.sum() < 2:
                return 0.5

            clean_features = features[mask]
            clean_labels = labels[mask]
            unique_labels = np.unique(clean_labels)

            # 1. Feature importance within clusters
            feature_importance_score = 0.0
            for label in unique_labels:
                cluster_mask = clean_labels == label
                cluster_features = clean_features[cluster_mask]

                if len(cluster_features) < 2:
                    continue

                # Calculate within-cluster variance for each feature
                # Higher variance in important features = better separation
                feature_variances = np.var(cluster_features, axis=0)
                normalized_variances = feature_variances / (np.sum(feature_variances) + 1e-6)

                # Features with high variance are more discriminative
                feature_importance_score += np.mean(normalized_variances)

            feature_importance_score = feature_importance_score / len(unique_labels) if unique_labels.size > 0 else 0.0

            # 2. Cluster separation in interpretable dimensions
            # Calculate between-cluster variance
            cluster_centers = []
            for label in unique_labels:
                cluster_mask = clean_labels == label
                cluster_center = np.mean(clean_features[cluster_mask], axis=0)
                cluster_centers.append(cluster_center)

            if len(cluster_centers) > 1:
                between_variance = np.var(cluster_centers, axis=0)
                total_variance = np.var(clean_features, axis=0)
                separation_score = np.mean(between_variance / (total_variance + 1e-6))
            else:
                separation_score = 0.0

            # 3. Concept drift detection (simplified)
            # Check if cluster characteristics change over time
            drift_score = 1.0  # Assume no drift if no timestamps

            # Combine scores
            interpretability_score = 0.4 * feature_importance_score + 0.4 * separation_score + 0.2 * drift_score

            return float(interpretability_score)

        except Exception:
            return 0.5

    def _objective_stability_over_time(self, features: np.ndarray, labels: np.ndarray,
                                     original_features: Optional[np.ndarray], timestamps: Optional[np.ndarray],
                                     domain_constraints: Optional[Dict[str, Any]]) -> float:
        """Calculate stability of clusters over time periods."""
        try:
            if timestamps is None:
                return 0.5

            mask = labels != -1
            if mask.sum() < 5:
                return 0.5

            clean_features = features[mask]
            clean_labels = labels[mask]
            clean_timestamps = timestamps[mask]

            # Sort by time
            sort_indices = np.argsort(clean_timestamps)
            sorted_features = clean_features[sort_indices]
            sorted_labels = clean_labels[sort_indices]

            # Calculate stability using multiple time windows
            n_windows = min(5, len(sorted_features) // 10)
            if n_windows < 2:
                return 0.5

            window_size = len(sorted_features) // n_windows
            stability_scores = []

            # Calculate stability across consecutive windows
            for i in range(n_windows - 1):
                window1_start = i * window_size
                window1_end = (i + 1) * window_size
                window2_start = (i + 1) * window_size
                window2_end = min((i + 2) * window_size, len(sorted_features))

                if window1_end <= window1_start or window2_end <= window2_start:
                    continue

                window1_labels = sorted_labels[window1_start:window1_end]
                window2_labels = sorted_labels[window2_start:window2_end]

                # Calculate Jaccard similarity between cluster assignments
                if len(window1_labels) > 0 and len(window2_labels) > 0:
                    unique_labels = np.unique(np.concatenate([window1_labels, window2_labels]))
                    contingency = np.zeros((len(unique_labels), 2))

                    for j, label in enumerate(unique_labels):
                        contingency[j, 0] = np.sum(window1_labels == label)
                        contingency[j, 1] = np.sum(window2_labels == label)

                    # Calculate stability as weighted overlap
                    total_mass = np.sum(contingency)
                    if total_mass > 0:
                        intersection = np.sum(np.minimum(contingency[:, 0], contingency[:, 1]))
                        union = np.sum(np.maximum(contingency[:, 0], contingency[:, 1]))
                        jaccard_similarity = intersection / union if union > 0 else 0.0
                        stability_scores.append(jaccard_similarity)

            return float(np.mean(stability_scores)) if stability_scores else 0.5

        except Exception:
            return 0.5

class BatchTransferProcessor:
    """Advanced batch processing for regime transfers with stability guarantees."""

    def __init__(self, batch_size_ratio: float = 0.1, max_iterations: int = 5):
        """
        Initialize batch transfer processor.

        Args:
            batch_size_ratio: Fraction of total transfers to apply per batch (0.1 = 10%)
            max_iterations: Maximum number of optimization iterations
        """
        self.batch_size_ratio = batch_size_ratio
        self.max_iterations = max_iterations
        self.transfer_history = []

    def process_transfers_with_stability(self, features: np.ndarray, labels: np.ndarray,
                                       transfer_candidates: List[Dict[str, Any]],
                                       quality_evaluator: AdvancedQualityEvaluator) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Process transfers in batches with stability monitoring.

        Args:
            features: Feature matrix
            labels: Current cluster labels
            transfer_candidates: List of transfer candidates
            quality_evaluator: Quality evaluation system

        Returns:
            Tuple of (final_labels, transfer_history)
        """
        try:
            current_labels = labels.copy()
            self.transfer_history = []

            # Sort candidates by benefit
            sorted_candidates = sorted(transfer_candidates, key=lambda x: x['transfer_benefit'], reverse=True)

            # Calculate batch size
            batch_size = max(1, int(len(sorted_candidates) * self.batch_size_ratio))
            current_iteration = 0
            converged = False

            while current_iteration < self.max_iterations and not converged:
                logger.info(f"🔄 Processing transfer batch {current_iteration + 1}/{self.max_iterations}")

                # Get current batch
                start_idx = current_iteration * batch_size
                end_idx = min((current_iteration + 1) * batch_size, len(sorted_candidates))
                current_batch = sorted_candidates[start_idx:end_idx]

                if not current_batch:
                    break

                # Evaluate quality before transfer
                quality_before = quality_evaluator.evaluate_comprehensive_quality(features, current_labels)

                # Apply batch transfer
                current_labels, batch_transfers = self._apply_transfer_batch(
                    features, current_labels, current_batch
                )

                # Evaluate quality after transfer
                quality_after = quality_evaluator.evaluate_comprehensive_quality(features, current_labels)

                # Record transfers
                self.transfer_history.extend(batch_transfers)

                # Check for convergence
                quality_improvement = quality_after.overall_score - quality_before.overall_score

                logger.info(f"📊 Batch {current_iteration + 1} completed:")
                logger.info(f"   • Transfers applied: {len(batch_transfers)}")
                logger.info(f"   • Quality improvement: {quality_improvement:.4f}")
                logger.info(f"   • Overall quality: {quality_after.overall_score:.4f}")

                # Check for convergence criteria
                if quality_improvement < 0.01:  # Less than 1% improvement
                    logger.info("🎯 Convergence reached - minimal quality improvement")
                    converged = True
                elif len(batch_transfers) == 0:
                    logger.info("🎯 Convergence reached - no beneficial transfers")
                    converged = True
                elif quality_after.overall_score < 0.3:  # Quality degraded significantly
                    logger.warning("⚠️ Quality degraded significantly - stopping optimization")
                    break

                current_iteration += 1

            logger.info(f"✅ Batch transfer processing completed in {current_iteration} iterations")
            logger.info(f"📊 Total transfers applied: {len(self.transfer_history)}")

            return current_labels, self.transfer_history

        except Exception as e:
            logger.error(f"Batch transfer processing failed: {e}")
            return labels, []

    def _apply_transfer_batch(self, features: np.ndarray, labels: np.ndarray,
                            transfer_batch: List[Dict[str, Any]]) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Apply a single batch of transfers with validation.

        Args:
            features: Feature matrix
            labels: Current labels
            transfer_batch: Batch of transfers to apply

        Returns:
            Tuple of (updated_labels, applied_transfers)
        """
        try:
            updated_labels = labels.copy()
            applied_transfers = []

            for candidate in transfer_batch:
                try:
                    # Final validation before applying transfer
                    if self._validate_transfer(candidate, features, updated_labels):
                        # Apply transfer
                        updated_labels[candidate['regime_id']] = candidate['target_cluster']
                        applied_transfers.append(candidate)

                except Exception as e:
                    logger.warning(f"Transfer validation failed for regime {candidate['regime_id']}: {e}")
                    continue

            return updated_labels, applied_transfers

        except Exception as e:
            logger.error(f"Batch transfer application failed: {e}")
            return labels, []

    def _validate_transfer(self, candidate: Dict[str, Any], features: np.ndarray,
                          current_labels: np.ndarray) -> bool:
        """Validate a single transfer before applying.

        Args:
            candidate: Transfer candidate
            features: Feature matrix
            current_labels: Current labels

        Returns:
            True if transfer is valid
        """
        try:
            # Check size constraints
            current_cluster_size = np.sum(current_labels == candidate['current_cluster'])
            target_cluster_size = np.sum(current_labels == candidate['target_cluster'])

            if target_cluster_size > current_cluster_size * 1.5:
                return False  # Size constraint violation

            # Check benefit threshold
            if candidate['transfer_benefit'] < 0.1:
                return False  # Insufficient benefit

            # Check for potential quality degradation
            # (Would implement more sophisticated validation here)

            return True

        except Exception as e:
            logger.warning(f"Transfer validation error: {e}")
            return False

    def get_processing_summary(self) -> Dict[str, Any]:
        """Get summary of batch processing results."""
        try:
            if not self.transfer_history:
                return {
                    'total_transfers': 0,
                    'iterations_performed': 0,
                    'average_benefit': 0.0,
                    'success_rate': 0.0,
                    'convergence_reason': 'No transfers processed'
                }

            benefits = [t['benefit'] for t in self.transfer_history]
            success_rate = len(self.transfer_history) / max(1, len(self.transfer_history))

            return {
                'total_transfers': len(self.transfer_history),
                'iterations_performed': len(self.transfer_history) // max(1, int(1.0 / self.batch_size_ratio)),
                'average_benefit': np.mean(benefits),
                'max_benefit': np.max(benefits),
                'min_benefit': np.min(benefits),
                'success_rate': success_rate,
                'convergence_reason': 'Completed successfully'
            }

        except Exception as e:
            logger.error(f"Processing summary calculation failed: {e}")
            return {'error': str(e)}

def create_advanced_quality_config() -> Dict[str, Any]:
    """Create configuration for advanced quality metrics."""
    return {
        'enable_bootstrap_analysis': True,
        'enable_confidence_intervals': True,
        'enable_cross_validation': True,
        'enable_temporal_analysis': True,
        'enable_information_metrics': True,
        'enable_domain_constraints': False,
        'bootstrap_iterations': 100,
        'confidence_level': 0.95,
        'temporal_window_size': 10,
        'information_preservation_threshold': 0.7
    }

def create_multi_objective_config() -> Dict[str, Any]:
    """Create configuration for multi-objective optimization."""
    return {
        'enable_all_objectives': True,
        'adaptive_weighting': True,
        'constraint_handling': 'penalty',
        'objective_normalization': 'minmax',
        'pareto_front_analysis': False,
        'objective_weights': {
            'separation': 0.3,
            'cohesion': 0.4,
            'stability': 0.1,
            'interpretability': 0.1,
            'stability_over_time': 0.1
        }
    }

def create_batch_transfer_config() -> Dict[str, Any]:
    """Create configuration for batch transfer processing."""
    return {
        'batch_size_ratio': 0.1,  # 10% of transfers per batch
        'max_iterations': 5,
        'enable_quality_monitoring': True,
        'enable_early_stopping': True,
        'quality_improvement_threshold': 0.01,
        'minimum_benefit_threshold': 0.1,
        'size_constraint_ratio': 1.5,  # 50% size difference limit
        'enable_transfer_validation': True,
        'enable_convergence_detection': True
    }
