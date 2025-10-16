"""
Clustering Metrics Calculator for NAS-TAS Clustering.

This module computes comprehensive clustering quality metrics including:
- Silhouette Score
- Davies-Bouldin Index (DBI)
- Calinski-Harabasz Index (CH)
- CV Ratio (BCSS/WCSS)
- Temporal Consistency
- Balance Score
- Composite J Score

Provides both incremental and full recompute modes for efficiency.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import time
from datetime import datetime
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.neighbors import NearestNeighbors
import warnings

from src.utils.tprint import (
    tprint, tprint_info, tprint_success, tprint_warning, tprint_error
)
from src.utils.common_operations import (
    get_m1_gpu_manager, get_m1_memory_optimizer, get_m1_cpu_optimizer
)
from src.utils.common_utilities import (
    calculate_data_quality_metrics, safe_dataframe_operation,
    validate_dataframe_columns, create_summary_statistics
)
from src.utils.math_validation import (
    safe_divide, validate_finite, safe_log, safe_sqrt
)
from src.utils.matrix_operations.unified_operations import UnifiedMatrixOperations

from ..shared_utils import get_logger
from .iterative_optimization import ClusteringStats


@dataclass
class MetricsConfig:
    """Configuration for metrics calculation."""
    # Computation modes
    use_incremental_mode: bool = True
    cache_results: bool = True

    # Temporal analysis
    enable_temporal_consistency: bool = True
    temporal_window_size: int = 20
    temporal_decay_factor: float = 0.95

    # Balance analysis
    enable_balance_analysis: bool = True
    target_balance_ratio: float = 1.0

    # Performance settings
    parallel_computation: bool = False
    memory_optimization: bool = True

    # Reporting
    generate_detailed_reports: bool = True
    export_validation_reports: bool = True


@dataclass
class MetricResult:
    """Result container for individual metrics."""
    value: float
    confidence_interval: Optional[Tuple[float, float]] = None
    computation_time: float = 0.0
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class MetricsReport:
    """Comprehensive metrics report."""
    # Core metrics
    silhouette: MetricResult
    dbi: MetricResult  # Davies-Bouldin Index
    ch: MetricResult   # Calinski-Harabasz Index
    cv_ratio: MetricResult

    # Advanced metrics
    temporal_consistency: Optional[MetricResult] = None
    balance_score: Optional[MetricResult] = None
    composite_j: Optional[MetricResult] = None

    # Metadata
    n_samples: int = 0
    n_features: int = 0
    n_clusters: int = 0
    computation_timestamp: str = ""
    config_used: Optional[MetricsConfig] = None


class ClusteringMetrics:
    """
    Comprehensive clustering metrics calculator.

    Computes all required metrics with both incremental and full recompute modes.
    Provides detailed validation reports and quality assessments.
    """

    def __init__(self, config: Optional[MetricsConfig] = None):
        """Initialize the metrics calculator."""
        self.config = config or MetricsConfig()
        self.logger = get_logger('ClusteringMetrics')

        # Cache for incremental computations
        self._metric_cache = {}

        # Hardware service integration
        try:
            from .hardware_service import HardwareService
            self.hardware_service = HardwareService(verbose=False)  # Less verbose for metrics
            self.hardware_integration_enabled = True
        except ImportError:
            self.hardware_service = None
            self.hardware_integration_enabled = False

        # Initialize hardware optimizations
        self.hardware_manager = None
        self.memory_optimizer = None
        self.matrix_ops = None

        # Initialize M1 hardware optimizations if available
        if self.config.parallel_computation or self.config.memory_optimization:
            self._initialize_hardware_optimizations()

        # Performance tracking
        self.performance_metrics = {
            "total_computation_time": 0.0,
            "computation_count": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "hardware_accelerations": 0,
            "memory_optimizations": 0
        }

    def _initialize_hardware_optimizations(self) -> None:
        """Initialize hardware optimizations for metrics computation."""
        try:
            # Initialize matrix operations with hardware acceleration
            self.matrix_ops = UnifiedMatrixOperations()

            # Get hardware managers for metrics computation
            self.hardware_manager = get_m1_gpu_manager()
            self.memory_optimizer = get_m1_memory_optimizer()

            if self.hardware_manager or self.memory_optimizer:
                tprint("🖥️ Hardware optimizations initialized for metrics computation", "INFO")
            else:
                tprint("⚠️ Hardware optimizations not available for metrics, using CPU fallback", "WARNING")

        except Exception as e:
            tprint(f"❌ Hardware initialization for metrics failed: {e}", "ERROR")
            self.hardware_manager = None
            self.memory_optimizer = None
            self.matrix_ops = None

    async def compute_all_metrics(
        self,
        context: Any,
        config: Any
    ) -> Any:
        """
        Compute all clustering metrics for the given context.

        Args:
            context: Clustering context containing features and assignments
            config: Configuration object

        Returns:
            Updated context with computed metrics
        """
        try:
            tprint("🔢 Computing comprehensive clustering metrics...", "INFO")

            start_time = time.time()

            # Extract data from context
            features = getattr(context, 'optimized_features', None)
            assignments = getattr(context, 'optimized_assignments', None)

            if features is None or assignments is None:
                tprint("⚠️ No optimized features or assignments found in context", "WARNING")
                return context

            # Generate comprehensive metrics report
            report = await self._compute_metrics_report(features, assignments, config)

            # Store in context
            context.clustering_metrics = report
            context.metrics_computation_time = time.time() - start_time

            tprint(f"✅ Metrics computation completed in {context.metrics_computation_time:.2f}s", "SUCCESS")

            # Export validation reports if requested
            if self.config.export_validation_reports:
                await self._export_validation_reports(report, context)

            return context

        except Exception as e:
            tprint(f"❌ Metrics computation failed: {e}", "ERROR")
            raise ValueError(f"Metrics computation failed: {e}")

    async def _compute_metrics_report(
        self,
        features: np.ndarray,
        assignments: np.ndarray,
        config: Any
    ) -> MetricsReport:
        """Compute comprehensive metrics report."""

        try:
            n_samples, n_features = features.shape
            n_clusters = len(np.unique(assignments))

            tprint(f"Computing metrics for {n_samples} samples, {n_features} features, {n_clusters} clusters", "INFO")

            # Core metrics computation
            silhouette_result = await self._compute_silhouette_score(features, assignments)
            dbi_result = await self._compute_dbi_score(features, assignments)
            ch_result = await self._compute_ch_score(features, assignments)
            cv_result = await self._compute_cv_ratio(features, assignments)

            # Advanced metrics (if enabled)
            temporal_result = None
            if self.config.enable_temporal_consistency:
                temporal_result = await self._compute_temporal_consistency(features, assignments, config)

            balance_result = None
            if self.config.enable_balance_analysis:
                balance_result = await self._compute_balance_score(features, assignments)

            # Composite J score
            composite_result = await self._compute_composite_j_score(
                cv_result, balance_result, silhouette_result, temporal_result
            )

            # Create comprehensive report
            report = MetricsReport(
                silhouette=silhouette_result,
                dbi=dbi_result,
                ch=ch_result,
                cv_ratio=cv_result,
                temporal_consistency=temporal_result,
                balance_score=balance_result,
                composite_j=composite_result,
                n_samples=n_samples,
                n_features=n_features,
                n_clusters=n_clusters,
                computation_timestamp=datetime.now().isoformat(),
                config_used=self.config
            )

            return report

        except Exception as e:
            tprint(f"❌ Metrics report computation failed: {e}", "ERROR")
            raise

    async def _compute_silhouette_score(
        self,
        features: np.ndarray,
        assignments: np.ndarray
    ) -> MetricResult:
        """Compute silhouette score with error handling and hardware acceleration."""
        start_time = time.time()

        try:
            if len(np.unique(assignments)) < 2:
                tprint("⚠️ Cannot compute silhouette score: need at least 2 clusters", "WARNING")
                return MetricResult(
                    value=0.0,
                    computation_time=time.time() - start_time,
                    metadata={"error": "insufficient_clusters"}
                )

            # Apply memory optimization if hardware service is available
            optimized_features = features
            if self.hardware_integration_enabled and self.hardware_service:
                try:
                    optimized_features, optimization_info = self.hardware_service.optimize_memory(features)
                    if optimization_info.get("hardware_optimization_used", False):
                        self.performance_metrics["memory_optimizations"] += 1
                except Exception as e:
                    tprint(f"⚠️ Memory optimization failed for silhouette: {e}", "WARNING")

            # Use sample for large datasets to improve performance
            sample_features = optimized_features
            sample_assignments = assignments

            if len(optimized_features) > 10000:
                sample_size = min(5000, len(optimized_features))
                sample_indices = np.random.choice(len(optimized_features), sample_size, replace=False)
                sample_features = optimized_features[sample_indices]
                sample_assignments = assignments[sample_indices]

            silhouette_value = silhouette_score(sample_features, sample_assignments)

            return MetricResult(
                value=silhouette_value,
                computation_time=time.time() - start_time,
                metadata={"method": "sklearn", "sample_size": len(sample_features)}
            )

        except Exception as e:
            tprint(f"⚠️ Silhouette score computation failed: {e}", "WARNING")
            return MetricResult(
                value=0.0,
                computation_time=time.time() - start_time,
                metadata={"error": str(e)}
            )

    async def _compute_dbi_score(
        self,
        features: np.ndarray,
        assignments: np.ndarray
    ) -> MetricResult:
        """Compute Davies-Bouldin Index with error handling."""
        start_time = time.time()

        try:
            if len(np.unique(assignments)) < 2:
                return MetricResult(
                    value=float('inf'),
                    computation_time=time.time() - start_time,
                    metadata={"error": "insufficient_clusters"}
                )

            # Use sample for large datasets
            if len(features) > 10000:
                sample_size = min(5000, len(features))
                sample_indices = np.random.choice(len(features), sample_size, replace=False)
                features_sample = features[sample_indices]
                assignments_sample = assignments[sample_indices]

                dbi_value = davies_bouldin_score(features_sample, assignments_sample)
            else:
                dbi_value = davies_bouldin_score(features, assignments)

            return MetricResult(
                value=dbi_value,
                computation_time=time.time() - start_time,
                metadata={"method": "sklearn", "sample_size": len(features)}
            )

        except Exception as e:
            tprint(f"⚠️ DBI computation failed: {e}", "WARNING")
            return MetricResult(
                value=float('inf'),
                computation_time=time.time() - start_time,
                metadata={"error": str(e)}
            )

    async def _compute_ch_score(
        self,
        features: np.ndarray,
        assignments: np.ndarray
    ) -> MetricResult:
        """Compute Calinski-Harabasz Index with error handling."""
        start_time = time.time()

        try:
            if len(np.unique(assignments)) < 2:
                return MetricResult(
                    value=0.0,
                    computation_time=time.time() - start_time,
                    metadata={"error": "insufficient_clusters"}
                )

            # Use sample for large datasets
            if len(features) > 10000:
                sample_size = min(5000, len(features))
                sample_indices = np.random.choice(len(features), sample_size, replace=False)
                features_sample = features[sample_indices]
                assignments_sample = assignments[sample_indices]

                ch_value = calinski_harabasz_score(features_sample, assignments_sample)
            else:
                ch_value = calinski_harabasz_score(features, assignments)

            return MetricResult(
                value=ch_value,
                computation_time=time.time() - start_time,
                metadata={"method": "sklearn", "sample_size": len(features)}
            )

        except Exception as e:
            tprint(f"⚠️ CH score computation failed: {e}", "WARNING")
            return MetricResult(
                value=0.0,
                computation_time=time.time() - start_time,
                metadata={"error": str(e)}
            )

    async def _compute_cv_ratio(
        self,
        features: np.ndarray,
        assignments: np.ndarray
    ) -> MetricResult:
        """Compute CV ratio (BCSS/WCSS) - key metric for clustering quality."""
        start_time = time.time()

        try:
            # Calculate WCSS (Within-Cluster Sum of Squares)
            unique_clusters = np.unique(assignments)
            wcss = 0.0

            for cluster in unique_clusters:
                cluster_mask = assignments == cluster
                if np.any(cluster_mask):
                    cluster_points = features[cluster_mask]
                    cluster_center = np.mean(cluster_points, axis=0)
                    wcss += np.sum((cluster_points - cluster_center) ** 2)

            # Calculate BCSS (Between-Cluster Sum of Squares)
            overall_center = np.mean(features, axis=0)
            bcss = 0.0

            for cluster in unique_clusters:
                cluster_mask = assignments == cluster
                if np.any(cluster_mask):
                    cluster_points = features[cluster_mask]
                    cluster_center = np.mean(cluster_points, axis=0)
                    cluster_size = len(cluster_points)
                    bcss += cluster_size * np.sum((cluster_center - overall_center) ** 2)

            # CV ratio using safe division
            cv_ratio = safe_divide(bcss, wcss, 0.0)

            return MetricResult(
                value=cv_ratio,
                computation_time=time.time() - start_time,
                metadata={
                    "wcss": wcss,
                    "bcss": bcss,
                    "n_clusters": len(unique_clusters)
                }
            )

        except Exception as e:
            tprint(f"⚠️ CV ratio computation failed: {e}", "WARNING")
            return MetricResult(
                value=0.0,
                computation_time=time.time() - start_time,
                metadata={"error": str(e)}
            )

    async def _compute_temporal_consistency(
        self,
        features: np.ndarray,
        assignments: np.ndarray,
        config: Any
    ) -> MetricResult:
        """Compute temporal consistency score."""
        start_time = time.time()

        try:
            # This is a simplified implementation
            # In a full implementation, this would analyze temporal stability
            # of cluster assignments over time windows

            window_size = min(self.config.temporal_window_size, len(features) // 4)

            if len(features) < 2 * window_size:
                return MetricResult(
                    value=0.5,  # Neutral score for insufficient data
                    computation_time=time.time() - start_time,
                    metadata={"error": "insufficient_temporal_data"}
                )

            # Calculate consistency across rolling windows
            consistency_scores = []

            for i in range(window_size, len(features) - window_size, window_size // 2):
                window_assignments = assignments[i-window_size:i+window_size]

                # Calculate local consistency (simplified)
                unique_in_window = len(np.unique(window_assignments))
                if unique_in_window > 0:
                    # Higher consistency if fewer clusters in window
                    consistency = 1.0 / unique_in_window
                    consistency_scores.append(consistency)

            if not consistency_scores:
                return MetricResult(
                    value=0.5,
                    computation_time=time.time() - start_time,
                    metadata={"error": "no_consistency_windows"}
                )

            # Apply decay factor for recency weighting
            weights = np.array([self.config.temporal_decay_factor ** i for i in range(len(consistency_scores))])
            weights = weights / np.sum(weights)  # Normalize

            temporal_consistency = np.average(consistency_scores, weights=weights)

            return MetricResult(
                value=temporal_consistency,
                computation_time=time.time() - start_time,
                metadata={
                    "window_size": window_size,
                    "decay_factor": self.config.temporal_decay_factor,
                    "n_windows": len(consistency_scores)
                }
            )

        except Exception as e:
            tprint(f"⚠️ Temporal consistency computation failed: {e}", "WARNING")
            return MetricResult(
                value=0.5,
                computation_time=time.time() - start_time,
                metadata={"error": str(e)}
            )

    async def _compute_balance_score(
        self,
        features: np.ndarray,
        assignments: np.ndarray
    ) -> MetricResult:
        """Compute cluster balance score."""
        start_time = time.time()

        try:
            unique_clusters = np.unique(assignments)
            cluster_sizes = []

            for cluster in unique_clusters:
                cluster_sizes.append(np.sum(assignments == cluster))

            cluster_sizes = np.array(cluster_sizes)

            if len(cluster_sizes) <= 1:
                return MetricResult(
                    value=1.0,  # Perfect balance with single cluster
                    computation_time=time.time() - start_time,
                    metadata={"n_clusters": len(unique_clusters)}
                )

            # Calculate balance score based on size distribution
            target_size = len(features) / len(unique_clusters)
            size_ratios = cluster_sizes / target_size

            # Balance score: how close sizes are to target (1.0 = perfect balance)
            balance_score = 1.0 - np.std(size_ratios) / np.mean(size_ratios)

            # Ensure score is between 0 and 1
            balance_score = max(0.0, min(1.0, balance_score))

            return MetricResult(
                value=balance_score,
                computation_time=time.time() - start_time,
                metadata={
                    "cluster_sizes": cluster_sizes.tolist(),
                    "target_size": target_size,
                    "size_std": np.std(cluster_sizes),
                    "size_mean": np.mean(cluster_sizes)
                }
            )

        except Exception as e:
            tprint(f"⚠️ Balance score computation failed: {e}", "WARNING")
            return MetricResult(
                value=0.0,
                computation_time=time.time() - start_time,
                metadata={"error": str(e)}
            )

    async def _compute_composite_j_score(
        self,
        cv_result: MetricResult,
        balance_result: Optional[MetricResult],
        silhouette_result: MetricResult,
        temporal_result: Optional[MetricResult]
    ) -> MetricResult:
        """Compute composite J score combining multiple metrics."""
        start_time = time.time()

        try:
            # Define weights for different metrics (can be made configurable)
            weights = {
                "cv_ratio": 0.40,
                "balance": 0.25,
                "silhouette": 0.20,
                "temporal": 0.15
            }

            composite_score = 0.0
            total_weight = 0.0

            # CV ratio (higher is better)
            if cv_result.value > 0:
                composite_score += weights["cv_ratio"] * cv_result.value
                total_weight += weights["cv_ratio"]

            # Balance score (higher is better)
            if balance_result is not None:
                composite_score += weights["balance"] * balance_result.value
                total_weight += weights["balance"]

            # Silhouette score (higher is better, normalized to 0-1)
            silhouette_normalized = (silhouette_result.value + 1) / 2  # Convert from [-1,1] to [0,1]
            composite_score += weights["silhouette"] * silhouette_normalized
            total_weight += weights["silhouette"]

            # Temporal consistency (higher is better)
            if temporal_result is not None:
                composite_score += weights["temporal"] * temporal_result.value
                total_weight += weights["temporal"]

            # Normalize by total weight
            if total_weight > 0:
                composite_score /= total_weight
            else:
                composite_score = 0.0

            return MetricResult(
                value=composite_score,
                computation_time=time.time() - start_time,
                metadata={
                    "weights": weights,
                    "cv_ratio": cv_result.value,
                    "balance": balance_result.value if balance_result else None,
                    "silhouette": silhouette_result.value,
                    "temporal": temporal_result.value if temporal_result else None,
                    "total_weight": total_weight
                }
            )

        except Exception as e:
            tprint(f"⚠️ Composite J score computation failed: {e}", "WARNING")
            return MetricResult(
                value=0.0,
                computation_time=time.time() - start_time,
                metadata={"error": str(e)}
            )

    def generate_quality_report(
        self,
        features: np.ndarray,
        assignments: np.ndarray
    ) -> Dict[str, Any]:
        """Generate detailed quality assessment report."""

        try:
            report = {
                "quality_assessment": {},
                "cluster_analysis": {},
                "recommendations": [],
                "warnings": []
            }

            # Basic quality metrics
            n_clusters = len(np.unique(assignments))
            n_samples = len(features)

            report["quality_assessment"] = {
                "n_samples": n_samples,
                "n_clusters": n_clusters,
                "cluster_balance": self._assess_cluster_balance(assignments),
                "feature_utilization": self._assess_feature_utilization(features, assignments),
                "separation_quality": self._assess_separation_quality(features, assignments)
            }

            # Per-cluster analysis
            report["cluster_analysis"] = self._analyze_clusters(features, assignments)

            # Generate recommendations
            report["recommendations"] = self._generate_recommendations(
                features, assignments, report["quality_assessment"]
            )

            return report

        except Exception as e:
            tprint(f"❌ Quality report generation failed: {e}", "ERROR")
            return {"error": str(e)}

    def _assess_cluster_balance(self, assignments: np.ndarray) -> Dict[str, Any]:
        """Assess cluster size balance."""
        unique_clusters, cluster_sizes = np.unique(assignments, return_counts=True)

        if len(cluster_sizes) <= 1:
            return {"score": 1.0, "status": "single_cluster"}

        # Calculate balance metrics
        target_size = len(assignments) / len(cluster_sizes)
        size_ratios = cluster_sizes / target_size
        balance_score = 1.0 - np.std(size_ratios) / np.mean(size_ratios)
        balance_score = max(0.0, min(1.0, balance_score))

        # Determine status
        if balance_score > 0.8:
            status = "well_balanced"
        elif balance_score > 0.6:
            status = "moderately_balanced"
        else:
            status = "poorly_balanced"

        return {
            "score": balance_score,
            "status": status,
            "cluster_sizes": cluster_sizes.tolist(),
            "size_std": np.std(cluster_sizes),
            "size_cv": np.std(cluster_sizes) / np.mean(cluster_sizes) if np.mean(cluster_sizes) > 0 else 0
        }

    def _assess_feature_utilization(self, features: np.ndarray, assignments: np.ndarray) -> Dict[str, Any]:
        """Assess how well features are utilized across clusters."""
        try:
            n_features = features.shape[1]
            unique_clusters = np.unique(assignments)

            # Calculate within-cluster variance for each feature
            feature_utilization = []

            for feature_idx in range(n_features):
                feature_values = features[:, feature_idx]

                # Total variance
                total_var = np.var(feature_values)

                if total_var == 0:
                    feature_utilization.append(0.0)
                    continue

                # Within-cluster variance
                within_var = 0.0
                for cluster in unique_clusters:
                    cluster_mask = assignments == cluster
                    if np.any(cluster_mask):
                        cluster_values = feature_values[cluster_mask]
                        within_var += np.var(cluster_values) * len(cluster_values) / len(feature_values)

                # Utilization score (higher is better)
                utilization = 1.0 - (within_var / total_var) if total_var > 0 else 0.0
                feature_utilization.append(utilization)

            avg_utilization = np.mean(feature_utilization)

            return {
                "average_utilization": avg_utilization,
                "utilization_per_feature": feature_utilization,
                "highly_utilized_features": np.sum(np.array(feature_utilization) > 0.7),
                "poorly_utilized_features": np.sum(np.array(feature_utilization) < 0.3)
            }

        except Exception as e:
            return {"error": str(e)}

    def _assess_separation_quality(self, features: np.ndarray, assignments: np.ndarray) -> Dict[str, Any]:
        """Assess cluster separation quality."""
        try:
            unique_clusters = np.unique(assignments)

            if len(unique_clusters) < 2:
                return {"score": 0.0, "status": "insufficient_clusters"}

            # Calculate pairwise distances between cluster centers
            cluster_centers = []
            for cluster in unique_clusters:
                cluster_mask = assignments == cluster
                if np.any(cluster_mask):
                    cluster_centers.append(np.mean(features[cluster_mask], axis=0))

            if len(cluster_centers) < 2:
                return {"score": 0.0, "status": "insufficient_clusters"}

            cluster_centers = np.array(cluster_centers)

            # Calculate minimum distance between any two centers
            min_inter_center_distance = float('inf')
            for i in range(len(cluster_centers)):
                for j in range(i + 1, len(cluster_centers)):
                    distance = np.linalg.norm(cluster_centers[i] - cluster_centers[j])
                    min_inter_center_distance = min(min_inter_center_distance, distance)

            # Calculate average within-cluster scatter
            within_scatter = 0.0
            for cluster in unique_clusters:
                cluster_mask = assignments == cluster
                if np.any(cluster_mask):
                    cluster_points = features[cluster_mask]
                    cluster_center = np.mean(cluster_points, axis=0)
                    scatter = np.mean([np.linalg.norm(point - cluster_center) for point in cluster_points])
                    within_scatter += scatter

            within_scatter /= len(unique_clusters)

            # Separation score (higher is better)
            if within_scatter > 0:
                separation_score = min_inter_center_distance / within_scatter
            else:
                separation_score = 0.0

            return {
                "score": separation_score,
                "min_inter_center_distance": min_inter_center_distance,
                "avg_within_scatter": within_scatter,
                "n_clusters": len(unique_clusters)
            }

        except Exception as e:
            return {"error": str(e)}

    def _analyze_clusters(self, features: np.ndarray, assignments: np.ndarray) -> Dict[str, Any]:
        """Analyze individual clusters."""
        try:
            unique_clusters = np.unique(assignments)
            cluster_analysis = {}

            for cluster in unique_clusters:
                cluster_mask = assignments == cluster
                cluster_points = features[cluster_mask]

                if len(cluster_points) == 0:
                    continue

                # Basic statistics
                cluster_analysis[f"cluster_{cluster}"] = {
                    "size": len(cluster_points),
                    "center": np.mean(cluster_points, axis=0).tolist(),
                    "std": np.std(cluster_points, axis=0).tolist(),
                    "min": np.min(cluster_points, axis=0).tolist(),
                    "max": np.max(cluster_points, axis=0).tolist(),
                    "density": len(cluster_points) / (np.max(cluster_points, axis=0) - np.min(cluster_points, axis=0)).prod() if cluster_points.shape[1] > 1 else 0
                }

            return cluster_analysis

        except Exception as e:
            return {"error": str(e)}

    def _generate_recommendations(
        self,
        features: np.ndarray,
        assignments: np.ndarray,
        quality_assessment: Dict[str, Any]
    ) -> List[str]:
        """Generate recommendations based on quality assessment."""
        recommendations = []

        try:
            # Balance recommendations
            balance_info = quality_assessment.get("cluster_balance", {})
            if isinstance(balance_info, dict) and balance_info.get("status") == "poorly_balanced":
                recommendations.append(
                    "Consider rebalancing clusters - some clusters are significantly larger than others"
                )

            # Separation recommendations
            separation_info = quality_assessment.get("separation_quality", {})
            if isinstance(separation_info, dict) and separation_info.get("score", 0) < 1.0:
                recommendations.append(
                    "Clusters may be too close together - consider increasing number of clusters or different features"
                )

            # Feature utilization recommendations
            utilization_info = quality_assessment.get("feature_utilization", {})
            if isinstance(utilization_info, dict):
                poor_features = utilization_info.get("poorly_utilized_features", 0)
                if poor_features > 0:
                    recommendations.append(
                        f"Consider removing {poor_features} poorly utilized features that don't contribute to cluster separation"
                    )

            # Size recommendations
            if len(features) > 1000 and len(np.unique(assignments)) > 20:
                recommendations.append(
                    "Large dataset with many clusters - consider using sampling or dimensionality reduction"
                )

        except Exception as e:
            recommendations.append(f"Error generating recommendations: {e}")

        return recommendations

    async def _export_validation_reports(self, report: MetricsReport, context: Any) -> None:
        """Export validation reports to files."""
        try:
            # This would implement file export functionality
            # For now, just log the key metrics
            tprint("📊 Clustering Quality Metrics Summary:", "INFO")
            tprint(f"  Silhouette Score: {report.silhouette.value:.4f}", "INFO")
            tprint(f"  Davies-Bouldin Index: {report.dbi.value:.4f}", "INFO")
            tprint(f"  Calinski-Harabasz Score: {report.ch.value:.4f}", "INFO")
            tprint(f"  CV Ratio: {report.cv_ratio.value:.4f}", "INFO")

            if report.balance_score:
                tprint(f"  Balance Score: {report.balance_score.value:.4f}", "INFO")

            if report.temporal_consistency:
                tprint(f"  Temporal Consistency: {report.temporal_consistency.value:.4f}", "INFO")

            if report.composite_j:
                tprint(f"  Composite J Score: {report.composite_j.value:.4f}", "INFO")

        except Exception as e:
            tprint(f"⚠️ Validation report export failed: {e}", "WARNING")

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of metrics computation performance."""
        return {
            "total_computation_time": self.performance_metrics["total_computation_time"],
            "computation_count": self.performance_metrics["computation_count"],
            "cache_hit_ratio": (
                self.performance_metrics["cache_hits"] /
                (self.performance_metrics["cache_hits"] + self.performance_metrics["cache_misses"])
                if (self.performance_metrics["cache_hits"] + self.performance_metrics["cache_misses"]) > 0
                else 0.0
            ),
            "average_computation_time": (
                self.performance_metrics["total_computation_time"] / self.performance_metrics["computation_count"]
                if self.performance_metrics["computation_count"] > 0
                else 0.0
            )
        }

    def reset_metrics(self) -> None:
        """Reset metrics calculator state."""
        try:
            self._metric_cache.clear()
            self.performance_metrics = {
                "total_computation_time": 0.0,
                "computation_count": 0,
                "cache_hits": 0,
                "cache_misses": 0
            }
            tprint("Metrics calculator reset", "INFO")
        except Exception as e:
            tprint(f"❌ Metrics reset failed: {e}", "ERROR")
