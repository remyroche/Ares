"""Metric calculations for NAS/TAS regime analysis with enhanced validation and quality metrics."""
from __future__ import annotations

from typing import Dict, Any, Optional, Tuple
import time

import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler

# Import common operations for data quality and validation
from src.utils.common_operations import (
    calculate_data_quality_metrics,
    create_data_quality_report,
    validate_dataframe_columns,
    safe_convert_dtypes,
    optimize_dataframe_dtypes,
    get_dataframe_info,
    create_summary_statistics,
    safe_fillna,
    safe_merge_dataframes,
    safe_drop_columns,
    safe_rename_columns,
    validate_timestamp_column,
    safe_timestamp_conversion,
    safe_resample,
    align_dataframes,
    validate_dataframe_schema,
    guard_dataframe_nulls,
    get_memory_usage,
    optimize_memory,
    memory_checkpoint,
    gpu_context
)

# Import math validation for safe operations
from src.utils.math_validation import (
    safe_mean,
    safe_std,
    safe_correlation,
    safe_covariance,
    validate_finite,
    validate_positive,
    validate_range,
    safe_percentage_change,
    safe_weighted_average,
    safe_kelly_calculation,
    safe_percentile,
    safe_matrix_inverse,
    validate_correlation_matrix,
    safe_divide,
    safe_log,
    safe_sqrt,
    safe_power
)

# Import tprint for enhanced logging
from src.utils.tprint import (
    tprint,
    tprint_info,
    tprint_warning,
    tprint_error,
    tprint_success,
    tprint_performance,
    tprint_timer,
    tprint_structured
)

from ..shared_utils.calibration_registry import get_metric_thresholds

# Import M1 hardware optimizations
try:
    from src.utils.hardware.m1_memory_optimizer import get_m1_memory_optimizer
    from src.utils.hardware.m1_cpu_optimizer import get_m1_cpu_optimizer
    from src.utils.hardware.m1_gpu_utils import get_m1_gpu_manager
    M1_HARDWARE_AVAILABLE = True
except ImportError:
    M1_HARDWARE_AVAILABLE = False
    get_m1_memory_optimizer = lambda: None
    get_m1_cpu_optimizer = lambda: None
    get_m1_gpu_manager = lambda: None

def calculate_regime_distribution(labels: np.ndarray, regime_type: str) -> Dict[str, Any]:
    """Calculate distribution statistics for regimes with enhanced validation and quality metrics."""
    with tprint_timer(f"Calculating regime distribution for {regime_type}"):
        try:
            # Validate input data
            labels = validate_finite(labels, "regime_labels")

            # Initialize hardware optimizers
            memory_optimizer = get_m1_memory_optimizer()

            # Use memory checkpoint for large datasets
            with memory_checkpoint("regime_distribution_calculation"):
                unique_labels, counts = np.unique(labels, return_counts=True)
                total_samples = len(labels)

                # Validate data consistency
                if total_samples == 0:
                    tprint_warning(f"No samples found for {regime_type} regimes")
                    return {
                        "regime_type": regime_type,
                        "total_samples": 0,
                        "num_regimes": 0,
                        "regime_counts": {},
                        "regime_percentages": {},
                        "regime_balance": {"error": "no_data"}
                    }

                distribution = {
                    "regime_type": regime_type,
                    "total_samples": int(total_samples),
                    "num_regimes": len(unique_labels),
                    "regime_counts": {},
                    "regime_percentages": {},
                    "regime_balance": {},
                    "quality_metrics": {}
                }

                # Initialize variables to avoid UnboundLocalError
                min_pct = 0.0
                max_pct = 0.0
                std_pct = 0.0

                # Calculate regime statistics with safe math operations
                percentages = []
                for label, count in zip(unique_labels, counts):
                    # Use safe division to avoid division by zero
                    percentage = safe_divide(count * 100, total_samples, default=0.0)
                    percentages.append(percentage)

                    distribution["regime_counts"][f"regime_{int(label)}"] = int(count)
                    distribution["regime_percentages"][f"regime_{int(label)}"] = round(percentage, 2)

                # Calculate balance metrics with safe operations
                if percentages:
                    min_pct = safe_percentile(np.array(percentages), 0.0)
                    max_pct = safe_percentile(np.array(percentages), 100.0)
                    std_pct = safe_std(np.array(percentages))
                    balance_score = safe_divide(1.0 - std_pct, 100.0, default=0.0)

                    distribution["regime_balance"] = {
                        "min_percentage": round(float(min_pct), 2),
                        "max_percentage": round(float(max_pct), 2),
                        "std_percentage": round(float(std_pct), 2),
                        "balance_score": round(float(balance_score), 3),
                    }
                else:
                    distribution["regime_balance"] = {
                        "min_percentage": 0.0,
                        "max_percentage": 0.0,
                        "std_percentage": 0.0,
                        "balance_score": 0.0,
                    }

                # Calculate additional quality metrics using initialized variables
                distribution["quality_metrics"] = {
                    "regime_diversity": len(unique_labels) / max(1, total_samples),
                    "balance_ratio": safe_divide(min_pct, max_pct, default=0.0),
                    "concentration_index": safe_divide(std_pct, 100.0, default=0.0),
                    "regime_stability": 1.0 - safe_divide(std_pct, 100.0, default=1.0)
                }

                # Log distribution statistics
                tprint_structured({
                    "regime_type": regime_type,
                    "total_samples": total_samples,
                    "num_regimes": len(unique_labels),
                    "balance_score": distribution["regime_balance"]["balance_score"],
                    "quality_metrics": distribution["quality_metrics"]
                })

                tprint_success(f"Calculated regime distribution: {len(unique_labels)} regimes")
                return distribution

        except Exception as exc:
            tprint_error(f"Failed to calculate regime distribution for {regime_type}: {exc}")
            raise

def calculate_clustering_metrics(features: Optional[np.ndarray], labels: np.ndarray, regime_type: str) -> Dict[str, Any]:
    """Calculate clustering quality metrics for the provided feature set with enhanced validation and M1 optimizations."""
    with tprint_timer(f"Calculating clustering metrics for {regime_type}"):
        try:
            # Check if features are available
            if features is None:
                tprint_warning(f"⚠️  No features available for {regime_type} clustering metrics")
                return {
                    "regime_type": regime_type,
                    "skipped": True,
                    "reason": "no_features",
                    "message": "Clustering metrics require features. Only regime distribution available."
                }

            # Validate input data
            features = validate_finite(features, "clustering_features")
            labels = validate_finite(labels, "clustering_labels")

            # Initialize hardware optimizers
            memory_optimizer = get_m1_memory_optimizer()
            cpu_optimizer = get_m1_cpu_optimizer()
            gpu_manager = get_m1_gpu_manager()

            # Use memory checkpoint for large datasets
            with memory_checkpoint("clustering_metrics_calculation"):
                # Validate data consistency
                if len(features) != len(labels):
                    raise ValueError(f"Feature and label length mismatch: {len(features)} vs {len(labels)}")

                if len(features) == 0:
                    tprint_warning(f"No data available for {regime_type} clustering metrics")
                    return {
                        "regime_type": regime_type,
                        "silhouette_score": 0.0,
                        "davies_bouldin_score": float('inf'),
                        "calinski_harabasz_score": 0.0,
                        "cv_score": 0.0,
                        "interpretation": {"error": "no_data"},
                        "quality_metrics": {}
                    }

                # Scale features with M1 optimization if available
                scaler = StandardScaler()
                if M1_HARDWARE_AVAILABLE and cpu_optimizer:
                    # Optimize scaling for M1
                    features_scaled = cpu_optimizer.optimize_scaling_operation(features, scaler)
                else:
                    features_scaled = scaler.fit_transform(features)

                # Validate scaled features
                features_scaled = validate_finite(features_scaled, "scaled_features")

                # Calculate clustering metrics with safe operations
                silhouette = safe_silhouette_score(features_scaled, labels)
                davies_bouldin = safe_davies_bouldin_score(features_scaled, labels)
                calinski_harabasz = safe_calinski_harabasz_score(features_scaled, labels)
                cv_score = calculate_cv_score(features_scaled, labels)

                # Calculate additional quality metrics
                quality_metrics = calculate_advanced_quality_metrics(features_scaled, labels)

                result = {
                    "regime_type": regime_type,
                    "silhouette_score": round(float(silhouette), 4),
                    "davies_bouldin_score": round(float(davies_bouldin), 4),
                    "calinski_harabasz_score": round(float(calinski_harabasz), 4),
                    "cv_score": round(float(cv_score), 4),
                    "interpretation": {
                        "silhouette": interpret_silhouette(silhouette),
                        "davies_bouldin": interpret_davies_bouldin(davies_bouldin),
                        "cv_score": interpret_cv_score(cv_score),
                    },
                    "quality_metrics": quality_metrics,
                    "data_info": {
                        "n_samples": len(features),
                        "n_features": features.shape[1] if len(features.shape) > 1 else 1,
                        "n_clusters": len(np.unique(labels)),
                        "memory_usage_mb": features.nbytes / (1024**2)
                    }
                }

                # Log metrics
                tprint_structured({
                    "regime_type": regime_type,
                    "silhouette_score": result["silhouette_score"],
                    "davies_bouldin_score": result["davies_bouldin_score"],
                    "cv_score": result["cv_score"],
                    "quality_metrics": quality_metrics
                })

                tprint_success(f"Calculated clustering metrics for {regime_type}")
                return result

        except Exception as exc:
            tprint_error(f"Failed to calculate clustering metrics for {regime_type}: {exc}")
            raise

def safe_silhouette_score(features: np.ndarray, labels: np.ndarray) -> float:
    """Safely calculate silhouette score with error handling using math_validation."""
    try:
        # Validate inputs using math_validation
        features = validate_finite(features, "silhouette_features")
        labels = validate_finite(labels, "silhouette_labels")

        if len(np.unique(labels)) < 2:
            return 0.0

        # Check for sufficient samples per cluster
        unique_labels, counts = np.unique(labels, return_counts=True)
        if np.any(counts < 2):
            tprint_warning("Some clusters have less than 2 samples, returning 0.0")
            return 0.0

        score = silhouette_score(features, labels)
        return validate_finite(score, "silhouette_score")
    except Exception as exc:
        tprint_warning(f"Silhouette score calculation failed: {exc}")
        return 0.0

def safe_davies_bouldin_score(features: np.ndarray, labels: np.ndarray) -> float:
    """Safely calculate Davies-Bouldin score with error handling using math_validation."""
    try:
        # Validate inputs using math_validation
        features = validate_finite(features, "davies_bouldin_features")
        labels = validate_finite(labels, "davies_bouldin_labels")

        if len(np.unique(labels)) < 2:
            return float('inf')

        # Check for sufficient samples per cluster
        unique_labels, counts = np.unique(labels, return_counts=True)
        if np.any(counts < 2):
            tprint_warning("Some clusters have less than 2 samples, returning inf")
            return float('inf')

        score = davies_bouldin_score(features, labels)
        return validate_finite(score, "davies_bouldin_score")
    except Exception as exc:
        tprint_warning(f"Davies-Bouldin score calculation failed: {exc}")
        return float('inf')

def safe_calinski_harabasz_score(features: np.ndarray, labels: np.ndarray) -> float:
    """Safely calculate Calinski-Harabasz score with error handling using math_validation."""
    try:
        # Validate inputs using math_validation
        features = validate_finite(features, "calinski_harabasz_features")
        labels = validate_finite(labels, "calinski_harabasz_labels")

        if len(np.unique(labels)) < 2:
            return 0.0

        # Check for sufficient samples per cluster
        unique_labels, counts = np.unique(labels, return_counts=True)
        if np.any(counts < 2):
            tprint_warning("Some clusters have less than 2 samples, returning 0.0")
            return 0.0

        score = calinski_harabasz_score(features, labels)
        return validate_finite(score, "calinski_harabasz_score")
    except Exception as exc:
        tprint_warning(f"Calinski-Harabasz score calculation failed: {exc}")
        return 0.0

def calculate_advanced_quality_metrics(features: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
    """Calculate advanced quality metrics for clustering evaluation."""
    try:
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)

        if n_clusters < 2:
            return {"error": "insufficient_clusters"}

        # Calculate cluster separation metrics
        cluster_centers = []
        cluster_sizes = []
        for label in unique_labels:
            cluster_mask = labels == label
            cluster_features = features[cluster_mask]
            if len(cluster_features) > 0:
                cluster_centers.append(safe_mean(cluster_features, axis=0))
                cluster_sizes.append(len(cluster_features))

        cluster_centers = np.array(cluster_centers)
        cluster_sizes = np.array(cluster_sizes)

        # Calculate separation metrics
        center_distances = []
        for i in range(len(cluster_centers)):
            for j in range(i + 1, len(cluster_centers)):
                dist = np.linalg.norm(cluster_centers[i] - cluster_centers[j])
                center_distances.append(dist)

        avg_center_distance = safe_mean(np.array(center_distances)) if center_distances else 0.0

        # Calculate cluster compactness
        within_cluster_distances = []
        for label in unique_labels:
            cluster_mask = labels == label
            cluster_features = features[cluster_mask]
            if len(cluster_features) > 1:
                center = safe_mean(cluster_features, axis=0)
                distances = [np.linalg.norm(point - center) for point in cluster_features]
                within_cluster_distances.extend(distances)

        avg_within_distance = safe_mean(np.array(within_cluster_distances)) if within_cluster_distances else 0.0

        # Calculate quality ratios
        separation_ratio = safe_divide(avg_center_distance, avg_within_distance, default=0.0)
        size_balance = safe_divide(np.min(cluster_sizes), np.max(cluster_sizes), default=0.0)

        return {
            "n_clusters": n_clusters,
            "avg_center_distance": float(avg_center_distance),
            "avg_within_distance": float(avg_within_distance),
            "separation_ratio": float(separation_ratio),
            "size_balance": float(size_balance),
            "cluster_sizes": cluster_sizes.tolist(),
            "compactness_score": float(1.0 - safe_divide(avg_within_distance, avg_center_distance, default=1.0)),
            "separation_score": float(min(1.0, separation_ratio / 2.0))
        }

    except Exception as exc:
        tprint_warning(f"Failed to calculate advanced quality metrics: {exc}")
        return {"error": str(exc)}

def calculate_cv_score(features: np.ndarray, labels: np.ndarray) -> float:
    """Calculate coefficient of variation score with safe math operations."""
    try:
        unique_labels = np.unique(labels)
        within_cv_scores = []

        for label in unique_labels:
            cluster_mask = labels == label
            cluster_features = features[cluster_mask]
            if len(cluster_features) <= 1:
                continue

            feature_cvs = []
            for feature_idx in range(cluster_features.shape[1]):
                feature_values = cluster_features[:, feature_idx]
                feature_values = validate_finite(feature_values, f"feature_{feature_idx}")

                std = safe_std(feature_values)
                mean_abs = safe_mean(np.abs(feature_values))

                # Use math_validation for safe division
                if std > 0 and mean_abs > 0:
                    cv = safe_divide(std, mean_abs, default=0.0)
                    cv = validate_finite(cv, f"cv_feature_{feature_idx}")
                    cv = validate_positive(cv, f"cv_positive_{feature_idx}")
                    feature_cvs.append(cv)

            if feature_cvs:
                within_cv_scores.append(safe_mean(np.array(feature_cvs)))

        # Calculate between-cluster CV
        cluster_centers = []
        for label in unique_labels:
            cluster_mask = labels == label
            cluster_features = features[cluster_mask]
            if len(cluster_features) > 0:
                center = safe_mean(cluster_features, axis=0)
                cluster_centers.append(center)

        if len(cluster_centers) > 1:
            cluster_centers = np.asarray(cluster_centers)
            cluster_centers = validate_finite(cluster_centers, "cluster_centers")

            between_std = safe_std(cluster_centers)
            between_mean_abs = safe_mean(np.abs(cluster_centers))

            # Use math_validation for safe division
            between_cv = safe_divide(between_std, between_mean_abs, default=0.0)
            between_cv = validate_finite(between_cv, "between_cv")
            between_cv = validate_positive(between_cv, "between_cv_positive")
        else:
            between_cv = 0.0

        within_cv = safe_mean(np.array(within_cv_scores)) if within_cv_scores else 0.0
        within_cv = validate_finite(within_cv, "within_cv")
        within_cv = validate_positive(within_cv, "within_cv_positive")

        # Calculate final CV score with validation
        cv_score = 0.6 * max(0.0, 1.0 - within_cv) + 0.4 * min(1.0, between_cv)
        cv_score = validate_finite(cv_score, "final_cv_score")
        cv_score = validate_range(cv_score, 0.0, 1.0, "cv_score_range")

        return float(cv_score)

    except Exception as exc:
        tprint_warning(f"Failed to calculate CV score: {exc}")
        return 0.0

def _resolve_metric_thresholds(metric: str, fallback: Dict[str, float]) -> Dict[str, float]:
    """Merge calibrated thresholds with fallbacks for robustness."""

    thresholds = get_metric_thresholds(metric)
    if not thresholds:
        return fallback

    resolved = fallback.copy()
    for key, value in thresholds.items():
        if isinstance(value, (int, float)) and np.isfinite(value):
            resolved[key] = float(value)
    return resolved

def interpret_silhouette(score: float) -> str:
    """Interpret silhouette score."""
    thresholds = _resolve_metric_thresholds(
        'silhouette',
        {'excellent': 0.7, 'good': 0.5, 'fair': 0.3},
    )

    if score >= thresholds['excellent']:
        return "Excellent clustering"
    if score >= thresholds['good']:
        return "Good clustering"
    if score >= thresholds['fair']:
        return "Fair clustering"
    return "Poor clustering"

def interpret_davies_bouldin(score: float) -> str:
    """Interpret Davies-Bouldin score (lower is better)."""
    thresholds = _resolve_metric_thresholds(
        'davies_bouldin',
        {'excellent': 0.5, 'good': 1.0, 'fair': 2.0},
    )

    if score <= thresholds['excellent']:
        return "Excellent separation"
    if score <= thresholds['good']:
        return "Good separation"
    if score <= thresholds['fair']:
        return "Fair separation"
    return "Poor separation"

def interpret_cv_score(score: float) -> str:
    """Interpret coefficient of variation score."""
    thresholds = _resolve_metric_thresholds(
        'cv_score',
        {'excellent': 0.8, 'good': 0.6, 'fair': 0.4},
    )

    if score >= thresholds['excellent']:
        return "Excellent regime distinction"
    if score >= thresholds['good']:
        return "Good regime distinction"
    if score >= thresholds['fair']:
        return "Fair regime distinction"
    return "Poor regime distinction"
