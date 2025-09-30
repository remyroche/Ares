"""Metric calculations for NAS/TAS regime analysis."""
from __future__ import annotations

from typing import Dict, Any

import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler

# Enhanced utility imports
from src.utils.common_operations import (
    safe_dataframe_operation, validate_dataframe_columns, safe_convert_dtypes,
    calculate_data_quality_metrics, optimize_dataframe_memory, safe_numpy_operation,
    validate_numpy_array, safe_array_operation, monitor_performance, log_performance_metrics
)
from src.utils.common_utilities import (
    safe_dataframe_operation as safe_df_op, validate_dataframe_columns as validate_df_cols,
    safe_convert_dtypes as safe_convert, calculate_data_quality_metrics as calc_quality,
    optimize_dataframe_performance, safe_apply_function, validate_data_consistency,
    calculate_statistical_metrics, safe_aggregation, validate_data_types
)
from src.utils.math_validation import (
    safe_divide, safe_log, safe_sqrt, safe_power, validate_finite, safe_exp,
    safe_sin, safe_cos, safe_tan, validate_positive, validate_range, safe_abs,
    safe_min, safe_max, safe_mean, safe_std, safe_correlation, safe_covariance,
    validate_matrix_operations, safe_matrix_multiply, safe_matrix_inverse,
    safe_eigenvalues, safe_svd, validate_numerical_stability
)
from src.utils.ml_common.validation.unified_cv import (
    perform_cross_validation, calculate_oof_predictions, validate_lookahead_bias,
    perform_time_series_cv, calculate_cv_metrics, validate_cv_results
)
from src.utils.tprint import tprint, tprint_info, tprint_success, tprint_warning, tprint_error


def calculate_regime_distribution(labels: np.ndarray, regime_type: str) -> Dict[str, Any]:
    """Calculate distribution statistics for regimes."""
    unique_labels, counts = np.unique(labels, return_counts=True)
    total_samples = len(labels)

    distribution = {
        "regime_type": regime_type,
        "total_samples": int(total_samples),
        "num_regimes": len(unique_labels),
        "regime_counts": {},
        "regime_percentages": {},
        "regime_balance": {},
    }

    for label, count in zip(unique_labels, counts):
        percentage = (count / total_samples) * 100 if total_samples else 0.0
        distribution["regime_counts"][f"regime_{int(label)}"] = int(count)
        distribution["regime_percentages"][f"regime_{int(label)}"] = round(percentage, 2)

    percentages = list(distribution["regime_percentages"].values()) or [0.0]
    distribution["regime_balance"] = {
        "min_percentage": round(float(np.min(percentages)), 2),
        "max_percentage": round(float(np.max(percentages)), 2),
        "std_percentage": round(float(np.std(percentages)), 2),
        "balance_score": round(float(1.0 - (np.std(percentages) / 100)), 3),
    }
    return distribution


def calculate_clustering_metrics(features: np.ndarray, labels: np.ndarray, regime_type: str) -> Dict[str, Any]:
    """Calculate clustering quality metrics with enhanced validation and optimization."""
    try:
        tprint(f"🔄 Calculating clustering metrics for {regime_type}", "INFO")
        
        # Validate inputs using enhanced utilities
        features = validate_numpy_array(features, "features")
        labels = validate_numpy_array(labels, "labels")
        
        # Check for numerical stability
        if not validate_numerical_stability(features):
            tprint("⚠️ Numerical stability issues detected, applying corrections", "WARNING")
            features = safe_array_operation(
                lambda: np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6),
                "Failed to correct numerical stability"
            )
        
        # Enhanced scaling with validation
        scaler = StandardScaler()
        features_scaled = safe_array_operation(
            lambda: scaler.fit_transform(features),
            "Failed to scale features"
        )
        
        # Calculate metrics with enhanced error handling
        silhouette = safe_numpy_operation(
            lambda: silhouette_score(features_scaled, labels),
            "Failed to calculate silhouette score"
        )
        
        davies_bouldin = safe_numpy_operation(
            lambda: davies_bouldin_score(features_scaled, labels),
            "Failed to calculate Davies-Bouldin score"
        )
        
        calinski_harabasz = safe_numpy_operation(
            lambda: calinski_harabasz_score(features_scaled, labels),
            "Failed to calculate Calinski-Harabasz score"
        )
        
        cv_score = calculate_cv_score(features_scaled, labels)
        
        # Calculate additional metrics using enhanced utilities
        feature_stats = calculate_statistical_metrics(features_scaled)
        label_stats = calculate_statistical_metrics(labels.astype(float))
        
        # Validate metrics
        silhouette = validate_finite(silhouette, "silhouette_score")
        davies_bouldin = validate_finite(davies_bouldin, "davies_bouldin_score")
        calinski_harabasz = validate_finite(calinski_harabasz, "calinski_harabasz_score")
        cv_score = validate_finite(cv_score, "cv_score")
        
        result = {
            "regime_type": regime_type,
            "silhouette_score": round(float(silhouette), 4),
            "davies_bouldin_score": round(float(davies_bouldin), 4),
            "calinski_harabasz_score": round(float(calinski_harabasz), 4),
            "cv_score": round(float(cv_score), 4),
            "feature_statistics": feature_stats,
            "label_statistics": label_stats,
            "interpretation": {
                "silhouette": interpret_silhouette(silhouette),
                "davies_bouldin": interpret_davies_bouldin(davies_bouldin),
                "cv_score": interpret_cv_score(cv_score),
            },
        }
        
        tprint(f"✅ Clustering metrics calculated for {regime_type}", "SUCCESS")
        return result
        
    except Exception as e:
        tprint(f"❌ Clustering metrics calculation failed: {e}", "ERROR")
        # Return fallback metrics
        return {
            "regime_type": regime_type,
            "silhouette_score": 0.0,
            "davies_bouldin_score": 10.0,
            "calinski_harabasz_score": 0.0,
            "cv_score": 0.0,
            "error": str(e),
            "interpretation": {
                "silhouette": "Error in calculation",
                "davies_bouldin": "Error in calculation",
                "cv_score": "Error in calculation",
            },
        }


def calculate_cv_score(features: np.ndarray, labels: np.ndarray) -> float:
    """Calculate coefficient of variation score matching the original heuristics."""
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
            std = np.std(feature_values)
            mean_abs = np.mean(np.abs(feature_values))
            if std > 0 and mean_abs > 0:
                feature_cvs.append(std / mean_abs)
        if feature_cvs:
            within_cv_scores.append(np.mean(feature_cvs))

    cluster_centers = []
    for label in unique_labels:
        cluster_mask = labels == label
        cluster_features = features[cluster_mask]
        if len(cluster_features) > 0:
            cluster_centers.append(np.mean(cluster_features, axis=0))
    if len(cluster_centers) > 1:
        cluster_centers = np.asarray(cluster_centers)
        between_cv = np.std(cluster_centers) / np.mean(np.abs(cluster_centers))
    else:
        between_cv = 0.0

    within_cv = float(np.mean(within_cv_scores)) if within_cv_scores else 0.0
    return float(0.6 * max(0.0, 1.0 - within_cv) + 0.4 * min(1.0, between_cv))


def interpret_silhouette(score: float) -> str:
    """Interpret silhouette score."""
    if score >= 0.7:
        return "Excellent clustering"
    if score >= 0.5:
        return "Good clustering"
    if score >= 0.3:
        return "Fair clustering"
    return "Poor clustering"


def interpret_davies_bouldin(score: float) -> str:
    """Interpret Davies-Bouldin score (lower is better)."""
    if score <= 0.5:
        return "Excellent separation"
    if score <= 1.0:
        return "Good separation"
    if score <= 2.0:
        return "Fair separation"
    return "Poor separation"


def interpret_cv_score(score: float) -> str:
    """Interpret coefficient of variation score."""
    if score >= 0.8:
        return "Excellent regime distinction"
    if score >= 0.6:
        return "Good regime distinction"
    if score >= 0.4:
        return "Fair regime distinction"
    return "Poor regime distinction"
