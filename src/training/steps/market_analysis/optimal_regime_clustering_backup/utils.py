"""
Optimal Regime Clustering Utilities

This module contains utility functions for cluster analysis, validation, and visualization.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
import warnings
import logging
from dataclasses import dataclass
from enum import Enum

# Import unified matrix operations for performance optimization
try:
    from src.utils.matrix_operations import get_unified_matrix_operations
    MATRIX_OPERATIONS_AVAILABLE = True
except ImportError:
    MATRIX_OPERATIONS_AVAILABLE = False

# Configure logging
logger = logging.getLogger(__name__)

class ClusterQualityMetric(Enum):
    """Enumeration of cluster quality metrics."""
    SILHOUETTE = "silhouette"
    CALINSKI_HARABASZ = "calinski_harabasz"
    DAVIES_BOULDIN = "davies_bouldin"
    COHERENCE = "coherence"
    SEPARATION = "separation"
    COVERAGE = "coverage"
    STABILITY = "stability"

@dataclass
class ClusterValidationResult:
    """Result of cluster validation."""
    is_valid: bool
    scores: Dict[str, float]
    warnings: List[str]
    recommendations: List[str]

@dataclass
class ClusterStatistics:
    """Statistics for cluster analysis."""
    n_clusters: int
    cluster_sizes: np.ndarray
    cluster_percentages: np.ndarray
    noise_percentage: float
    coverage_percentage: float
    mean_cluster_size: float
    std_cluster_size: float
    min_cluster_size: float
    max_cluster_size: float

def load_regime_data(data_path: str, config: Dict[str, Any]) -> pd.DataFrame:
    """Load regime discovery data.

    Args:
        data_path: Path to the regime data
        config: Configuration dictionary

    Returns:
        DataFrame with regime data
    """
    try:
        # Try to load from various sources
        if data_path.endswith('.parquet'):
            df = pd.read_parquet(data_path)
        elif data_path.endswith('.csv'):
            df = pd.read_csv(data_path)
        else:
            raise ValueError(f"Unsupported file format: {data_path}")

        logger.info(f"Loaded HMM regime data: {df.shape[0]} rows, {df.shape[1]} columns")
        return df

    except Exception as e:
        logger.error(f"Error loading HMM regime data from {data_path}: {e}")
        raise

def prepare_clustering_features(data: pd.DataFrame, config: Dict[str, Any]) -> Tuple[np.ndarray, pd.DataFrame]:
    """Prepare features for clustering.

    Args:
        data: Input data containing regime features
        config: Configuration dictionary

    Returns:
        Tuple of (features_array, feature_metadata)
    """
    try:
        # Debug: Check what config contains
        logger.info(f"Config received: {type(config)}, keys: {list(config.keys()) if isinstance(config, dict) else 'Not a dict'}")
        logger.info(f"Feature dimensions in config: {config.get('feature_dimensions', 'NOT_FOUND')}, type: {type(config.get('feature_dimensions', 'NOT_FOUND'))}")

        # Select relevant features based on 4D dimensions
        feature_dimensions = config.get('feature_dimensions', ['volume', 'volatility', 'momentum', 'trend'])

        # Ensure feature_dimensions is iterable
        if not hasattr(feature_dimensions, '__iter__') or isinstance(feature_dimensions, (str, int)):
            logger.warning(f"Feature dimensions is not iterable: {feature_dimensions} (type: {type(feature_dimensions)}), using default")
            feature_dimensions = ['volume', 'volatility', 'momentum', 'trend']

        feature_columns = []
        for dim in feature_dimensions:
            # Find columns that contain the dimension name
            dim_cols = [col for col in data.columns if dim.lower() in col.lower()]
            feature_columns.extend(dim_cols)

        if not feature_columns:
            # Fallback to all numeric columns
            feature_columns = data.select_dtypes(include=[np.number]).columns.tolist()

        features = data[feature_columns].copy()

        # Handle missing values
        features = features.fillna(features.median())

        # Scale features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        # Create metadata
        metadata = {
            'feature_columns': feature_columns,
            'scaler': scaler,
            'n_features': len(feature_columns),
            'n_samples': features_scaled.shape[0]
        }

        logger.info(f"Prepared clustering features: {features_scaled.shape[1]} features, {features_scaled.shape[0]} samples")
        logger.info(f"Features type: {type(features_scaled)}, shape: {features_scaled.shape}")
        logger.info(f"Metadata type: {type(metadata)}, keys: {list(metadata.keys()) if hasattr(metadata, 'keys') else 'Not a dict'}")
        logger.info(f"Feature columns type: {type(feature_columns)}, length: {len(feature_columns) if hasattr(feature_columns, '__len__') else 'Not len-able'}")
        return features_scaled, metadata

    except Exception as e:
        logger.error(f"Error preparing clustering features: {e}")
        raise

def calculate_cluster_statistics(labels: np.ndarray, config: Dict[str, Any]) -> ClusterStatistics:
    """Calculate cluster statistics.

    Args:
        labels: Cluster labels array
        config: Configuration dictionary

    Returns:
        ClusterStatistics object
    """
    try:
        # Cache for performance - avoid recalculating if called multiple times
        total_samples = len(labels)

        # Use numpy operations more efficiently
        unique_labels, counts = np.unique(labels, return_counts=True)

        # Separate noise points (usually labeled as -1)
        noise_mask = unique_labels == -1
        if noise_mask.any():
            noise_count = counts[noise_mask][0]
            noise_labels = unique_labels[~noise_mask]
            noise_counts = counts[~noise_mask]
        else:
            noise_count = 0
            noise_labels = unique_labels
            noise_counts = counts

        noise_percentage = noise_count / total_samples if total_samples > 0 else 0.0
        coverage_percentage = 1.0 - noise_percentage

        # Calculate cluster statistics using numpy operations (faster than Python loops)
        cluster_sizes = noise_counts
        cluster_percentages = cluster_sizes / total_samples if total_samples > 0 else np.zeros_like(cluster_sizes)

        # Use numpy aggregation functions for better performance
        mean_size = float(np.mean(cluster_sizes)) if len(cluster_sizes) > 0 else 0.0
        std_size = float(np.std(cluster_sizes)) if len(cluster_sizes) > 1 else 0.0
        min_size = float(np.min(cluster_sizes)) if len(cluster_sizes) > 0 else 0.0
        max_size = float(np.max(cluster_sizes)) if len(cluster_sizes) > 0 else 0.0

        stats = ClusterStatistics(
            n_clusters=len(noise_labels),
            cluster_sizes=cluster_sizes,
            cluster_percentages=cluster_percentages,
            noise_percentage=noise_percentage,
            coverage_percentage=coverage_percentage,
            mean_cluster_size=mean_size,
            std_cluster_size=std_size,
            min_cluster_size=min_size,
            max_cluster_size=max_size
        )

        # Only log every 10th call to reduce log spam during iterative optimization
        import time
        if not hasattr(calculate_cluster_statistics, '_last_log_time'):
            calculate_cluster_statistics._last_log_time = 0

        current_time = time.time()
        if current_time - calculate_cluster_statistics._last_log_time > 10:  # Log every 10 seconds
            logger.info(f"Cluster statistics: {stats.n_clusters} clusters, {noise_percentage:.3f} noise, {coverage_percentage:.3f} coverage")
            calculate_cluster_statistics._last_log_time = current_time

        return stats

    except Exception as e:
        logger.error(f"Error calculating cluster statistics: {e}")
        raise

def calculate_cluster_quality_metrics(features: np.ndarray, labels: np.ndarray,
                                      feature_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
    """Calculate comprehensive cluster quality metrics.

    Args:
        features: Feature matrix
        labels: Cluster labels

    Returns:
        Dictionary of quality metrics
    """
    try:
        metrics = {}

        # Remove noise points for quality calculations
        mask = labels != -1
        if mask.sum() > 0:
            clean_features = features[mask]
            clean_labels = labels[mask]

            # Calculate standard metrics
            if len(np.unique(clean_labels)) > 1:
                metrics['silhouette'] = silhouette_score(clean_features, clean_labels)
                metrics['calinski_harabasz'] = calinski_harabasz_score(clean_features, clean_labels)
                metrics['davies_bouldin'] = davies_bouldin_score(clean_features, clean_labels)
            else:
                metrics['silhouette'] = 0.0
                metrics['calinski_harabasz'] = 0.0
                metrics['davies_bouldin'] = float('inf')

            # Cluster-size coefficient of variation (CV) across non-noise clusters
            unique_clean = np.unique(clean_labels)
            if unique_clean.size > 1:
                size_counts = np.array([np.sum(clean_labels == lab) for lab in unique_clean], dtype=float)
                mean_count = float(np.mean(size_counts))
                if mean_count > 0:
                    metrics['cluster_size_coefficient_of_variation'] = float(np.std(size_counts) / (mean_count + 1e-12))
                else:
                    metrics['cluster_size_coefficient_of_variation'] = 0.0
            else:
                metrics['cluster_size_coefficient_of_variation'] = 0.0

            # Within-cluster feature CV (overall and by dimension) using unscaled features if available
            try:
                if feature_metadata is not None and 'scaler' in feature_metadata and 'feature_columns' in feature_metadata:
                    scaler = feature_metadata['scaler']
                    feature_columns = feature_metadata['feature_columns']
                    # Reconstruct unscaled features from StandardScaler parameters
                    if hasattr(scaler, 'scale_') and hasattr(scaler, 'mean_') and scaler.scale_ is not None:
                        unscaled_features = clean_features * scaler.scale_ + scaler.mean_
                    else:
                        # Fallback: if scaler params not available, use features as-is
                        unscaled_features = clean_features

                    # Build dimension index mapping
                    dim_to_indices: Dict[str, List[int]] = {
                        'volume': [], 'volatility': [], 'momentum': [], 'trend': []
                    }
                    for idx, col in enumerate(feature_columns):
                        col_l = str(col).lower()
                        if 'volume' in col_l:
                            dim_to_indices['volume'].append(idx)
                        if 'volatility' in col_l or 'std' in col_l:
                            dim_to_indices['volatility'].append(idx)
                        if 'momentum' in col_l:
                            dim_to_indices['momentum'].append(idx)
                        if 'trend' in col_l:
                            dim_to_indices['trend'].append(idx)

                    # Helper to compute per-feature CV within a cluster, with stability guards
                    def feature_cv(values: np.ndarray) -> float:
                        mean_v = float(np.mean(values))
                        std_v = float(np.std(values))
                        denom = abs(mean_v) if abs(mean_v) > 1e-12 else 1e-12
                        return float(std_v / denom)

                    unique_clusters = np.unique(clean_labels)
                    n_samples_total = unscaled_features.shape[0]

                    # Overall within-cluster feature CV (all features)
                    weighted_cv_sum = 0.0
                    weighted_count = 0
                    for lab in unique_clusters:
                        idx = clean_labels == lab
                        if np.sum(idx) <= 1:
                            continue
                        cluster_feats = unscaled_features[idx]
                        # Compute CV per feature then mean across features
                        cvs = []
                        for j in range(cluster_feats.shape[1]):
                            cvs.append(feature_cv(cluster_feats[:, j]))
                        cluster_cv = float(np.mean(cvs)) if cvs else 0.0
                        weight = float(np.sum(idx)) / max(1.0, n_samples_total)
                        weighted_cv_sum += cluster_cv * weight
                        weighted_count += weight

                    if weighted_count > 0:
                        metrics['within_cluster_feature_cv_overall'] = float(weighted_cv_sum / weighted_count)
                    else:
                        metrics['within_cluster_feature_cv_overall'] = 0.0

                    # By-dimension CVs
                    for dim_name, indices in dim_to_indices.items():
                        if not indices:
                            continue
                        weighted_cv_sum_dim = 0.0
                        weighted_count_dim = 0.0
                        for lab in unique_clusters:
                            idx = clean_labels == lab
                            if np.sum(idx) <= 1:
                                continue
                            cluster_feats_dim = unscaled_features[idx][:, indices]
                            cvs_dim = []
                            for j in range(cluster_feats_dim.shape[1]):
                                cvs_dim.append(feature_cv(cluster_feats_dim[:, j]))
                            cluster_cv_dim = float(np.mean(cvs_dim)) if cvs_dim else 0.0
                            weight = float(np.sum(idx)) / max(1.0, n_samples_total)
                            weighted_cv_sum_dim += cluster_cv_dim * weight
                            weighted_count_dim += weight
                        if weighted_count_dim > 0:
                            metrics[f'within_cluster_feature_cv_{dim_name}'] = float(weighted_cv_sum_dim / weighted_count_dim)
                        else:
                            metrics[f'within_cluster_feature_cv_{dim_name}'] = 0.0
            except Exception as cv_err:
                logger.warning(f"Error computing within-cluster feature CV metrics: {cv_err}")
        else:
            metrics['silhouette'] = 0.0
            metrics['calinski_harabasz'] = 0.0
            metrics['davies_bouldin'] = float('inf')

        logger.info(f"Cluster quality metrics: {metrics}")
        return metrics

    except Exception as e:
        logger.warning(f"Error calculating cluster quality metrics: {e}")
        return {
            'silhouette': 0.0,
            'calinski_harabasz': 0.0,
            'davies_bouldin': float('inf')
        }

def calculate_cluster_quality_metrics_optimized(features: np.ndarray, labels: np.ndarray,
                                               use_matrix_ops: bool = True) -> Dict[str, float]:
    """
    Calculate comprehensive cluster quality metrics using optimized matrix operations.

    This function uses unified matrix operations for better performance on large datasets,
    with automatic fallback to sklearn implementations.

    Args:
        features: Feature matrix
        labels: Cluster labels
        use_matrix_ops: Whether to use optimized matrix operations

    Returns:
        Dictionary of quality metrics
    """
    try:
        metrics = {}

        # Remove noise points for quality calculations
        mask = labels != -1
        if mask.sum() > 0:
            clean_features = features[mask]
            clean_labels = labels[mask]

            # Calculate standard metrics with optimization
            if len(np.unique(clean_labels)) > 1:
                if use_matrix_ops and MATRIX_OPERATIONS_AVAILABLE and clean_features.shape[0] > 1000:
                    # Use optimized implementations for large datasets
                    logger.info("🚀 Using optimized matrix operations for quality metrics (large dataset)")
                    metrics['silhouette'] = _calculate_silhouette_optimized(clean_features, clean_labels)
                    metrics['calinski_harabasz'] = _calculate_ch_score_optimized(clean_features, clean_labels)
                    metrics['davies_bouldin'] = _calculate_db_score_optimized(clean_features, clean_labels)
                else:
                    # Use sklearn implementations for smaller datasets
                    logger.info("📊 Using sklearn implementations for quality metrics (small dataset)")
                    metrics['silhouette'] = silhouette_score(clean_features, clean_labels)
                    metrics['calinski_harabasz'] = calinski_harabasz_score(clean_features, clean_labels)
                    metrics['davies_bouldin'] = davies_bouldin_score(clean_features, clean_labels)
            else:
                metrics['silhouette'] = 0.0
                metrics['calinski_harabasz'] = 0.0
                metrics['davies_bouldin'] = float('inf')
        else:
            metrics['silhouette'] = 0.0
            metrics['calinski_harabasz'] = 0.0
            metrics['davies_bouldin'] = float('inf')

        logger.info(f"Cluster quality metrics (optimized): {metrics}")
        return metrics

    except Exception as e:
        logger.warning(f"Error calculating optimized cluster quality metrics: {e}")
        # Fallback to original implementation
        logger.info("⚠️ Falling back to sklearn implementation")
        return calculate_cluster_quality_metrics(features, labels)

def _calculate_silhouette_optimized(features: np.ndarray, labels: np.ndarray) -> float:
    """
    Optimized Silhouette Score calculation using matrix operations.

    This implementation uses batch processing and vectorized operations
    to improve performance on large datasets.
    """
    try:
        if not MATRIX_OPERATIONS_AVAILABLE:
            return silhouette_score(features, labels)

        n_samples = len(features)
        if n_samples < 10:  # Too small for optimization
            return silhouette_score(features, labels)

        # Get unified matrix operations
        matrix_ops = get_unified_matrix_operations()

        # Calculate pairwise distances using optimized operations
        logger.info(f"🔄 Calculating optimized silhouette score for {n_samples} samples")

        # Use batch processing for large datasets
        if n_samples > 5000:
            return _calculate_silhouette_batched(features, labels, matrix_ops)
        else:
            return _calculate_silhouette_direct(features, labels, matrix_ops)

    except Exception as e:
        logger.warning(f"Optimized silhouette calculation failed: {e}")
        return silhouette_score(features, labels)

def _calculate_silhouette_batched(features: np.ndarray, labels: np.ndarray, matrix_ops) -> float:
    """Calculate silhouette score using batched processing."""
    try:
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)

        # Sample a subset for distance calculations (10% or max 1000 samples)
        sample_size = min(1000, int(0.1 * len(features)))
        indices = np.random.choice(len(features), sample_size, replace=False)
        sample_features = features[indices]
        sample_labels = labels[indices]

        # Calculate silhouette on sample
        if len(np.unique(sample_labels)) > 1:
            silhouette_sample = silhouette_score(sample_features, sample_labels)
            return float(silhouette_sample)
        else:
            return 0.0

    except Exception as e:
        logger.warning(f"Batched silhouette calculation failed: {e}")
        return 0.0

def _calculate_silhouette_direct(features: np.ndarray, labels: np.ndarray, matrix_ops) -> float:
    """Calculate silhouette score using direct matrix operations."""
    try:
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)

        # Calculate pairwise distances using matrix operations
        # Use cosine similarity for efficiency
        similarity_matrix = matrix_ops.calculate_pairwise_similarities(features, method='cosine')

        # Calculate silhouette score manually
        silhouette_values = []

        for i in range(len(features)):
            # Get same cluster points
            same_cluster = features[labels == labels[i]]
            if len(same_cluster) <= 1:
                continue

            # Calculate a(i) - mean distance to same cluster
            a_i = np.mean([1 - similarity_matrix[i, j] for j in range(len(features))
                          if labels[j] == labels[i] and j != i])

            # Calculate b(i) - mean distance to nearest cluster
            b_i = float('inf')
            for cluster_id in unique_labels:
                if cluster_id == labels[i]:
                    continue

                other_cluster = features[labels == cluster_id]
                if len(other_cluster) == 0:
                    continue

                # Calculate mean distance to other cluster
                distances = [1 - similarity_matrix[i, j] for j in range(len(features))
                           if labels[j] == cluster_id]
                if distances:
                    mean_distance = np.mean(distances)
                    b_i = min(b_i, mean_distance)

            # Calculate silhouette value
            if a_i < b_i:
                silhouette_val = 1 - a_i / b_i
            elif a_i > b_i:
                silhouette_val = b_i / a_i - 1
            else:
                silhouette_val = 0

            silhouette_values.append(silhouette_val)

        return float(np.mean(silhouette_values)) if silhouette_values else 0.0

    except Exception as e:
        logger.warning(f"Direct silhouette calculation failed: {e}")
        return 0.0

def _calculate_ch_score_optimized(features: np.ndarray, labels: np.ndarray) -> float:
    """
    Optimized Calinski-Harabasz Score calculation using matrix operations.

    This implementation uses vectorized operations for better performance.
    """
    try:
        if not MATRIX_OPERATIONS_AVAILABLE:
            return calinski_harabasz_score(features, labels)

        n_samples, n_features = features.shape
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)

        if n_clusters <= 1 or n_samples < n_clusters:
            return 0.0

        # Calculate cluster centers
        centers = np.zeros((n_clusters, n_features))
        for i, label in enumerate(unique_labels):
            centers[i] = np.mean(features[labels == label], axis=0)

        # Calculate overall mean
        overall_mean = np.mean(features, axis=0)

        # Calculate between-cluster dispersion (BC)
        bc = 0
        for i, center in enumerate(centers):
            cluster_size = np.sum(labels == unique_labels[i])
            bc += cluster_size * np.sum((center - overall_mean) ** 2)

        # Calculate within-cluster dispersion (WC)
        wc = 0
        for i, label in enumerate(unique_labels):
            cluster_points = features[labels == label]
            center = centers[i]
            wc += np.sum((cluster_points - center) ** 2)

        # Calculate CH score
        if wc == 0:
            return 0.0

        ch_score = (bc / (n_clusters - 1)) / (wc / (n_samples - n_clusters))
        return float(ch_score)

    except Exception as e:
        logger.warning(f"Optimized CH score calculation failed: {e}")
        return calinski_harabasz_score(features, labels)

def _calculate_db_score_optimized(features: np.ndarray, labels: np.ndarray) -> float:
    """
    Optimized Davies-Bouldin Score calculation using matrix operations.

    This implementation uses vectorized operations for better performance.
    """
    try:
        if not MATRIX_OPERATIONS_AVAILABLE:
            return davies_bouldin_score(features, labels)

        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)

        if n_clusters <= 1:
            return float('inf')

        # Calculate cluster centers and sizes
        centers = []
        cluster_sizes = []

        for label in unique_labels:
            cluster_points = features[labels == label]
            centers.append(np.mean(cluster_points, axis=0))
            cluster_sizes.append(len(cluster_points))

        centers = np.array(centers)

        # Calculate within-cluster dispersions (S_i)
        dispersions = []
        for i, label in enumerate(unique_labels):
            cluster_points = features[labels == label]
            center = centers[i]
            dispersion = np.mean(np.sum((cluster_points - center) ** 2, axis=1))
            dispersions.append(dispersion)

        # Calculate between-cluster distances (M_ij)
        max_db_score = 0
        for i in range(n_clusters):
            max_ratio = 0
            for j in range(n_clusters):
                if i != j:
                    # Calculate distance between centers
                    center_dist = np.linalg.norm(centers[i] - centers[j])
                    if center_dist == 0:
                        ratio = float('inf')
                    else:
                        ratio = (dispersions[i] + dispersions[j]) / center_dist

                    max_ratio = max(max_ratio, ratio)

            max_db_score = max(max_db_score, max_ratio)

        return float(max_db_score)

    except Exception as e:
        logger.warning(f"Optimized DB score calculation failed: {e}")
        return davies_bouldin_score(features, labels)

def validate_cluster_quality(stats: ClusterStatistics, quality_metrics: Dict[str, float], config: Dict[str, Any]) -> ClusterValidationResult:
    """Validate cluster quality against configuration thresholds.

    Args:
        stats: Cluster statistics
        quality_metrics: Quality metrics
        config: Configuration dictionary

    Returns:
        ClusterValidationResult
    """
    try:
        warnings = []
        recommendations = []
        is_valid = True

        # Check coverage - handle both flat and nested config structures
        coverage_pct = config.get('target_coverage_pct', 0.95)
        if stats.coverage_percentage < coverage_pct:
            warnings.append(f"Coverage {stats.coverage_percentage:.3f} below target {coverage_pct}")
            is_valid = False

        # Check noise percentage - handle both flat and nested config structures
        max_noise_pct = config.get('max_noise_pct', 0.05)
        if stats.noise_percentage > max_noise_pct:
            warnings.append(f"Noise percentage {stats.noise_percentage:.3f} exceeds limit {max_noise_pct}")
            is_valid = False

        # Check cluster size distribution - handle both flat and nested config structures
        min_size_pct = config.get('min_cluster_size_pct', 0.03)
        max_size_pct = config.get('max_cluster_size_pct', 0.08)

        if np.min(stats.cluster_percentages) < min_size_pct:
            warnings.append(f"Smallest cluster {np.min(stats.cluster_percentages):.3f} below minimum {min_size_pct}")
            is_valid = False

        if np.max(stats.cluster_percentages) > max_size_pct:
            warnings.append(f"Largest cluster {np.max(stats.cluster_percentages):.3f} exceeds maximum {max_size_pct}")
            is_valid = False

        # Check quality metrics - handle both flat and nested config structures
        quality_config = config.get('quality_metrics', config)
        min_silhouette = quality_config.get('min_silhouette_score', config.get('min_silhouette_score', 0.3))
        min_ch = quality_config.get('min_calinski_harabasz_score', config.get('min_calinski_harabasz_score', 100.0))
        max_db = quality_config.get('min_davies_bouldin_score', config.get('min_davies_bouldin_score', 1.5))

        if quality_metrics.get('silhouette', 0.0) < min_silhouette:
            warnings.append(f"Silhouette score {quality_metrics.get('silhouette', 0.0):.3f} below threshold {min_silhouette}")
            is_valid = False

        if quality_metrics.get('calinski_harabasz', 0.0) < min_ch:
            warnings.append(f"Calinski-Harabasz score {quality_metrics.get('calinski_harabasz', 0.0):.3f} below threshold {min_ch}")
            is_valid = False

        if quality_metrics.get('davies_bouldin', float('inf')) > max_db:
            warnings.append(f"Davies-Bouldin score {quality_metrics.get('davies_bouldin', float('inf')):.3f} above threshold {max_db}")
            is_valid = False

        # Generate recommendations
        if not is_valid:
            recommendations.append("Consider adjusting clustering parameters")
            recommendations.append("Review feature preprocessing")
            if stats.noise_percentage > 0.1:
                recommendations.append("Consider noise reduction techniques")

        result = ClusterValidationResult(
            is_valid=is_valid,
            scores=quality_metrics,
            warnings=warnings,
            recommendations=recommendations
        )

        logger.info(f"Cluster validation: {'VALID' if is_valid else 'INVALID'}")
        return result

    except Exception as e:
        logger.error(f"Error validating cluster quality: {e}")
        return ClusterValidationResult(
            is_valid=False,
            scores={},
            warnings=[str(e)],
            recommendations=["Error during validation"]
        )

def create_cluster_summary_report(stats: ClusterStatistics, quality_metrics: Dict[str, float], validation: ClusterValidationResult) -> Dict[str, Any]:
    """Create comprehensive cluster summary report.

    Args:
        stats: Cluster statistics
        quality_metrics: Quality metrics
        validation: Validation result

    Returns:
        Summary report dictionary
    """
    try:
        # Calculate sample counts accurately
        total_non_noise_samples = int(stats.cluster_sizes.sum()) if len(stats.cluster_sizes) > 0 else 0
        # Derive total_samples using coverage percentage while avoiding rounding drift
        if stats.coverage_percentage > 0:
            estimated_total = total_non_noise_samples / max(stats.coverage_percentage, 1e-12)
            # Round to nearest int for readability, then recompute noise exactly
            total_samples = int(round(estimated_total))
            noise_samples = max(0, total_samples - total_non_noise_samples)
        else:
            total_samples = total_non_noise_samples
            noise_samples = 0

        # Convert numpy arrays to lists only when needed (and use numpy's efficient conversion)
        cluster_sizes_list = stats.cluster_sizes.tolist() if len(stats.cluster_sizes) > 0 else []
        cluster_percentages_list = stats.cluster_percentages.tolist() if len(stats.cluster_percentages) > 0 else []

        report = {
            'summary': {
                'n_clusters': stats.n_clusters,
                'total_samples': total_samples,
                'noise_samples': noise_samples,
                'coverage_percentage': float(stats.coverage_percentage),
                'noise_percentage': float(stats.noise_percentage),
                'coverage_samples': int(total_non_noise_samples),
                'noise_samples': int(noise_samples),
                'is_valid': validation.is_valid
            },
            'cluster_distribution': {
                'sizes': cluster_sizes_list,
                'percentages': cluster_percentages_list,
                'size_statistics': {
                    'mean': float(stats.mean_cluster_size),
                    'std': float(stats.std_cluster_size),
                    'min': float(stats.min_cluster_size),
                    'max': float(stats.max_cluster_size)
                }
            },
            'quality_metrics': quality_metrics,
            'validation': {
                'warnings': validation.warnings,
                'recommendations': validation.recommendations
            }
        }

        # Only log when actually needed (not during iterative optimization)
        logger.debug("Created cluster summary report")
        return report

    except Exception as e:
        logger.error(f"Error creating cluster summary report: {e}")
        return {'error': str(e)}

def detect_outliers(features: np.ndarray, method: str = "isolation_forest", contamination: float = 0.1) -> np.ndarray:
    """Detect outliers using specified method.

    Args:
        features: Feature matrix
        method: Detection method
        contamination: Expected contamination rate

    Returns:
        Boolean array of outlier indicators
    """
    try:
        if method == "isolation_forest":
            from sklearn.ensemble import IsolationForest
            detector = IsolationForest(contamination=contamination, random_state=42)
            outliers = detector.fit_predict(features)
            outlier_mask = outliers == -1

        elif method == "local_outlier_factor":
            from sklearn.neighbors import LocalOutlierFactor
            detector = LocalOutlierFactor(contamination=contamination)
            outliers = detector.fit_predict(features)
            outlier_mask = outliers == -1

        elif method == "none":
            # No outlier detection
            outlier_mask = np.zeros(len(features), dtype=bool)

        else:
            raise ValueError(f"Unknown outlier detection method: {method}")

        logger.info(f"Detected {outlier_mask.sum()} outliers using {method}")
        return outlier_mask

    except Exception as e:
        logger.warning(f"Error detecting outliers: {e}")
        return np.zeros(len(features), dtype=bool)

def optimize_cluster_parameters(features: np.ndarray, config: Dict[str, Any]) -> Dict[str, Any]:
    """Optimize clustering parameters for best results.

    Args:
        features: Feature matrix
        config: Configuration dictionary

    Returns:
        Optimized parameters
    """
    try:
        logger.info("Optimizing cluster parameters...")

        # This is a simplified parameter optimization
        # In a full implementation, this would use grid search or Bayesian optimization

        optimized_params = {
            'min_cluster_size': max(50, int(len(features) * 0.001)),
            'min_samples': max(10, int(len(features) * 0.0005)),
            'cluster_selection_epsilon': 0.1
        }

        logger.info(f"Optimized parameters: {optimized_params}")
        return optimized_params

    except Exception as e:
        logger.warning(f"Error optimizing cluster parameters: {e}")
        return config.get('default_params', {})