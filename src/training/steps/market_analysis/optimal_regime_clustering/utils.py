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

def load_hmm_regime_data(data_path: str, config: Dict[str, Any]) -> pd.DataFrame:
    """Load HMM regime discovery data.

    Args:
        data_path: Path to the HMM regime data
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
        # Select relevant features based on 4D dimensions
        feature_columns = []
        for dim in config.get('feature_dimensions', ['volume', 'volatility', 'momentum', 'trend']):
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

        total_samples = len(labels)
        noise_percentage = noise_count / total_samples
        coverage_percentage = 1.0 - noise_percentage

        # Calculate cluster statistics
        cluster_sizes = noise_counts
        cluster_percentages = cluster_sizes / total_samples

        stats = ClusterStatistics(
            n_clusters=len(noise_labels),
            cluster_sizes=cluster_sizes,
            cluster_percentages=cluster_percentages,
            noise_percentage=noise_percentage,
            coverage_percentage=coverage_percentage,
            mean_cluster_size=np.mean(cluster_sizes),
            std_cluster_size=np.std(cluster_sizes),
            min_cluster_size=np.min(cluster_sizes),
            max_cluster_size=np.max(cluster_sizes)
        )

        logger.info(f"Cluster statistics: {stats.n_clusters} clusters, {noise_percentage".3f"} noise, {coverage_percentage".3f"} coverage")
        return stats

    except Exception as e:
        logger.error(f"Error calculating cluster statistics: {e}")
        raise

def calculate_cluster_quality_metrics(features: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
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

        # Check coverage
        if stats.coverage_percentage < config.get('target_coverage_pct', 0.95):
            warnings.append(f"Coverage {stats.coverage_percentage".3f"} below target {config.get('target_coverage_pct', 0.95)}")
            is_valid = False

        # Check noise percentage
        if stats.noise_percentage > config.get('max_noise_pct', 0.05):
            warnings.append(f"Noise percentage {stats.noise_percentage".3f"} exceeds limit {config.get('max_noise_pct', 0.05)}")
            is_valid = False

        # Check cluster size distribution
        min_size_pct = config.get('min_cluster_size_pct', 0.03)
        max_size_pct = config.get('max_cluster_size_pct', 0.08)

        if np.min(stats.cluster_percentages) < min_size_pct:
            warnings.append(f"Smallest cluster {np.min(stats.cluster_percentages)".3f"} below minimum {min_size_pct}")
            is_valid = False

        if np.max(stats.cluster_percentages) > max_size_pct:
            warnings.append(f"Largest cluster {np.max(stats.cluster_percentages)".3f"} exceeds maximum {max_size_pct}")
            is_valid = False

        # Check quality metrics
        min_silhouette = config.get('min_silhouette_score', 0.3)
        min_ch = config.get('min_calinski_harabasz_score', 100.0)
        max_db = config.get('min_davies_bouldin_score', 1.5)

        if quality_metrics.get('silhouette', 0.0) < min_silhouette:
            warnings.append(f"Silhouette score {quality_metrics.get('silhouette', 0.0)".3f"} below threshold {min_silhouette}")
            is_valid = False

        if quality_metrics.get('calinski_harabasz', 0.0) < min_ch:
            warnings.append(f"Calinski-Harabasz score {quality_metrics.get('calinski_harabasz', 0.0)".3f"} below threshold {min_ch}")
            is_valid = False

        if quality_metrics.get('davies_bouldin', float('inf')) > max_db:
            warnings.append(f"Davies-Bouldin score {quality_metrics.get('davies_bouldin', float('inf'))".3f"} above threshold {max_db}")
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
        report = {
            'summary': {
                'n_clusters': stats.n_clusters,
                'total_samples': int(stats.cluster_sizes.sum() + stats.noise_percentage * stats.cluster_sizes.sum() / (1 - stats.noise_percentage)),
                'noise_samples': int(stats.noise_percentage * stats.cluster_sizes.sum() / (1 - stats.noise_percentage)),
                'coverage_percentage': stats.coverage_percentage,
                'noise_percentage': stats.noise_percentage,
                'is_valid': validation.is_valid
            },
            'cluster_distribution': {
                'sizes': stats.cluster_sizes.tolist(),
                'percentages': stats.cluster_percentages.tolist(),
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

        logger.info("Created cluster summary report")
        return report

    except Exception as e:
        logger.error(f"Error creating cluster summary report: {e}")
        return {'error': str(e)}

def bootstrap_cluster_stability(features: np.ndarray, labels: np.ndarray, n_iterations: int = 100) -> float:
    """Calculate cluster stability using bootstrap sampling.

    Args:
        features: Feature matrix
        labels: Cluster labels
        n_iterations: Number of bootstrap iterations

    Returns:
        Stability score (0-1)
    """
    try:
        from sklearn.utils import resample

        stability_scores = []

        for i in range(n_iterations):
            # Bootstrap sample
            indices = resample(np.arange(len(features)), replace=True, random_state=i)
            sample_features = features[indices]
            sample_labels = labels[indices]

            # Remove noise points
            mask = sample_labels != -1
            if mask.sum() > 10:  # Need minimum samples
                stability_scores.append(silhouette_score(sample_features[mask], sample_labels[mask]))

        if stability_scores:
            stability = np.mean(stability_scores)
        else:
            stability = 0.0

        logger.info(f"Bootstrap stability score: {stability".3f"}")
        return stability

    except Exception as e:
        logger.warning(f"Error calculating bootstrap stability: {e}")
        return 0.0

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