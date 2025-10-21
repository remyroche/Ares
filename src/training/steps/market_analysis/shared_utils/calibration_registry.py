"""
Calibration Registry Module

Provides calibration and quality threshold management.
"""

from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


def get_quality_thresholds(algorithm: str = "default") -> Dict[str, float]:
    """
    Get quality thresholds for a specific algorithm.
    
    Args:
        algorithm: Algorithm name
        
    Returns:
        Dictionary of quality thresholds
    """
    thresholds = {
        "default": {
            "min_silhouette_score": 0.3,
            "min_calinski_harabasz_score": 100.0,
            "min_davies_bouldin_score": 1.5,
            "min_cluster_size": 10,
            "max_cluster_size": 1000,
            "min_separation": 0.1
        },
        "kmeans": {
            "min_silhouette_score": 0.2,
            "min_calinski_harabasz_score": 80.0,
            "min_davies_bouldin_score": 2.0,
            "min_cluster_size": 5,
            "max_cluster_size": 2000,
            "min_separation": 0.05
        },
        "gmm": {
            "min_silhouette_score": 0.25,
            "min_calinski_harabasz_score": 90.0,
            "min_davies_bouldin_score": 1.8,
            "min_cluster_size": 8,
            "max_cluster_size": 1500,
            "min_separation": 0.08
        },
        "hdbscan": {
            "min_silhouette_score": 0.15,
            "min_calinski_harabasz_score": 70.0,
            "min_davies_bouldin_score": 2.5,
            "min_cluster_size": 3,
            "max_cluster_size": 3000,
            "min_separation": 0.03
        }
    }
    
    return thresholds.get(algorithm, thresholds["default"])


def get_calibration_config(algorithm: str = "default") -> Dict[str, Any]:
    """
    Get calibration configuration for a specific algorithm.
    
    Args:
        algorithm: Algorithm name
        
    Returns:
        Dictionary of calibration configuration
    """
    configs = {
        "default": {
            "n_iter": 100,
            "tolerance": 1e-4,
            "random_state": 42,
            "n_init": 10
        },
        "kmeans": {
            "n_iter": 300,
            "tolerance": 1e-4,
            "random_state": 42,
            "n_init": 10
        },
        "gmm": {
            "n_iter": 200,
            "tolerance": 1e-6,
            "random_state": 42,
            "n_init": 5
        },
        "hdbscan": {
            "min_cluster_size": 5,
            "min_samples": 3,
            "cluster_selection_epsilon": 0.0,
            "metric": "euclidean"
        }
    }
    
    return configs.get(algorithm, configs["default"])


def validate_quality_metrics(metrics: Dict[str, float], algorithm: str = "default") -> bool:
    """
    Validate quality metrics against thresholds.
    
    Args:
        metrics: Dictionary of quality metrics
        algorithm: Algorithm name
        
    Returns:
        True if metrics meet quality thresholds
    """
    thresholds = get_quality_thresholds(algorithm)
    
    for metric, threshold in thresholds.items():
        if metric in metrics:
            if metric in ["min_davies_bouldin_score"]:
                # Lower is better for Davies-Bouldin score
                if metrics[metric] > threshold:
                    logger.warning(f"Quality metric {metric} failed: {metrics[metric]} > {threshold}")
                    return False
            else:
                # Higher is better for other metrics
                if metrics[metric] < threshold:
                    logger.warning(f"Quality metric {metric} failed: {metrics[metric]} < {threshold}")
                    return False
    
    return True


def get_algorithm_recommendations(metrics: Dict[str, float]) -> Dict[str, Any]:
    """
    Get algorithm recommendations based on metrics.
    
    Args:
        metrics: Dictionary of quality metrics
        
    Returns:
        Dictionary of recommendations
    """
    recommendations = {
        "algorithm": "kmeans",
        "reason": "Default recommendation",
        "confidence": 0.5
    }
    
    if "silhouette_score" in metrics:
        if metrics["silhouette_score"] > 0.5:
            recommendations["algorithm"] = "gmm"
            recommendations["reason"] = "High silhouette score suggests Gaussian mixture model"
            recommendations["confidence"] = 0.8
        elif metrics["silhouette_score"] < 0.2:
            recommendations["algorithm"] = "hdbscan"
            recommendations["reason"] = "Low silhouette score suggests density-based clustering"
            recommendations["confidence"] = 0.7
    
    return recommendations


def get_current_calibration(algorithm: str = "default") -> Dict[str, Any]:
    """
    Get current calibration settings for an algorithm.
    
    Args:
        algorithm: Algorithm name
        
    Returns:
        Dictionary of current calibration settings
    """
    return {
        "algorithm": algorithm,
        "quality_thresholds": get_quality_thresholds(algorithm),
        "calibration_config": get_calibration_config(algorithm),
        "status": "active",
        "last_updated": "2024-01-01T00:00:00Z"
    }


def update_quality_calibration(algorithm: str, new_thresholds: Dict[str, float]) -> bool:
    """
    Update quality calibration for an algorithm.
    
    Args:
        algorithm: Algorithm name
        new_thresholds: New quality thresholds
        
    Returns:
        True if update was successful
    """
    try:
        # In a real implementation, this would update persistent storage
        logger.info(f"Updated quality calibration for {algorithm}: {new_thresholds}")
        return True
    except Exception as e:
        logger.error(f"Failed to update quality calibration for {algorithm}: {e}")
        return False